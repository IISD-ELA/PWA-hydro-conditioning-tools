"""Step 0 downstream-half regression — picks up from Interim, runs to outputs.

Why "downstream half":
  Thomas's dataset is missing the raw LiDAR .tif inputs to ``merge_rasters``.
  We cannot regenerate the merge from scratch. However, the merged + resampled
  DEM lives in ``Interim/merged_average_dem_filled_resample_5m.tif``, so we
  can re-run the rest of Step 0 starting from that artifact:

      [resampled DEM]                                 ←── starting point (have)
            │
            ▼
      gen_depressions_raster(DEM, NHN_proj, output)
            │  (fill_burn → raster_calculator → conditional_evaluation)
            ▼
      [depressions raster .tif]
            │
            ▼
      calc_depression_depths(CLRH_proj_lidar, watershed, deps_raster, gdf, processed)
            │  (vector_polygons_to_raster → gdalwarp align → zonal_statistics → join)
            ▼
      [CLRH_basins_depression_depths.shp]             ←── compare to baseline

What this test proves (and doesn't):
  * PROVES: ``gen_depressions_raster`` + ``calc_depression_depths`` produce
    outputs matching the legacy baseline values (within numerical tolerance —
    sha256 will differ across GDAL versions, so we compare the value
    statistics committed in grassmere-baseline.json instead).
  * PROVES: WBT + gdalwarp + the new context-managed/error-checked wrappers
    work end-to-end on real grassmere data.
  * DOES NOT PROVE: the upstream half (merge_rasters, project_subbasins_to_lidar,
    clip_lidar_to_shapefile, resample_lidar_raster, project_subbasins_to_nhn,
    clip_nhn_to_watershed, append_culvert_lines). That's blocked on Thomas's
    raw LiDAR + NHN .shp.

Marked ``regression`` AND ``slow`` — fill_burn on the grassmere DEM takes
~5-10 minutes locally. To skip on a fast dev loop::

    pytest -m "not slow"

To run only this test::

    pytest tests/regression/test_step0_downstream.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pytest
import rasterio

from pwa_tools.depression import calc_depression_depths, gen_depressions_raster


# Loose tolerance — small numerical differences are expected across:
#   - GDAL versions (slightly different fill-burn outputs)
#   - WBT versions (different zonal_statistics rounding)
#   - System architectures (float ordering)
# We assert order-of-magnitude agreement, not bit identity.
_REL_TOL = 5e-2  # 5%
_VALUE_REL_TOL = 1e-3  # tighter for derived statistics


@pytest.fixture(scope="module")
def downstream_inputs(grassmere_data_dir: Path, tmp_path_factory) -> dict:
    """Stage Interim files into a fresh tmp tree and return paths.

    Why copy: ``gen_depressions_raster`` writes intermediate ``_FillBurn``
    and ``_FillBurn_Deps`` files alongside the input. We don't want to
    pollute ``data/HydroConditioning/Interim/`` (which holds the
    legacy run's artifacts).
    """
    interim = grassmere_data_dir / "HydroConditioning" / "Interim"
    required = {
        "lidar_resampled": "merged_average_dem_filled_resample_5m.tif",
        "nhn_projected_shp": "NHN_05OJ001_4_0_HD_SLWATER_1_clip_projected_EPSG3158.shp",
        "clrh_projected_shp": "finalcat_info_v1-0_projected_EPSG3158.shp",
    }
    missing = [name for name in required.values() if not (interim / name).exists()]
    if missing:
        pytest.skip(f"Missing Interim inputs for downstream e2e: {missing}")

    tmp = tmp_path_factory.mktemp("step0_downstream")
    interim_copy = tmp / "Interim"
    interim_copy.mkdir()
    processed = tmp / "Processed"
    processed.mkdir()

    # Copy LiDAR raster
    shutil.copy2(interim / required["lidar_resampled"], interim_copy / required["lidar_resampled"])
    # Copy all shapefile sidecars (.shp + .shx + .dbf + .prj + .cpg)
    for which in ("nhn_projected_shp", "clrh_projected_shp"):
        stem = required[which].rsplit(".", 1)[0]
        for sibling in interim.glob(f"{stem}.*"):
            shutil.copy2(sibling, interim_copy / sibling.name)

    return {
        "lidar_resampled": interim_copy / required["lidar_resampled"],
        "nhn_projected": interim_copy / required["nhn_projected_shp"],
        "clrh_projected": interim_copy / required["clrh_projected_shp"],
        "processed": processed,
    }


@pytest.fixture(scope="module")
def regenerated_outputs(downstream_inputs: dict) -> dict:
    """Run gen_depressions_raster + calc_depression_depths once per session."""
    deps_output = (
        downstream_inputs["lidar_resampled"].parent
        / "merged_average_dem_filled_resample_5m_FillBurn_Deps_Corr.tif"
    )

    deps_raster_path = gen_depressions_raster(
        lidar_resampled_path=downstream_inputs["lidar_resampled"],
        nhn_projected_path=downstream_inputs["nhn_projected"],
        output_path=deps_output,
    )

    clrh_gdf = gpd.read_file(downstream_inputs["clrh_projected"])
    deps_shp_path = calc_depression_depths(
        clrh_proj_lidar_path=downstream_inputs["clrh_projected"],
        watershed_name="grassmere",
        depressions_raster_path=deps_raster_path,
        clrh_gdf=clrh_gdf,
        processed_dir=downstream_inputs["processed"],
    )

    return {
        "deps_raster": deps_raster_path,
        "deps_shp": deps_shp_path,
    }


@pytest.mark.regression
@pytest.mark.slow
def test_regenerated_depression_raster_matches_baseline(
    regenerated_outputs: dict, baseline_manifest: dict,
) -> None:
    """The newly-generated depressions raster matches baseline value stats."""
    expected = baseline_manifest["samples"]["depression_raster"]

    with rasterio.open(regenerated_outputs["deps_raster"]) as src:
        assert src.crs.to_string() == expected["crs"]
        assert src.height == expected["height"]
        assert src.width == expected["width"]

        n_nonzero = 0
        max_v = float("-inf")
        global_count = 0
        global_sum = 0.0
        for _ji, window in src.block_windows(1):
            arr = src.read(1, window=window)
            valid = arr != src.nodata
            global_count += int(valid.sum())
            if valid.any():
                global_sum += float(arr[valid].sum())
            mask = valid & (arr != 0)
            n_nonzero += int(mask.sum())
            if mask.any():
                max_v = max(max_v, float(arr[mask].max()))

        global_mean = global_sum / global_count if global_count else 0.0

    # Loose tolerance — fill_burn outputs vary slightly across GDAL/WBT versions.
    assert n_nonzero == pytest.approx(expected["values"]["n_nonzero"], rel=_REL_TOL)
    assert max_v == pytest.approx(expected["values"]["max"], rel=_REL_TOL)
    assert global_mean == pytest.approx(expected["values"]["mean"], rel=_REL_TOL)


@pytest.mark.regression
@pytest.mark.slow
def test_regenerated_depression_depths_shp_matches_baseline(
    regenerated_outputs: dict, baseline_manifest: dict,
) -> None:
    """The newly-generated depression-depths shapefile matches baseline stats."""
    expected = baseline_manifest["samples"]["depression_depths"]

    gdf = gpd.read_file(regenerated_outputs["deps_shp"])

    assert len(gdf) == expected["n_features"]
    # The new file uses Deps_Depth (mm) and Deps_Vol_m (m³) — shapefile-truncated
    # versions of Deps_Depth_mm and Deps_Vol_m3 from the source code.
    assert "Deps_Depth" in gdf.columns
    assert "Deps_Vol_m" in gdf.columns

    actual = gdf["Deps_Depth"]
    expected_d = expected["deps_depth_mm"]
    # Tighter tolerance on aggregate stats — these are derived from the
    # zonal_statistics output which is robust to small raster differences.
    assert actual.min() == pytest.approx(expected_d["min"], rel=_VALUE_REL_TOL)
    assert actual.max() == pytest.approx(expected_d["max"], rel=_VALUE_REL_TOL)
    assert actual.mean() == pytest.approx(expected_d["mean"], rel=_VALUE_REL_TOL)
    assert actual.median() == pytest.approx(expected_d["median"], rel=_VALUE_REL_TOL)
