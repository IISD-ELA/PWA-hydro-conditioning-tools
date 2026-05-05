"""Path A regression: existing Processed/ outputs match baseline samples.

Read the already-existing legacy run in
``data/HydroConditioning/Processed/`` via the new pwa_tools API and the
standard geo libraries, then compare against the value samples committed
in ``grassmere-baseline.json``.

What this proves:
  * The new package's I/O paths (``read_shapefile``, rasterio context
    managers, pandas readers) work on real grassmere data.
  * The committed baseline samples haven't drifted from what the legacy
    pipeline produced.
  * Schemas (column lists, raster CRS / dimensions) are stable.

What this does NOT prove:
  * That re-running the new pipeline produces these outputs.
    (That's Path B — see test_step0_downstream.py.)

Marked ``regression`` so it auto-skips when the data isn't on disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import rasterio

from pwa_tools.io.shapefile import read_shapefile


# Loose tolerance — geopandas/pandas may compute summary stats with
# slightly different numerical paths than whatever produced the
# baseline. We're comparing apples to apples in shape and within
# reasonable scientific precision, not chasing bit-identical floats.
_REL_TOL = 1e-4


@pytest.mark.regression
def test_aspect_ratios_csv_matches_baseline(
    grassmere_data_dir: Path, baseline_manifest: dict,
) -> None:
    csv_path = grassmere_data_dir / "HydroConditioning" / "Processed" / "aspect_ratios.csv"
    if not csv_path.exists():
        pytest.skip(f"Expected baseline file missing: {csv_path}")

    expected = baseline_manifest["samples"]["aspect_ratios"]
    df = pd.read_csv(csv_path)

    assert list(df.columns) == expected["columns"]
    assert len(df) == expected["n_rows"]

    actual = df["Aspect_Ratio"]
    assert actual.min() == pytest.approx(expected["aspect_ratio"]["min"], rel=_REL_TOL)
    assert actual.mean() == pytest.approx(expected["aspect_ratio"]["mean"], rel=_REL_TOL)
    assert actual.max() == pytest.approx(expected["aspect_ratio"]["max"], rel=_REL_TOL)


@pytest.mark.regression
def test_depression_depths_shp_matches_baseline_via_read_shapefile(
    grassmere_data_dir: Path, baseline_manifest: dict,
) -> None:
    """Validates the legacy depression-depths shapefile loads cleanly via the
    new ``read_shapefile`` and matches the baseline samples."""
    shp_path = (
        grassmere_data_dir / "HydroConditioning" / "Processed"
        / "CLRH_basins_depression_depths.shp"
    )
    if not shp_path.exists():
        pytest.skip(f"Expected baseline file missing: {shp_path}")

    expected = baseline_manifest["samples"]["depression_depths"]

    # Use the new package API — proves read_shapefile handles real grassmere
    # data, not just synthetic fixtures.
    gdf = read_shapefile(shp_path, target_crs=None)

    # Schema parity (set comparison — the baseline ordering follows
    # geopandas's column-discovery order, which is stable but easier to
    # assert on as a set so future column additions surface as a diff).
    assert set(gdf.columns) == set(expected["columns"])
    assert len(gdf) == expected["n_features"]

    # Value parity for the depression columns this package produces.
    # Note: shapefile format truncates column names to 10 chars, so
    # Deps_Depth_mm → Deps_Depth and Deps_Vol_m3 → Deps_Vol_m on disk.
    # The baseline samples key is "deps_depth_mm" because the values
    # are in mm; the column itself just stores the magnitude.
    actual = gdf["Deps_Depth"]
    expected_d = expected["deps_depth_mm"]
    assert actual.min() == pytest.approx(expected_d["min"], rel=_REL_TOL)
    assert actual.max() == pytest.approx(expected_d["max"], rel=_REL_TOL)
    assert actual.mean() == pytest.approx(expected_d["mean"], rel=_REL_TOL)
    assert actual.median() == pytest.approx(expected_d["median"], rel=_REL_TOL)
    assert actual.std() == pytest.approx(expected_d["std"], rel=_REL_TOL)
    # First-5 fingerprint catches row-order drift that summary stats miss.
    assert list(actual.head(5)) == pytest.approx(expected_d["first_5"], rel=_REL_TOL)

    actual_v = gdf["Deps_Vol_m"]
    expected_v = expected["deps_vol_m3"]
    assert actual_v.min() == pytest.approx(expected_v["min"], rel=_REL_TOL)
    assert actual_v.max() == pytest.approx(expected_v["max"], rel=_REL_TOL)
    assert actual_v.mean() == pytest.approx(expected_v["mean"], rel=_REL_TOL)
    assert actual_v.median() == pytest.approx(expected_v["median"], rel=_REL_TOL)


@pytest.mark.regression
def test_depression_raster_matches_baseline(
    grassmere_data_dir: Path, baseline_manifest: dict,
) -> None:
    raster_path = (
        grassmere_data_dir / "HydroConditioning" / "Processed"
        / "merged_average_dem_filled_clip_resample_5m_FillBurn_Deps_Corr.tif"
    )
    if not raster_path.exists():
        pytest.skip(f"Expected baseline file missing: {raster_path}")

    expected = baseline_manifest["samples"]["depression_raster"]

    with rasterio.open(raster_path) as src:
        assert src.crs.to_string() == expected["crs"]
        assert src.height == expected["height"]
        assert src.width == expected["width"]
        assert src.nodata == pytest.approx(expected["nodata"])

        # Stream the raster in windows so we don't load 218 MB into memory
        # for a small statistics check.
        import numpy as np

        n_nonzero = 0
        sum_v = 0.0
        max_v = float("-inf")
        min_v = float("inf")
        for _ji, window in src.block_windows(1):
            arr = src.read(1, window=window)
            mask = (arr != src.nodata) & (arr != 0)
            n_nonzero += int(mask.sum())
            if mask.any():
                values = arr[mask]
                sum_v += float(values.sum())
                max_v = max(max_v, float(values.max()))
                # min over nonzero is uninformative (deps are non-negative
                # and zeros dominate); baseline records min=0 over all
                # valid pixels including zeros.
            # min over all valid (including 0)
            valid = arr != src.nodata
            if valid.any():
                min_v = min(min_v, float(arr[valid].min()))

        # Mean is over nonzero values per the baseline convention.
        # Reconstruct: baseline says n_nonzero = 5904520 and mean ~0.06,
        # which only makes sense if the mean is the GLOBAL mean over all
        # valid pixels, not just nonzero. Use the global one.
        # Re-run for global mean using the manifest's definition.
        global_count = 0
        global_sum = 0.0
        for _ji, window in src.block_windows(1):
            arr = src.read(1, window=window)
            valid = arr != src.nodata
            global_count += int(valid.sum())
            global_sum += float(arr[valid].sum())
        global_mean = global_sum / global_count if global_count else 0.0

    assert n_nonzero == expected["values"]["n_nonzero"]
    assert max_v == pytest.approx(expected["values"]["max"], rel=_REL_TOL)
    assert min_v == pytest.approx(expected["values"]["min"], abs=1e-6)
    assert global_mean == pytest.approx(expected["values"]["mean"], rel=_REL_TOL)
