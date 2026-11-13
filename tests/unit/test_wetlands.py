"""Unit tests for pwa_tools.wetlands — catchment size estimates and threshold parameter behaviour.

This module contains unit tests for the wetlands functionality in pwa_tools. This includes testing of the catchment size estimates and the behaviour of threshold parameters.

These tests exercise the optional depth_llim / area_llim / volume_llim
arguments added to gen_wetland_polygons without touching the filesystem
beyond what pytest's tmp_path fixture provides.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, box
from tests.unit.test_depression import _make_fake_wbt
from unittest.mock import MagicMock, patch

from pwa_tools import wetlands
from pwa_tools.wetlands import (
    _AREA_THRESHOLD_M2,
    _DEPTH_THRESHOLD_M,
    _SNAP_DIST_M,
    _VOLUME_THRESHOLD_M3,
    gen_wetland_catchment_polygons,
    gen_wetland_polygons,
    gen_direction_raster,
    gen_accum_raster
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_depression_raster(
    path: Path,
    data: np.ndarray,
    pixel_size_m: float = 1.0,
    nodata: float = -9999.0,
) -> Path:
    """Write a single-band float32 raster for testing."""
    height, width = data.shape
    transform = from_bounds(0, 0, width * pixel_size_m, height * pixel_size_m, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs="EPSG:32614",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data.astype(np.float32), 1)
    return path


def _write_wetlands_shapefile(
    path: Path, records: list[tuple[int, object]], crs: str = "EPSG:32614",
) -> Path:
    """Write a minimal wetland-polygons shapefile with a wetland_id column."""
    ids, geoms = zip(*records)
    gdf = gpd.GeoDataFrame({"wetland_id": list(ids), "geometry": list(geoms)}, crs=crs)
    gdf.to_file(path)
    return path


def _write_points_shapefile(
    path: Path, coords: list[tuple[float, float]], crs: str = "EPSG:32614",
) -> Path:
    """Write a minimal points shapefile — no attributes required."""
    gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs=crs)
    gdf.to_file(path)
    return path


# ---------------------------------------------------------------------------
# Flow routing raster pre-processing
# ---------------------------------------------------------------------------

def test_gen_direction_raster(tmp_path: Path) -> None:
    """Step 10c: gen_direction_raster() must invoke WBT d8_pointer and return the output path."""
    # Arrange: a flat DEM and the output location the function under test should write to
    dem_path = _write_depression_raster(tmp_path / "dem.tif", np.ones((10, 10)))
    output_path = tmp_path

    fake_wbt = _make_fake_wbt()

    # Stub d8_pointer to write a raster to whatever output path it's called with
    def _run_d8_pointer(**kwargs):
        _write_depression_raster(Path(kwargs["output"]), np.ones((10, 10)))
        return 0

    fake_wbt.d8_pointer.side_effect = _run_d8_pointer

    # Replace the real WBT session context manager with our fake so no external tool runs
    with patch("pwa_tools.wetlands.wbt_session") as mock_session:
        mock_session.return_value.__enter__ = lambda s: fake_wbt
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        result_path = gen_direction_raster(dem_path, output_path)

    # Assert: WBT was invoked exactly once and the expected output file was produced
    fake_wbt.d8_pointer.assert_called_once()
    assert result_path == output_path / "dem_d8.tif"
    assert output_path.exists()




def test_gen_accum_raster(tmp_path: Path) -> None:
    """Step 10d: gen_accum_raster() must invoke WBT d8_flow_accum on the direction raster."""
    # Arrange: a flow-direction raster as input and the output location for accumulation
    direction_path = _write_depression_raster(
        tmp_path / "direction.tif", np.random.uniform(1, 100, (10, 10))
    )
    output_path = tmp_path

    fake_wbt = _make_fake_wbt()

    # Stub d8_flow_accum to write a known constant raster so the result is checkable
    def _run_d8_flow_accum(**kwargs):
        _write_depression_raster(Path(kwargs["output"]), np.full((10, 10), 5.0))
        return 0

    fake_wbt.d8_flow_accumulation.side_effect = _run_d8_flow_accum

    # Replace the real WBT session context manager with our fake so no external tool runs
    with patch("pwa_tools.wetlands.wbt_session") as mock_session:
        mock_session.return_value.__enter__ = lambda s: fake_wbt
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        result_path = gen_accum_raster(direction_path, output_path)

    # Assert: WBT was invoked exactly once and the output raster carries the stubbed value
    fake_wbt.d8_flow_accumulation.assert_called_once()
    assert result_path == output_path / (direction_path.stem + "_accum.tif")
    with rasterio.open(result_path) as src:
        assert src.read(1).max() == pytest.approx(5.0)



# ---------------------------------------------------------------------------
# Default thresholds — signature exposes correct defaults
# ---------------------------------------------------------------------------


def test_signature_defaults_match_module_constants() -> None:
    """The three optional parameters must default to the module-level constants
    so existing callers (and the runner) are unaffected when thresholds are
    omitted."""
    import inspect

    sig = inspect.signature(gen_wetland_polygons)
    assert sig.parameters["depth_llim"].default == _DEPTH_THRESHOLD_M
    assert sig.parameters["area_llim"].default == _AREA_THRESHOLD_M2
    assert sig.parameters["volume_llim"].default == _VOLUME_THRESHOLD_M3


# ---------------------------------------------------------------------------
# Custom depth_llim
# ---------------------------------------------------------------------------


def test_depth_llim_filters_shallow_depressions(tmp_path: Path) -> None:
    """Pixels below depth_llim must be excluded even if they would pass
    the default threshold."""
    # 5x5 raster: one 3x3 block at depth 0.05 m (below default 0.1 m),
    # one isolated pixel at depth 0.5 m (passes both thresholds if area/vol ok).
    data = np.zeros((10, 10))
    # Large block at 0.05 m depth — passes if depth_llim=0.01, excluded at default
    data[1:8, 1:8] = 0.05
    raster_path = _write_depression_raster(tmp_path / "dep.tif", data, pixel_size_m=10.0)

    # With default depth_llim (0.1 m): shallow block is filtered out → no polygons
    _, gdf_default = gen_wetland_polygons(raster_path, tmp_path / "out_default")
    assert len(gdf_default) == 0, "default depth_llim should filter the shallow block"

    # With depth_llim=0.01: shallow block passes → area = 49 * 100 = 4900 m²,
    # volume = 49 * 100 * 0.05 = 245 m³ → both pass default area/vol thresholds
    _, gdf_custom = gen_wetland_polygons(
        raster_path, tmp_path / "out_custom", depth_llim=0.01
    )
    assert len(gdf_custom) == 1, "lowered depth_llim should admit the shallow block"


# ---------------------------------------------------------------------------
# Custom area_llim
# ---------------------------------------------------------------------------


def test_area_llim_filters_small_depressions(tmp_path: Path) -> None:
    """A block whose area is below area_llim must be excluded."""
    # 5x5 block at 1 m depth with 1 m pixels → area = 25 m², volume = 25 m³
    # Default area threshold is 4000 m² so it's normally filtered out.
    data = np.zeros((20, 20))
    data[5:10, 5:10] = 1.0
    raster_path = _write_depression_raster(tmp_path / "dep.tif", data, pixel_size_m=1.0)

    # Default area_llim (4000 m²): small block excluded
    _, gdf_default = gen_wetland_polygons(raster_path, tmp_path / "out_default")
    assert len(gdf_default) == 0, "default area_llim should filter the small block"

    # Lowered area_llim=10 m²: block area 25 m² passes, volume 25 m³ > 30? No, 25 < 30
    # So set volume_llim=10 as well to let it through
    _, gdf_custom = gen_wetland_polygons(
        raster_path, tmp_path / "out_custom", area_llim=10.0, volume_llim=10.0
    )
    assert len(gdf_custom) == 1, "lowered area_llim+volume_llim should admit the small block"


# ---------------------------------------------------------------------------
# Custom volume_llim
# ---------------------------------------------------------------------------


def test_volume_llim_filters_shallow_large_depressions(tmp_path: Path) -> None:
    """A block with enough area but low volume must be excluded when volume_llim
    is raised above its actual volume."""
    # 100x100 block at 0.2 m depth with 1 m pixels →
    # area = 10000 m² (> 4000), volume = 10000 * 0.2 = 2000 m³ (> 30)
    # Raise volume_llim to 5000 to force exclusion.
    data = np.zeros((120, 120))
    data[10:110, 10:110] = 0.2
    raster_path = _write_depression_raster(tmp_path / "dep.tif", data, pixel_size_m=1.0)

    # Default: included
    _, gdf_default = gen_wetland_polygons(raster_path, tmp_path / "out_default")
    assert len(gdf_default) == 1, "default thresholds should admit this block"

    # Raised volume_llim: excluded
    _, gdf_raised = gen_wetland_polygons(
        raster_path, tmp_path / "out_raised", volume_llim=5000.0
    )
    assert len(gdf_raised) == 0, "raised volume_llim should exclude the low-volume block"


# ---------------------------------------------------------------------------
# All three thresholds together
# ---------------------------------------------------------------------------


def test_all_three_thresholds_respected_simultaneously(tmp_path: Path) -> None:
    """Passing all three custom thresholds at once must work without conflicts."""
    data = np.zeros((30, 30))
    data[5:25, 5:25] = 0.5  # 20x20 block at 0.5 m → area=400 m², vol=200 m³ (1m pixels)
    raster_path = _write_depression_raster(tmp_path / "dep.tif", data, pixel_size_m=1.0)

    _, gdf = gen_wetland_polygons(
        raster_path,
        tmp_path / "out",
        depth_llim=0.1,
        area_llim=100.0,
        volume_llim=50.0,
    )
    assert len(gdf) == 1


# ---------------------------------------------------------------------------
# MultiPolygon merging — diagonal-only-touch pattern
# ---------------------------------------------------------------------------


def test_diagonal_touch_yields_one_multipolygon_row(tmp_path: Path) -> None:
    """Two blocks joined only diagonally must produce one row with a MultiPolygon.

    scipy.ndimage.label uses 8-connectivity, so diagonally-adjacent pixels are
    one component (same label).  rasterio.features.shapes uses 4-connectivity,
    so it returns two separate polygons for that component.  The dissolve step
    must collapse them into a single MultiPolygon record.
    """
    # Two 10x10 blocks that share only a diagonal corner at (14,14)/(15,15).
    # pixel_size=10 m → each block: area=10000 m², volume=10000 m³ (passes thresholds).
    data = np.zeros((40, 40))
    data[5:15, 5:15] = 1.0   # block A — last pixel at (14,14)
    data[15:25, 15:25] = 1.0  # block B — first pixel at (15,15), diagonal to (14,14)
    raster_path = _write_depression_raster(tmp_path / "dep.tif", data, pixel_size_m=10.0)

    _, gdf = gen_wetland_polygons(raster_path, tmp_path / "out")

    assert len(gdf) == 1, "diagonal-touch blocks should be one wetland record"
    assert gdf.geometry.iloc[0].geom_type == "MultiPolygon", (
        "separate polygon pieces of one wetland must be dissolved into a MultiPolygon"
    )


# ---------------------------------------------------------------------------
# Imperial unit columns
# ---------------------------------------------------------------------------

_M2_PER_ACRE = 4046.8564224
_M3_PER_ACFT = 1233.4818375
_FT_PER_M = 3.28084


def test_imperial_columns_present_and_accurate(tmp_path: Path) -> None:
    """The returned GDF (and written shapefile) must include area_ac, vol_acft,
    and med_dep_ft columns derived correctly from the metric values."""
    # 20x20 block at 0.5 m depth with 10 m pixels →
    # area = 400 * 100 = 40 000 m²,  volume = 40 000 * 0.5 = 20 000 m³
    data = np.zeros((30, 30))
    data[5:25, 5:25] = 0.5
    raster_path = _write_depression_raster(tmp_path / "dep.tif", data, pixel_size_m=10.0)

    _, gdf = gen_wetland_polygons(raster_path, tmp_path / "out")
    assert len(gdf) == 1

    row = gdf.iloc[0]

    assert "area_ac" in gdf.columns
    assert "vol_acft" in gdf.columns
    assert "med_dep_ft" in gdf.columns

    assert pytest.approx(row["area_ac"], rel=1e-4) == row["area_m2"] / _M2_PER_ACRE
    assert pytest.approx(row["vol_acft"], rel=1e-4) == row["volume_m3"] / _M3_PER_ACFT
    assert pytest.approx(row["med_dep_ft"], rel=1e-4) == row["med_dep_m"] * _FT_PER_M

def test_catchment_area_in_wetlands_polygon(tmp_path: Path) -> None:
    """Step 11: zonal stats added to gen_wetland_polygons() — max flow accumulation
    value per wetland used to estimate contributing catchment area."""
    # Depression raster: one 20x20 wetland block at 0.5 m depth, 10 m pixels
    # → area = 40,000 m² (passes default 4,000 m² area threshold)
    dep_data = np.zeros((30, 30))
    dep_data[5:25, 5:25] = 0.5
    depressions_path = _write_depression_raster(tmp_path / "dep.tif", dep_data, pixel_size_m=10.0)

    # Accumulation raster on the same grid, with one outlet cell inside the wetland
    # whose value implies a contributing area larger than the wetland's own footprint
    accum_data = np.ones((30, 30))
    accum_data[15, 15] = 500.0  # 500 cells * 100 m²/cell = 50,000 m² > 40,000 m² wetland area
    accum_path = _write_depression_raster(tmp_path / "accum.tif", accum_data, pixel_size_m=10.0)

    _, gdf = gen_wetland_polygons(
        depressions_path,
        tmp_path / "out",
        accum_raster_path=accum_path,
    )

    assert len(gdf) == 1
    assert "wet_cat_m2" in gdf.columns
    assert (gdf["wet_cat_m2"] > gdf["area_m2"]).all()


# ---------------------------------------------------------------------------
# Wetland catchment geometry delineation from user-marked points
# ---------------------------------------------------------------------------


def test_catchment_signature_default_snap_dist_matches_module_constant() -> None:
    """snap_dist and id_field must default to the documented values so the
    runner (and standalone callers) get sensible behaviour when omitted."""
    import inspect

    sig = inspect.signature(gen_wetland_catchment_polygons)
    assert sig.parameters["snap_dist"].default == _SNAP_DIST_M
    assert sig.parameters["id_field"].default == "wetland_id"


def test_gen_wetland_catchment_polygons_assigns_id_from_nearest_wetland(tmp_path: Path) -> None:
    """A delineated catchment's wetland_id is borrowed from the nearest wetland
    polygon — the user-supplied points need no attributes at all. The stub
    watershed raster uses non-sequential value codes (11, 22) to prove the
    polygon-to-wetland mapping doesn't depend on WhiteboxTools' internal
    pour-point numbering."""
    pixel_size = 10.0
    grid_shape = (10, 10)

    d8_path = _write_depression_raster(tmp_path / "d8.tif", np.ones(grid_shape), pixel_size_m=pixel_size)
    accum_path = _write_depression_raster(tmp_path / "accum.tif", np.ones(grid_shape), pixel_size_m=pixel_size)

    # Two spatially distinct wetlands (wetlands never overlap by construction).
    wetlands_path = _write_wetlands_shapefile(
        tmp_path / "wetlands.shp",
        [
            (1, box(0, 70, 30, 100)),   # rows 0-2, cols 0-2
            (2, box(60, 0, 90, 30)),    # rows 7-9, cols 6-8
        ],
    )
    # User-marked points — one inside each wetland, no attributes required.
    points_path = _write_points_shapefile(tmp_path / "points.shp", [(15.0, 85.0), (75.0, 15.0)])

    fake_wbt = _make_fake_wbt()

    def _run_snap_pour_points(**kwargs):
        # Stub: pass points through unchanged. The real snapping math is
        # WhiteboxTools' own domain, exercised for real in the integration test.
        gpd.read_file(kwargs["pour_pts"]).to_file(kwargs["output"])
        return 0

    def _run_watershed(**kwargs):
        data = np.full(grid_shape, -1.0, dtype=np.float32)
        data[0:3, 0:3] = 11.0   # catchment for the wetland-1 point
        data[7:10, 6:9] = 22.0  # catchment for the wetland-2 point
        _write_depression_raster(Path(kwargs["output"]), data, pixel_size_m=pixel_size, nodata=-1.0)
        return 0

    fake_wbt.snap_pour_points.side_effect = _run_snap_pour_points
    fake_wbt.watershed.side_effect = _run_watershed

    with patch("pwa_tools.wetlands.wbt_session") as mock_session:
        mock_session.return_value.__enter__ = lambda s: fake_wbt
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        result_path, gdf = gen_wetland_catchment_polygons(
            points_path, wetlands_path, d8_path, accum_path, tmp_path / "out",
        )

    fake_wbt.snap_pour_points.assert_called_once()
    fake_wbt.watershed.assert_called_once()
    assert result_path.exists()
    assert set(gdf["wetland_id"]) == {1, 2}

    by_id = gdf.set_index("wetland_id")
    assert by_id.loc[1, "geometry"].contains(Point(15.0, 85.0))
    assert by_id.loc[2, "geometry"].contains(Point(75.0, 15.0))
    assert by_id.loc[1, "cat_ar_m2"] == pytest.approx(900.0)
    assert by_id.loc[2, "cat_ar_m2"] == pytest.approx(900.0)
    assert by_id.loc[1, "cat_ar_ac"] == pytest.approx(900.0 / 4046.8564224)


def test_gen_wetland_catchment_polygons_assigns_negative_id_when_no_wetland_within_snap_dist(
    tmp_path: Path, caplog,
) -> None:
    """A point farther than snap_dist from every wetland polygon must not
    silently inherit a distant wetland's id -- it gets a new, unique
    negative id instead (never produced by gen_wetland_polygons, so it's
    unambiguous downstream). A point still within range of a real wetland
    in the same batch must keep its correct, positive id."""
    pixel_size = 10.0
    grid_shape = (10, 10)

    d8_path = _write_depression_raster(tmp_path / "d8.tif", np.ones(grid_shape), pixel_size_m=pixel_size)
    accum_path = _write_depression_raster(tmp_path / "accum.tif", np.ones(grid_shape), pixel_size_m=pixel_size)

    wetlands_path = _write_wetlands_shapefile(
        tmp_path / "wetlands.shp", [(5, box(0, 70, 30, 100))],
    )
    # Point A is inside the wetland; points B and C are both well beyond the
    # default snap_dist (20 m) from any wetland polygon.
    points_path = _write_points_shapefile(
        tmp_path / "points.shp", [(15.0, 85.0), (75.0, 15.0), (45.0, 45.0)],
    )

    fake_wbt = _make_fake_wbt()

    def _run_snap_pour_points(**kwargs):
        gpd.read_file(kwargs["pour_pts"]).to_file(kwargs["output"])
        return 0

    def _run_watershed(**kwargs):
        data = np.full(grid_shape, -1.0, dtype=np.float32)
        data[0:3, 0:3] = 11.0   # point A's catchment
        data[7:10, 6:9] = 22.0  # point B's catchment
        data[4:7, 3:6] = 44.0   # point C's catchment
        _write_depression_raster(Path(kwargs["output"]), data, pixel_size_m=pixel_size, nodata=-1.0)
        return 0

    fake_wbt.snap_pour_points.side_effect = _run_snap_pour_points
    fake_wbt.watershed.side_effect = _run_watershed

    with patch("pwa_tools.wetlands.wbt_session") as mock_session:
        mock_session.return_value.__enter__ = lambda s: fake_wbt
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        with caplog.at_level("WARNING"):
            _, gdf = gen_wetland_catchment_polygons(
                points_path, wetlands_path, d8_path, accum_path, tmp_path / "out",
            )

    assert "no wetland polygon within" in caplog.text

    matched_id = next(
        wid for wid, geom in zip(gdf["wetland_id"], gdf["geometry"])
        if geom.contains(Point(15.0, 85.0))
    )
    assert matched_id == 5

    unmatched_ids = [wid for wid in gdf["wetland_id"] if wid != 5]
    assert len(unmatched_ids) == 2
    assert all(wid < 0 for wid in unmatched_ids)
    assert len(set(unmatched_ids)) == 2  # distinct negative ids, not reused


def test_gen_wetland_catchment_polygons_passes_raster_paths_and_snap_dist(tmp_path: Path) -> None:
    """snap_pour_points and watershed must receive the configured raster paths
    and an overridden snap_dist."""
    pixel_size = 10.0
    grid_shape = (5, 5)
    d8_path = _write_depression_raster(tmp_path / "d8.tif", np.ones(grid_shape), pixel_size_m=pixel_size)
    accum_path = _write_depression_raster(tmp_path / "accum.tif", np.ones(grid_shape), pixel_size_m=pixel_size)
    wetlands_path = _write_wetlands_shapefile(tmp_path / "wetlands.shp", [(1, box(0, 30, 50, 50))])
    points_path = _write_points_shapefile(tmp_path / "points.shp", [(25.0, 40.0)])

    fake_wbt = _make_fake_wbt()

    def _run_snap_pour_points(**kwargs):
        gpd.read_file(kwargs["pour_pts"]).to_file(kwargs["output"])
        return 0

    def _run_watershed(**kwargs):
        data = np.full(grid_shape, -1.0, dtype=np.float32)
        data[0:2, 1:4] = 7.0
        _write_depression_raster(Path(kwargs["output"]), data, pixel_size_m=pixel_size, nodata=-1.0)
        return 0

    fake_wbt.snap_pour_points.side_effect = _run_snap_pour_points
    fake_wbt.watershed.side_effect = _run_watershed

    with patch("pwa_tools.wetlands.wbt_session") as mock_session:
        mock_session.return_value.__enter__ = lambda s: fake_wbt
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        gen_wetland_catchment_polygons(
            points_path, wetlands_path, d8_path, accum_path, tmp_path / "out", snap_dist=42.0,
        )

    snap_kwargs = fake_wbt.snap_pour_points.call_args.kwargs
    assert snap_kwargs["flow_accum"] == str(accum_path)
    assert snap_kwargs["snap_dist"] == 42.0

    watershed_kwargs = fake_wbt.watershed.call_args.kwargs
    assert watershed_kwargs["d8_pntr"] == str(d8_path)


def test_gen_wetland_catchment_polygons_raises_when_wetlands_missing_id_field(tmp_path: Path) -> None:
    """A wetlands_path without the id_field column must fail fast, before any
    WBT call is made."""
    d8_path = _write_depression_raster(tmp_path / "d8.tif", np.ones((5, 5)), pixel_size_m=10.0)
    accum_path = _write_depression_raster(tmp_path / "accum.tif", np.ones((5, 5)), pixel_size_m=10.0)

    wetlands_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 50, 50)], crs="EPSG:32614")
    wetlands_path = tmp_path / "wetlands.shp"
    wetlands_gdf.to_file(wetlands_path)
    points_path = _write_points_shapefile(tmp_path / "points.shp", [(25.0, 25.0)])

    with pytest.raises(ValueError, match="wetland_id"):
        gen_wetland_catchment_polygons(points_path, wetlands_path, d8_path, accum_path, tmp_path / "out")


def test_gen_wetland_catchment_polygons_raises_on_grid_mismatch(tmp_path: Path) -> None:
    """D8 pointer and flow-accumulation rasters on different grids must fail
    fast, before even reading the points/wetlands files."""
    d8_path = _write_depression_raster(tmp_path / "d8.tif", np.ones((10, 10)), pixel_size_m=10.0)
    accum_path = _write_depression_raster(tmp_path / "accum.tif", np.ones((5, 5)), pixel_size_m=10.0)

    with pytest.raises(ValueError, match="grid"):
        gen_wetland_catchment_polygons(
            Path("does_not_exist_points.shp"),
            Path("does_not_exist_wetlands.shp"),
            d8_path, accum_path, tmp_path / "out",
        )


def test_gen_wetland_catchment_polygons_raises_on_empty_points(tmp_path: Path) -> None:
    """An empty points file must raise, not silently produce an empty result."""
    d8_path = _write_depression_raster(tmp_path / "d8.tif", np.ones((5, 5)), pixel_size_m=10.0)
    accum_path = _write_depression_raster(tmp_path / "accum.tif", np.ones((5, 5)), pixel_size_m=10.0)

    empty_points = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32614")
    wetlands_gdf = gpd.GeoDataFrame(
        {"wetland_id": [1]}, geometry=[box(0, 0, 50, 50)], crs="EPSG:32614",
    )

    with patch("pwa_tools.wetlands.read_shapefile", side_effect=[empty_points, wetlands_gdf]):
        with pytest.raises(ValueError, match="no points"):
            gen_wetland_catchment_polygons(
                Path("points.shp"), Path("wetlands.shp"), d8_path, accum_path, tmp_path / "out",
            )


def test_gen_wetland_catchment_polygons_skips_points_with_no_catchment(
    tmp_path: Path, caplog,
) -> None:
    """A point whose snapped location lands on nodata in the watershed raster
    is logged and excluded, without failing the whole batch."""
    pixel_size = 10.0
    grid_shape = (10, 10)
    d8_path = _write_depression_raster(tmp_path / "d8.tif", np.ones(grid_shape), pixel_size_m=pixel_size)
    accum_path = _write_depression_raster(tmp_path / "accum.tif", np.ones(grid_shape), pixel_size_m=pixel_size)

    wetlands_path = _write_wetlands_shapefile(
        tmp_path / "wetlands.shp",
        [
            (1, box(0, 70, 30, 100)),
            (2, box(60, 0, 90, 30)),
        ],
    )
    points_path = _write_points_shapefile(tmp_path / "points.shp", [(15.0, 85.0), (75.0, 15.0)])

    fake_wbt = _make_fake_wbt()

    def _run_snap_pour_points(**kwargs):
        gpd.read_file(kwargs["pour_pts"]).to_file(kwargs["output"])
        return 0

    def _run_watershed(**kwargs):
        # Only the wetland-1 point's location has a valid catchment; the
        # wetland-2 point's snapped location lands on nodata.
        data = np.full(grid_shape, -1.0, dtype=np.float32)
        data[0:3, 0:3] = 11.0
        _write_depression_raster(Path(kwargs["output"]), data, pixel_size_m=pixel_size, nodata=-1.0)
        return 0

    fake_wbt.snap_pour_points.side_effect = _run_snap_pour_points
    fake_wbt.watershed.side_effect = _run_watershed

    with patch("pwa_tools.wetlands.wbt_session") as mock_session:
        mock_session.return_value.__enter__ = lambda s: fake_wbt
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        with caplog.at_level("WARNING"):
            _, gdf = gen_wetland_catchment_polygons(
                points_path, wetlands_path, d8_path, accum_path, tmp_path / "out",
            )

    assert list(gdf["wetland_id"]) == [1]
    assert "nodata" in caplog.text