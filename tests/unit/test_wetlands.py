"""Unit tests for pwa_tools.wetlands — threshold parameter behaviour.

These tests exercise the optional depth_llim / area_llim / volume_llim
arguments added to gen_wetland_polygons without touching the filesystem
beyond what pytest's tmp_path fixture provides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from pwa_tools import wetlands
from pwa_tools.wetlands import (
    _AREA_THRESHOLD_M2,
    _DEPTH_THRESHOLD_M,
    _VOLUME_THRESHOLD_M3,
    gen_wetland_polygons,
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
    assert pytest.approx(row["med_dep_ft"], rel=1e-4) == row["median_depth_m"] * _FT_PER_M

