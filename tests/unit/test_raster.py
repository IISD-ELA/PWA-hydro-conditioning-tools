"""Unit tests for pwa_tools.io.raster.

Tests resample (via mock subprocess), clip (with synthetic raster), and
fill_nodata_gaps + get_raster_resolution (with synthetic rasters).
merge_rasters is too complex for a pure unit test — integration tests
with multi-raster fixtures will be added in Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from pwa_tools.io.raster import (
    clip_lidar_to_shapefile,
    fill_nodata_gaps,
    get_raster_resolution,
    resample_lidar_raster,
)


# ============ Helpers ============


def _write_synthetic_raster(
    path: Path,
    data: np.ndarray | None = None,
    width: int = 10,
    height: int = 10,
    crs: str = "EPSG:4326",
    nodata: float = -9999.0,
) -> Path:
    """Write a tiny single-band GeoTIFF. Returns the path."""
    if data is None:
        data = np.ones((1, height, width), dtype=np.float32) * 100.0

    transform = from_bounds(0, 0, width, height, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)

    return path


# ============ resample_lidar_raster ============


def test_resample_calls_gdalwarp_with_check(tmp_path: Path) -> None:
    """Verify the subprocess command and that check=True is passed."""
    raster = _write_synthetic_raster(tmp_path / "dem.tif")

    with patch("pwa_tools.io.raster.subprocess.run") as mock_run:
        resample_lidar_raster(raster, resolution_m=5)

        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]  # first positional arg is the command list
        assert cmd[0] == "gdalwarp"
        assert "-tr" in cmd
        assert "5" in cmd
        assert str(raster) in cmd
        assert args[1]["check"] is True  # the fix for the missing check


def test_resample_passes_overwrite_flag(tmp_path: Path) -> None:
    """gdalwarp refuses to overwrite by default; the -overwrite flag is what
    lets re-runs of a partially-completed pipeline succeed instead of
    crashing on stale Interim/ outputs."""
    raster = _write_synthetic_raster(tmp_path / "dem.tif")

    with patch("pwa_tools.io.raster.subprocess.run") as mock_run:
        resample_lidar_raster(raster, resolution_m=5)
        cmd = mock_run.call_args[0][0]
        assert "-overwrite" in cmd


def test_resample_output_path_includes_resolution(tmp_path: Path) -> None:
    raster = _write_synthetic_raster(tmp_path / "dem.tif")

    with patch("pwa_tools.io.raster.subprocess.run"):
        result = resample_lidar_raster(raster, resolution_m=5)

    assert result.name == "dem_resample_5m.tif"
    assert result.parent == tmp_path


def test_resample_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resample_lidar_raster(tmp_path / "nonexistent.tif", resolution_m=5)


# ============ clip_lidar_to_shapefile ============


def test_clip_produces_output_file(tmp_path: Path) -> None:
    """Clip a 10x10 raster to a polygon covering the left half → output exists."""
    raster = _write_synthetic_raster(tmp_path / "dem.tif", width=10, height=10)

    # Polygon covering left half of the raster (x: 0-5, y: 0-10)
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(0, 0, 5, 10)],
        crs="EPSG:4326",
    )

    result = clip_lidar_to_shapefile(gdf, raster, tmp_path)
    assert result.exists()
    assert result.name == "dem_clip.tif"

    with rasterio.open(result) as src:
        assert src.width <= 10  # clipped should be narrower or same


# ============ get_raster_resolution ============


def test_get_raster_resolution_returns_pixel_size(tmp_path: Path) -> None:
    raster = _write_synthetic_raster(tmp_path / "dem.tif", width=10, height=10)
    res = get_raster_resolution(raster)
    assert res == pytest.approx(1.0)


# ============ fill_nodata_gaps ============


def test_fill_nodata_replaces_gaps(tmp_path: Path) -> None:
    """A raster with nodata pixels in the center should have them filled."""
    data = np.ones((1, 20, 20), dtype=np.float32) * 100.0
    nodata = -9999.0
    # Punch a 4x4 hole in the center
    data[0, 8:12, 8:12] = nodata

    input_path = _write_synthetic_raster(tmp_path / "with_gaps.tif", data=data, width=20, height=20, nodata=nodata)
    output_path = tmp_path / "filled.tif"

    fill_nodata_gaps(input_path, output_path, buffer_px=5)

    with rasterio.open(output_path) as src:
        filled = src.read(1)

    # The center should no longer be nodata
    center = filled[8:12, 8:12]
    assert not np.any(center == nodata), "Nodata gaps were not filled"
    # Filled values should be close to the surrounding 100.0
    assert np.all(center == pytest.approx(100.0)), "Filled values should match neighbors"


def test_fill_nodata_edge_block_trims_buffer_to_window_size(tmp_path: Path) -> None:
    """A nodata gap touching a *block edge* must be filled and the
    written window must match the original block's geometry (not the
    buffered read window).

    The implementation reads a *buffered* window — the original block
    plus ``buffer_px`` pixels on each side — so the
    distance-transform can see neighbours across block boundaries
    and reach into a gap that sits exactly at an edge. Before
    writing, the buffer must be trimmed back to the original window.

    A previous version of this function trimmed by computing
    ``data[buffer_px : buffer_px + height, ...]`` — which is wrong
    at the *top* and *left* edges of the raster, where the buffer
    gets clipped to 0 and the offset is no longer ``buffer_px``.
    The current implementation tracks the true offset via
    ``row_off - buffered_window.row_off`` and trims correctly. This
    test exercises both edges (top-left corner block touches
    raster origin) and the no-gap fast path's matching trim logic.

    Without this test, an edge-trim regression would manifest as
    shifted or duplicated rows in the output and silently corrupt
    every downstream depression-detection step — but only on the
    boundary rows/cols of a tiled raster, which fixture data with
    a single window never exercises."""
    # 32x32 raster, blocks of 16 → 4 blocks total, two of which touch
    # the top edge (row_off == 0). Profile must opt in to tiled
    # blocks so block_windows() returns more than one window.
    nodata = -9999.0
    data = np.ones((1, 32, 32), dtype=np.float32) * 100.0
    # Gap at the very top-left corner: forces the no-gap fast path
    # for some blocks and the slow path for the corner block.
    data[0, 0:3, 0:3] = nodata
    # And a strip of unique values along the right edge so a wrong
    # trim offset would write the wrong columns and we'd notice.
    data[0, :, 31] = 42.0

    input_path = tmp_path / "edge_gap.tif"
    transform = from_bounds(0, 0, 32, 32, 32, 32)
    with rasterio.open(
        input_path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(data)

    output_path = tmp_path / "edge_filled.tif"
    fill_nodata_gaps(input_path, output_path, buffer_px=4)

    with rasterio.open(output_path) as src:
        filled = src.read(1)

    # Corner gap is gone
    assert not np.any(filled[0:3, 0:3] == nodata), (
        "Top-left corner gap not filled"
    )
    # Right-edge strip preserved at the right column index — wrong
    # trim offset would shift it left and this would fail
    assert np.all(filled[:, 31] == pytest.approx(42.0)), (
        "Right edge column shifted — edge-block trim is mis-offset"
    )
    # And the body unchanged
    assert np.all(filled[5:30, 5:30] == pytest.approx(100.0))


def test_fill_nodata_passthrough_when_no_gaps(tmp_path: Path) -> None:
    """A raster with no nodata should pass through unchanged."""
    data = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
    input_path = _write_synthetic_raster(tmp_path / "clean.tif", data=data, width=10, height=10)
    output_path = tmp_path / "clean_filled.tif"

    fill_nodata_gaps(input_path, output_path, buffer_px=3)

    with rasterio.open(output_path) as src:
        result = src.read(1)

    np.testing.assert_array_equal(result, data[0])
