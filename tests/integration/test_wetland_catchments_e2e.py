"""End-to-end: delineate wetland catchment geometry with real WhiteboxTools.

Why this exists
---------------
``tests/unit/test_wetlands.py`` exercises ``gen_wetland_catchment_polygons``
against mocked WhiteboxTools calls — fast, but it can't catch a mismatch
between our assumptions and the real ``snap_pour_points``/``watershed``
behaviour. This test drives the real binary (bundled on Windows, or via the
``whitebox`` pip package elsewhere) on a small synthetic DEM, so it needs no
real watershed data and stays fast — unlike ``tests/regression/``, which
needs the grassmere reference dataset.

Synthetic scenario: a DEM that slopes uniformly from north to south with no
east-west gradient. D8 flow is unambiguously due south for every interior
cell (the diagonal neighbours have the same elevation drop but a longer
run, so they're never steeper), so flow accumulation at a cell depends only
on its row — a deterministic, hand-checkable catchment shape without needing
a real watershed.

Skips cleanly when no WhiteboxTools binary is available.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point, box

from pwa_tools.wetlands import (
    gen_accum_raster,
    gen_direction_raster,
    gen_wetland_catchment_polygons,
)


def _wbt_available() -> bool:
    try:
        from pwa_tools._wbt import _get_wbt_class

        _get_wbt_class()
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _wbt_available(), reason="No WhiteboxTools binary available"),
]

_PIXEL_SIZE_M = 10.0
_GRID_SHAPE = (20, 20)  # (rows, cols)


def _write_dem(path: Path) -> Path:
    """A DEM sloping uniformly north-to-south: elevation depends only on row."""
    height, width = _GRID_SHAPE
    rows = np.arange(height, dtype=np.float32).reshape(-1, 1)
    data = 100.0 - rows  # strictly decreasing southward, constant per column
    data = np.repeat(data, width, axis=1)

    transform = from_bounds(
        0, 0, width * _PIXEL_SIZE_M, height * _PIXEL_SIZE_M, width, height,
    )
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype=np.float32, crs="EPSG:32614", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


def test_gen_wetland_catchment_polygons_real_wbt(tmp_path: Path) -> None:
    dem_path = _write_dem(tmp_path / "dem.tif")

    d8_path = gen_direction_raster(dem_path, tmp_path)
    accum_path = gen_accum_raster(d8_path, tmp_path)

    # A single wetland near the top (upstream end) of the slope.
    wetland_row, wetland_col = 9, 10
    x = (wetland_col + 0.5) * _PIXEL_SIZE_M
    y = _GRID_SHAPE[0] * _PIXEL_SIZE_M - (wetland_row + 0.5) * _PIXEL_SIZE_M

    wetlands_gdf = gpd.GeoDataFrame(
        {"wetland_id": [1]},
        geometry=[box(x - 5, y - 5, x + 5, y + 5)],
        crs="EPSG:32614",
    )
    wetlands_path = tmp_path / "wetlands.shp"
    wetlands_gdf.to_file(wetlands_path)

    points_gdf = gpd.GeoDataFrame(geometry=[Point(x, y)], crs="EPSG:32614")
    points_path = tmp_path / "points.shp"
    points_gdf.to_file(points_path)

    # Small snap_dist: this DEM has no east-west gradient, so every cell in a
    # row shares the same flow-accumulation value. A large snap_dist could
    # legitimately pull the pour point several rows downstream to the single
    # highest-accumulation cell in the search window; keeping it tiny (well
    # under one pixel) keeps the test's expected catchment shape predictable.
    result_path, gdf = gen_wetland_catchment_polygons(
        points_path, wetlands_path, d8_path, accum_path, tmp_path / "out",
        snap_dist=1.0,
    )

    assert result_path.exists()
    assert len(gdf) == 1
    assert gdf.iloc[0]["wetland_id"] == 1

    pixel_area = _PIXEL_SIZE_M * _PIXEL_SIZE_M
    grid_area = _GRID_SHAPE[0] * _GRID_SHAPE[1] * pixel_area
    catchment_m2 = gdf.iloc[0]["cat_ar_m2"]
    # More than a single pixel (real upstream delineation happened) but well
    # short of the whole grid (flow never converges east-west on this DEM).
    assert pixel_area < catchment_m2 < grid_area
