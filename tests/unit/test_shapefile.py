"""Unit tests for pwa_tools.io.shapefile.read_shapefile.

Creates tiny single-point shapefiles in tmp_path to exercise all code paths.
No external data needed.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from pwa_tools.io.shapefile import read_shapefile


# ============ Helpers ============


def _write_point_shp(tmp_path: Path, name: str, crs: str) -> Path:
    """Write a 1-point shapefile with the given CRS. Returns path to .shp."""
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs=crs)
    path = tmp_path / f"{name}.shp"
    gdf.to_file(path)
    return path


# ============ Happy paths ============


def test_read_shapefile_no_reprojection(tmp_path: Path) -> None:
    """No target_crs → return with original CRS."""
    shp = _write_point_shp(tmp_path, "test", "EPSG:4326")
    gdf = read_shapefile(shp)
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == 1


def test_read_shapefile_reproject_to_different_crs(tmp_path: Path) -> None:
    """target_crs differs from file CRS → reproject."""
    shp = _write_point_shp(tmp_path, "test", "EPSG:4326")
    gdf = read_shapefile(shp, target_crs="EPSG:3158")
    assert gdf.crs.to_epsg() == 3158
    assert len(gdf) == 1


def test_read_shapefile_matching_crs_returns_as_is(tmp_path: Path) -> None:
    """target_crs matches file CRS → return unchanged.

    Pins the no-reprojection path. The legacy implementation in
    __init__.py raised UnboundLocalError on this branch because neither
    of its if/elif arms set the return variable.
    """
    shp = _write_point_shp(tmp_path, "test", "EPSG:4326")
    gdf = read_shapefile(shp, target_crs="EPSG:4326")
    assert gdf.crs.to_epsg() == 4326
    assert len(gdf) == 1


def test_read_shapefile_accepts_str_path(tmp_path: Path) -> None:
    """Str paths are coerced to Path internally."""
    shp = _write_point_shp(tmp_path, "test", "EPSG:4326")
    gdf = read_shapefile(str(shp))
    assert gdf.crs.to_epsg() == 4326


# ============ Error paths ============


def test_read_shapefile_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        read_shapefile(tmp_path / "nonexistent.shp")


def test_read_shapefile_raises_on_no_crs_when_target_set(tmp_path: Path) -> None:
    """If the shapefile has no CRS and target_crs is requested, raise early."""
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
    assert gdf.crs is None
    shp = tmp_path / "no_crs.shp"
    gdf.to_file(shp)

    with pytest.raises(ValueError, match="no CRS defined"):
        read_shapefile(shp, target_crs="EPSG:4326")
