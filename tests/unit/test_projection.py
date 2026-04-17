"""Tests for pwa_tools.projection — CRS reprojection helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from pwa_tools.projection import (
    _sanitize_crs_for_filename,
    project_subbasins_to_nhn,
)


def test_sanitize_crs_short() -> None:
    assert _sanitize_crs_for_filename("EPSG:4326") == "EPSG4326"


def test_sanitize_crs_truncates_long() -> None:
    result = _sanitize_crs_for_filename("NAD83(CSRS) / UTM zone 14N")
    assert len(result) <= 10
    assert result.isalnum()


def test_project_subbasins_to_nhn_writes_shapefile(tmp_path: Path) -> None:
    # Use coordinates within Manitoba (valid for EPSG:3158 / UTM 14N)
    subbasins = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(-97, 50)], crs="EPSG:4326",
    )
    nhn = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(600000, 5500000)], crs="EPSG:3158",
    )

    projected, out_path = project_subbasins_to_nhn(
        nhn, subbasins, "test_basins", tmp_path,
    )

    assert projected.crs.to_epsg() == 3158
    assert out_path.exists()
    assert "projected" in out_path.name
