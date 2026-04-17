"""Tests for pwa_tools.streams — stream network processing.

Tests the pure `extend_line` function and `append_culvert_lines` with
in-memory GeoDataFrames. `clip_nhn_to_watershed` requires WBT and is
deferred to integration tests.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point

from pwa_tools.streams import append_culvert_lines, extend_line


# ============ extend_line ============


def test_extend_line_extends_both_ends() -> None:
    line = LineString([(0, 0), (10, 0)])
    extended = extend_line(line, 5.0)

    coords = list(extended.coords)
    assert len(coords) == 4  # new_start, original 2, new_end
    assert coords[0][0] == pytest.approx(-5.0)
    assert coords[-1][0] == pytest.approx(15.0)


def test_extend_line_preserves_non_linestring() -> None:
    point = Point(0, 0)
    assert extend_line(point, 5.0) is point


def test_extend_line_diagonal() -> None:
    """45-degree line extended by sqrt(2) should move 1 unit on each axis."""
    line = LineString([(0, 0), (1, 1)])
    extended = extend_line(line, np.sqrt(2))

    coords = list(extended.coords)
    assert coords[0][0] == pytest.approx(-1.0, abs=1e-10)
    assert coords[0][1] == pytest.approx(-1.0, abs=1e-10)
    assert coords[-1][0] == pytest.approx(2.0, abs=1e-10)
    assert coords[-1][1] == pytest.approx(2.0, abs=1e-10)


# ============ append_culvert_lines ============


def test_append_culvert_lines_combines_sources() -> None:
    channels = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(2, 0), (3, 0)])],
        crs="EPSG:32614",
    )
    culverts = gpd.GeoDataFrame(
        {"id": [10]},
        geometry=[LineString([(1, 0), (2, 0)])],
        crs="EPSG:32614",
    )

    result = append_culvert_lines(channels, culverts)
    assert len(result) == 3
    assert list(result["source"]) == ["NHN", "NHN", "Culvert"]


def test_append_culvert_lines_reprojects_if_needed() -> None:
    channels = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:32614",
    )
    culverts = gpd.GeoDataFrame(
        {"id": [10]},
        geometry=[LineString([(-96, 50), (-96.01, 50)])],
        crs="EPSG:4326",  # different CRS
    )

    result = append_culvert_lines(channels, culverts)
    assert len(result) == 2


def test_append_culvert_lines_writes_shapefile(tmp_path: Path) -> None:
    channels = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:32614",
    )
    culverts = gpd.GeoDataFrame(
        {"id": [10]},
        geometry=[LineString([(1, 0), (2, 0)])],
        crs="EPSG:32614",
    )

    out_path = tmp_path / "burn_lines.shp"
    result = append_culvert_lines(channels, culverts, output_path=out_path)
    assert out_path.exists()
    assert len(result) == 2
