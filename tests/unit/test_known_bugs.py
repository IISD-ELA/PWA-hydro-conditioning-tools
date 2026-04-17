"""Regression tests for known bugs documented in project-review/bug-tracker.md.

Each test asserts the *correct* (post-fix) behavior and is marked `xfail` until
the corresponding fix lands. When a fix arrives, pytest reports `XPASS` for the
matching test and the `@pytest.mark.xfail` marker should be removed in the same
PR — that is the signal the bug is closed.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

import pwa_tools


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG-001: read_shapefile() raises UnboundLocalError when the input "
        "shapefile's CRS already matches state.crs_string and new_crs is the "
        "default empty string. Neither if/elif branch fires, leaving shape_out "
        "unassigned. See bug-tracker.md and pwa_tools/__init__.py:247-262. "
        "Fix scheduled in Phase 1 of the cleanup project."
    ),
)
def test_bug_001_read_shapefile_with_matching_crs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a shapefile is already in the project CRS, read_shapefile should
    return it unchanged. Currently raises UnboundLocalError.
    """
    # Set the project CRS that read_shapefile will compare against
    monkeypatch.setattr(pwa_tools.state, "crs_string", "EPSG:4326")

    # Write a tiny single-point shapefile in the same CRS as the project
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    gdf.to_file(tmp_path / "matching_crs.shp")

    # Call read_shapefile with the default new_crs=''.
    # Note the trailing '/' on the directory: the legacy code does
    # `directory + filename + ".shp"`, so the caller is required to terminate
    # the path. This is one of the things pathlib will fix in Phase 1.
    result = pwa_tools.read_shapefile(
        filename="matching_crs",
        directory=str(tmp_path) + "/",
    )

    # Post-fix expectation: shapefile returned with its original CRS preserved.
    assert result.crs.to_epsg() == 4326
    assert len(result) == 1
