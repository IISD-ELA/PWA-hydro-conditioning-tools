"""Regression checkpoints for behaviours the legacy ``pwa_tools.__init__``
code still gets wrong. The corresponding ``pwa_tools.io.*`` modules have
already been fixed (and are covered positively in ``test_shapefile.py``,
``test_raster.py``, etc.); the xfail markers below are reminders that the
legacy god-file consumer still needs migration before those bugs disappear
for real users.

When the legacy ``__init__.py`` function is deleted, drop both the marker
and the test in the same change.
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
        "Legacy pwa_tools.read_shapefile raises UnboundLocalError when "
        "the input shapefile's CRS already matches state.crs_string and "
        "new_crs is the default empty string — neither if/elif branch "
        "fires, leaving shape_out unassigned. The new pwa_tools.io.shapefile "
        "module fixes this; the xfail will go away once hydro_condition.py "
        "stops importing the legacy function."
    ),
)
def test_legacy_read_shapefile_with_matching_crs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a shapefile is already in the project CRS, read_shapefile should
    return it unchanged. The legacy implementation in __init__.py raises
    UnboundLocalError instead.
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
