"""Unit tests for pwa_tools.depression.calc_depression_depths resolution parameter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pwa_tools.depression import calc_depression_depths


def _make_fake_wbt():
    wbt = MagicMock()
    wbt.vector_polygons_to_raster.return_value = 0
    wbt.zonal_statistics.return_value = 0
    return wbt


@pytest.fixture()
def stub_inputs(tmp_path: Path):
    """Minimal filesystem stubs required for calc_depression_depths to run."""
    clrh_shp = tmp_path / "clrh_proj.shp"
    clrh_shp.touch()
    deps_tif = tmp_path / "deps.tif"

    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(0, 0, 100, 100, 10, 10)
    with rasterio.open(
        deps_tif, "w", driver="GTiff", height=10, width=10,
        count=1, dtype=np.float32, crs="EPSG:32614", transform=transform,
    ) as dst:
        dst.write(np.zeros((1, 10, 10), dtype=np.float32))

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"FID": [1], "BasArea": [1000.0], "geometry": [box(0, 0, 100, 100)]},
        crs="EPSG:32614",
    )
    return {"clrh_shp": clrh_shp, "deps_tif": deps_tif, "gdf": gdf}


def _run_calc(stub_inputs, tmp_path, resolution_m=5.0, **kwargs):
    """Run calc_depression_depths with WBT and subprocess fully mocked."""
    fake_wbt = _make_fake_wbt()
    fake_html = '<table><tr><th>Mean</th></tr><tr><td>0.1</td></tr></table>'

    with (
        patch("pwa_tools.depression.wbt_session") as mock_session,
        patch("pwa_tools.depression.subprocess.run") as mock_run,
        patch("pwa_tools.depression.pd.read_html", return_value=[
            __import__("pandas").DataFrame({"Mean": [0.1]})
        ]),
    ):
        mock_session.return_value.__enter__ = lambda s: fake_wbt
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        calc_depression_depths(
            clrh_proj_lidar_path=stub_inputs["clrh_shp"],
            watershed_name="test_ws",
            depressions_raster_path=stub_inputs["deps_tif"],
            clrh_gdf=stub_inputs["gdf"],
            processed_dir=tmp_path,
            resolution_m=resolution_m,
            **kwargs,
        )

        return fake_wbt, mock_run


def test_default_resolution_is_5(stub_inputs, tmp_path):
    fake_wbt, mock_run = _run_calc(stub_inputs, tmp_path)
    _, kwargs = fake_wbt.vector_polygons_to_raster.call_args
    assert kwargs["cell_size"] == 5.0
    gdalwarp_args = mock_run.call_args[0][0]
    tr_idx = gdalwarp_args.index("-tr")
    assert gdalwarp_args[tr_idx + 1] == "5.0"
    assert gdalwarp_args[tr_idx + 2] == "5.0"


def test_custom_resolution_forwarded_to_wbt_and_gdalwarp(stub_inputs, tmp_path):
    fake_wbt, mock_run = _run_calc(stub_inputs, tmp_path, resolution_m=10.0)
    _, kwargs = fake_wbt.vector_polygons_to_raster.call_args
    assert kwargs["cell_size"] == 10.0
    gdalwarp_args = mock_run.call_args[0][0]
    tr_idx = gdalwarp_args.index("-tr")
    assert gdalwarp_args[tr_idx + 1] == "10.0"
    assert gdalwarp_args[tr_idx + 2] == "10.0"
