"""Unit tests for pwa_tools.runner — conditional wetland-catchment wiring.

The rest of run_step0's pipeline (LiDAR resampling, stream clipping,
depression generation, ...) is already exercised end-to-end with real
WhiteboxTools/gdal in tests/regression/. This file isolates the one
behaviour this change introduces: gen_wetland_catchment_polygons is called
-- using the Raw/ points file and the D8/accumulation rasters already
produced for wetland_catchments -- only when
config.inputs.wetland_points_filename is set, and its result lands in
Step0Result.wetland_catchment_polygons.

Every externally-imported pipeline function is stubbed out: this file is
about wiring, not pipeline science.
"""

from __future__ import annotations

from pathlib import Path

from pwa_tools.config import PwaConfig, PwaPaths
from pwa_tools.runner import run_step0


def _patch_pipeline(monkeypatch, tmp_path: Path) -> dict:
    """Stub every pipeline function pwa_tools.runner calls so run_step0
    completes without touching the filesystem or any external binary.

    Returns a dict recording calls relevant to the new wiring: whether
    gen_wetland_polygons ran, and what gen_wetland_catchment_polygons was
    called with (empty if it wasn't called at all).
    """
    calls: dict = {}

    monkeypatch.setattr(PwaConfig, "stage_inputs", lambda self: None)
    monkeypatch.setattr(PwaPaths, "make_dirs", lambda self: None)
    monkeypatch.setattr(PwaPaths, "clean_interim", lambda self: None)

    monkeypatch.setattr("pwa_tools.runner.read_shapefile", lambda *a, **k: [])
    monkeypatch.setattr("pwa_tools.runner.crs_uses_metres", lambda *_: True)
    monkeypatch.setattr(
        "pwa_tools.runner.project_subbasins_to_lidar",
        lambda *a, **k: ([], "EPSG:32614", "32614", tmp_path / "clrh_proj.shp"),
    )
    monkeypatch.setattr(
        "pwa_tools.runner.project_subbasins_to_nhn",
        lambda *a, **k: ([], tmp_path / "nhn_proj.shp"),
    )
    monkeypatch.setattr(
        "pwa_tools.runner.clip_lidar_to_shapefile",
        lambda *a, **k: tmp_path / "lidar_clipped.tif",
    )
    monkeypatch.setattr(
        "pwa_tools.runner.resample_lidar_raster",
        lambda path, resolution_m=None: Path(str(path)),
    )
    monkeypatch.setattr(
        "pwa_tools.runner.clip_nhn_to_watershed",
        lambda *a, **k: tmp_path / "nhn_clipped.shp",
    )
    monkeypatch.setattr(
        "pwa_tools.runner.gen_depressions_raster",
        lambda *a, **k: (tmp_path / "deps.tif", tmp_path / "filled.tif"),
    )
    monkeypatch.setattr(
        "pwa_tools.runner.gen_direction_raster", lambda *a, **k: tmp_path / "d8.tif"
    )
    monkeypatch.setattr(
        "pwa_tools.runner.gen_accum_raster", lambda *a, **k: tmp_path / "accum.tif"
    )

    def _fake_gen_wetland_polygons(*a, **k):
        calls["gen_wetland_polygons_called"] = True
        return tmp_path / "wetlands.shp", []

    monkeypatch.setattr(
        "pwa_tools.runner.gen_wetland_polygons", _fake_gen_wetland_polygons
    )

    def _fake_catchments(points_path, wetlands_path, direction_raster_path, accum_raster_path, output_dir, **kwargs):
        calls["catchments_call"] = {
            "points_path": points_path,
            "wetlands_path": wetlands_path,
            "direction_raster_path": direction_raster_path,
            "accum_raster_path": accum_raster_path,
            "output_dir": output_dir,
        }
        return tmp_path / "catchments.shp", []

    monkeypatch.setattr(
        "pwa_tools.runner.gen_wetland_catchment_polygons", _fake_catchments
    )

    return calls


def _build_config(tmp_path: Path, *, wetland_points_filename: str | None) -> PwaConfig:
    data = {
        "watershed_name": "test_ws",
        "base_data_dir": str(tmp_path),
        "inputs": {
            "clrh_filename": "clrh",
            "lidar_filenames": ["lidar"],
            "nhn_filename": "nhn",
            "crs_string": "EPSG:3158",
            "wetland_points_filename": wetland_points_filename,
        },
        "summarize_by_subbasin": False,
        "wetland_catchments": wetland_points_filename is not None,
    }
    return PwaConfig.from_dict(data)


def test_run_step0_calls_catchment_delineation_when_points_configured(
    monkeypatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path, wetland_points_filename="wetland_points")
    calls = _patch_pipeline(monkeypatch, tmp_path)

    result = run_step0(config)

    catchments_call = calls["catchments_call"]
    assert catchments_call["points_path"] == config.paths.hydrocon_raw / "wetland_points.shp"
    assert catchments_call["wetlands_path"] == tmp_path / "wetlands.shp"
    assert catchments_call["direction_raster_path"] == tmp_path / "d8.tif"
    assert catchments_call["accum_raster_path"] == tmp_path / "accum.tif"
    assert result.wetland_catchment_polygons == tmp_path / "catchments.shp"


def test_run_step0_skips_catchment_delineation_when_points_not_configured(
    monkeypatch, tmp_path: Path
) -> None:
    config = _build_config(tmp_path, wetland_points_filename=None)
    calls = _patch_pipeline(monkeypatch, tmp_path)

    result = run_step0(config)

    assert "catchments_call" not in calls
    assert result.wetland_catchment_polygons is None
    # generate_wetlands defaults to False and no points file is configured,
    # so gen_wetland_polygons must not run either.
    assert "gen_wetland_polygons_called" not in calls


def test_run_step0_forces_wetland_polygons_when_points_configured_but_flag_omitted(
    monkeypatch, tmp_path: Path
) -> None:
    """generate_wetlands=False (the CLI default) must not suppress wetland
    polygon generation when wetland_points_filename needs it for ID lookup."""
    config = _build_config(tmp_path, wetland_points_filename="wetland_points")
    calls = _patch_pipeline(monkeypatch, tmp_path)

    result = run_step0(config, generate_wetlands=False)

    assert calls.get("gen_wetland_polygons_called") is True
    assert result.wetlands == tmp_path / "wetlands.shp"
    assert "catchments_call" in calls
