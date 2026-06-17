"""Tests for the pwa_tools.run_step0 CLI module.

Covers argument parsing, missing-config handling, and that the parsed
config + flags are plumbed through to pwa_tools.runner.run_step0.
"""

from __future__ import annotations

from pathlib import Path

from pwa_tools.run_step0 import _build_arg_parser, main
from pwa_tools.runner import Step0Result


def _stub_run_step0(monkeypatch, result: Step0Result | None = None) -> dict:
    """Replace pwa_tools.run_step0.run_step0 with a recorder. Returns the
    recorder dict so tests can assert on what was passed in."""
    if result is None:
        result = Step0Result(
            depression_depths=Path("/tmp/depths.shp"),
            depression_raster=Path("/tmp/depths.tif"),
        )
    captured: dict = {}

    def _fake(config, generate_wetlands: bool = False):
        captured["config"] = config
        captured["generate_wetlands"] = generate_wetlands
        return result

    monkeypatch.setattr("pwa_tools.run_step0.run_step0", _fake)
    return captured


def _write_minimal_config(yaml_path: Path, base_data_dir: Path) -> None:
    yaml_path.write_text(
        f"""
watershed_name: cypress_river
base_data_dir: {base_data_dir}
inputs:
  clrh_filename: finalcat_info_v1-0
  lidar_filenames:
    - Pembina_LiDAR_DEM
  nhn_filename: NHN_05MH000_3_0_HD_SLWATER_1
  culvert_filename: null
  crs_string: EPSG:3158
"""
    )


# ============ Parser ============


def test_parser_defaults_config_to_pwa_config_yml() -> None:
    args = _build_arg_parser().parse_args([])
    assert args.config == Path("pwa_config.yml")
    assert args.wetlands is False
    assert args.log_level == "INFO"


def test_parser_accepts_custom_config_and_wetlands_flag() -> None:
    args = _build_arg_parser().parse_args(["--config", "x.yml", "--wetlands"])
    assert args.config == Path("x.yml")
    assert args.wetlands is True


# ============ main() ============


def test_main_returns_1_when_config_missing(tmp_path: Path, capsys) -> None:
    rc = main(["--config", str(tmp_path / "absent.yml")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Config file not found" in err
    assert "init_config" in err  # actionable hint


def test_main_passes_config_and_wetlands_through_to_runner(
    monkeypatch, tmp_path: Path
) -> None:
    yaml_path = tmp_path / "pwa_config.yml"
    _write_minimal_config(yaml_path, tmp_path)
    captured = _stub_run_step0(monkeypatch)

    rc = main(["--config", str(yaml_path), "--wetlands"])
    assert rc == 0
    assert captured["generate_wetlands"] is True
    assert captured["config"].watershed_name == "cypress_river"


def test_main_omits_wetlands_path_when_not_generated(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    yaml_path = tmp_path / "pwa_config.yml"
    _write_minimal_config(yaml_path, tmp_path)
    _stub_run_step0(monkeypatch)

    rc = main(["--config", str(yaml_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Depression depths" in out
    assert "Wetland polygons" not in out


def test_main_prints_wetlands_path_when_generated(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    yaml_path = tmp_path / "pwa_config.yml"
    _write_minimal_config(yaml_path, tmp_path)
    _stub_run_step0(
        monkeypatch,
        Step0Result(
            depression_depths=Path("/tmp/d.shp"),
            depression_raster=Path("/tmp/d.tif"),
            wetlands=Path("/tmp/wetlands.shp"),
        ),
    )

    rc = main(["--config", str(yaml_path), "--wetlands"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Wetland polygons" in out
    assert "wetlands.shp" in out
