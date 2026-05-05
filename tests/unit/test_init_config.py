"""Tests for the pwa_tools.init_config CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pwa_tools.config import PwaConfig
from pwa_tools.init_config import _build_arg_parser, main


def test_arg_parser_requires_output_path():
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_minimal(tmp_path: Path):
    parser = _build_arg_parser()
    args = parser.parse_args([str(tmp_path / "out.yml")])
    assert args.output_path == tmp_path / "out.yml"
    assert args.base_data_dir is None
    assert args.force is False


def _stub_inputs(monkeypatch, answers: list[str]) -> None:
    """Replace input() with a queue of canned answers."""
    iterator = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: next(iterator))


def test_main_writes_valid_yaml(monkeypatch, tmp_path: Path, capsys):
    """End-to-end: feed stdin, get a YAML file that round-trips through PwaConfig."""
    out = tmp_path / "config.yml"
    # Match prompt_for_config's call sequence: watershed, clrh, lidar,
    # nhn, culvert, crs.
    _stub_inputs(monkeypatch, [
        "grassmere",                  # watershed
        "finalcat_info_v1-0",         # clrh
        "Pembina_LiDAR_DEM",          # lidar
        "NHN_05MH000_3_0_HD_SLWATER_1",  # nhn
        "",                           # culvert (empty)
        "EPSG:3158",                  # crs
    ])

    rc = main([
        str(out),
        "--base-data-dir", str(tmp_path / "Data"),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert f"Wrote {out}" in captured.out

    # Round-trips through PwaConfig
    config = PwaConfig.from_yaml(out)
    assert config.watershed_name == "grassmere"
    assert config.inputs.clrh_filename == "finalcat_info_v1-0"
    assert config.inputs.lidar_filenames == ["Pembina_LiDAR_DEM"]
    assert config.inputs.nhn_filename == "NHN_05MH000_3_0_HD_SLWATER_1"
    assert config.inputs.culvert_filename is None
    assert config.inputs.crs_string == "EPSG:3158"

    # And the YAML on disk is human-readable
    text = out.read_text()
    assert "watershed_name: grassmere" in text
    assert "!!python" not in text


def test_main_refuses_overwrite_without_force(monkeypatch, tmp_path: Path, capsys):
    out = tmp_path / "existing.yml"
    out.write_text("watershed_name: keep_me\n")

    # input() should never be called — we bail before prompting.
    monkeypatch.setattr(
        "builtins.input",
        lambda *_a, **_kw: pytest.fail("input() called despite refuse-overwrite"),
    )

    rc = main([str(out)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Refusing to overwrite" in err
    # File untouched
    assert out.read_text() == "watershed_name: keep_me\n"


def test_main_overwrites_with_force(monkeypatch, tmp_path: Path):
    out = tmp_path / "existing.yml"
    out.write_text("watershed_name: stale\n")
    _stub_inputs(monkeypatch, [
        "fresh",
        "finalcat_info_v1-0",
        "Pembina_LiDAR_DEM",
        "NHN_05MH000_3_0_HD_SLWATER_1",
        "",
        "EPSG:3158",
    ])
    rc = main([str(out), "--force", "--base-data-dir", str(tmp_path / "Data")])
    assert rc == 0
    assert "watershed_name: fresh" in out.read_text()


def test_main_creates_output_parent(monkeypatch, tmp_path: Path):
    """to_yaml's parent-dir creation flows through."""
    out = tmp_path / "deeply" / "nested" / "config.yml"
    _stub_inputs(monkeypatch, [
        "deepwater",
        "finalcat_info_v1-0",
        "Pembina_LiDAR_DEM",
        "NHN_05MH000_3_0_HD_SLWATER_1",
        "",
        "EPSG:3158",
    ])
    rc = main([str(out), "--base-data-dir", str(tmp_path / "Data")])
    assert rc == 0
    assert out.is_file()
