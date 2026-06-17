"""Tests for pwa_tools.ui — interactive prompt helpers.

Tests snake_case (pure) and prompt_for_config (with monkeypatched input).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pwa_tools.ui import prompt_for_config, snake_case


# ============ snake_case ============


def test_snake_case_spaces() -> None:
    assert snake_case("Cypress River") == "cypress_river"


def test_snake_case_hyphens() -> None:
    assert snake_case("manning-canal") == "manning_canal"


def test_snake_case_mixed() -> None:
    assert snake_case("Red River Valley") == "red_river_valley"


def test_snake_case_already_snake() -> None:
    assert snake_case("already_snake") == "already_snake"


# ============ prompt_for_config ============


def test_prompt_for_config_with_all_defaults(tmp_path: Path) -> None:
    """Pressing Enter at every prompt should produce a valid config from defaults."""
    # Empty string at every input() call → all defaults applied
    with patch("builtins.input", return_value=""):
        config = prompt_for_config(
            base_data_dir=tmp_path,
            defaults={
                "watershed": "grassmere",
                "clrh": "finalcat_info_v1-0",
                "lidar": "Pembina_LiDAR_DEM",
                "nhn": "NHN_05MH000_3_0_HD_SLWATER_1",
                "culvert": "",
                "crs": "EPSG:3158",
            },
        )

    assert config.watershed_name == "grassmere"
    assert config.inputs.clrh_filename == "finalcat_info_v1-0"
    assert config.inputs.lidar_filenames == ["Pembina_LiDAR_DEM"]
    assert config.inputs.nhn_filename == "NHN_05MH000_3_0_HD_SLWATER_1"
    assert config.inputs.culvert_filename is None
    assert config.inputs.crs_string == "EPSG:3158"
    assert config.paths.base_data == tmp_path


def test_prompt_for_config_with_user_values(tmp_path: Path) -> None:
    """User provides custom values at each prompt."""
    responses = iter([
        "Manning Canal",                    # watershed name
        "my_watershed",                     # CLRH filename
        "my_lidar",                         # LiDAR filename
        "my_nhn",                           # NHN filename
        "",                                 # culvert (empty → None)
        "EPSG:32614",                       # CRS
    ])

    with patch("builtins.input", side_effect=responses):
        config = prompt_for_config(
            base_data_dir=tmp_path,
            defaults={
                "watershed": "default_ws",
                "clrh": "default_clrh",
                "lidar": "default_lidar",
                "nhn": "default_nhn",
                "culvert": "",
                "crs": "EPSG:3158",
            },
        )

    assert config.watershed_name == "manning_canal"  # snake_case applied
    assert config.inputs.clrh_filename == "my_watershed"
    assert config.inputs.crs_string == "EPSG:32614"


def test_prompt_for_config_multi_lidar(tmp_path: Path) -> None:
    """Comma-separated LiDAR filenames should produce a list."""
    responses = iter([
        "",                                 # watershed → default
        "",                                 # CLRH → default
        "raster_a, raster_b, raster_c",    # LiDAR → multi
        "",                                 # NHN → default
        "",                                 # culvert → default (empty)
        "",                                 # CRS → default
    ])

    with patch("builtins.input", side_effect=responses):
        config = prompt_for_config(
            base_data_dir=tmp_path,
            defaults={
                "watershed": "test_ws",
                "clrh": "test_clrh",
                "lidar": "test_lidar",
                "nhn": "test_nhn",
                "culvert": "",
                "crs": "EPSG:3158",
            },
        )

    assert config.inputs.lidar_filenames == ["raster_a", "raster_b", "raster_c"]
    assert config.inputs.multiple_lidar_rasters is True


# ============ prompt_for_config staging warning ============


def test_prompt_for_config_warns_when_inputs_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the named inputs aren't staged on disk, prompt_for_config should
    print a non-blocking warning so users catch typos and case-mismatches
    without re-running the full pipeline."""
    with patch("builtins.input", return_value=""):
        config = prompt_for_config(
            base_data_dir=tmp_path,
            defaults={
                "watershed": "grassmere",
                "clrh": "finalcat_info_v1-0",
                "lidar": "Pembina_LiDAR_DEM",
                "nhn": "NHN_05MH000_3_0_HD_SLWATER_1",
                "culvert": "",
                "crs": "EPSG:3158",
            },
        )

    captured = capsys.readouterr()
    # Warning goes to stderr, lists at least one of the missing input filenames
    assert "finalcat_info_v1-0.shp" in captured.err
    # Should not raise — this is a warning, not an error
    assert config is not None


def test_prompt_for_config_no_warning_when_inputs_present(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If every named input file already exists on disk, no warning."""
    # Pre-stage the expected directory layout & files
    raw = tmp_path / "grassmere" / "HydroConditioning" / "Raw"
    raw.mkdir(parents=True)
    (raw / "finalcat_info_v1-0.shp").touch()
    (raw / "Pembina_LiDAR_DEM.tif").touch()
    (raw / "NHN_05MH000_3_0_HD_SLWATER_1.shp").touch()

    with patch("builtins.input", return_value=""):
        prompt_for_config(
            base_data_dir=tmp_path,
            defaults={
                "watershed": "grassmere",
                "clrh": "finalcat_info_v1-0",
                "lidar": "Pembina_LiDAR_DEM",
                "nhn": "NHN_05MH000_3_0_HD_SLWATER_1",
                "culvert": "",
                "crs": "EPSG:3158",
            },
        )

    captured = capsys.readouterr()
    # No warning text in stderr — the file paths from expected_input_files
    # should not appear at all
    assert "finalcat_info_v1-0.shp" not in captured.err


# ============ runner import ============


def test_runner_imports_cleanly() -> None:
    """Verify the runner module can be imported without triggering side effects."""
    from pwa_tools.runner import Step0Result, run_step0

    assert callable(run_step0)
    assert Step0Result is not None
