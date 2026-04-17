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


# ============ runner import ============


def test_runner_imports_cleanly() -> None:
    """Verify the runner module can be imported without triggering side effects."""
    from pwa_tools.runner import Step0Result, run_step0

    assert callable(run_step0)
    assert Step0Result is not None
