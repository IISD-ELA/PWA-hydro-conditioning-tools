"""Unit tests for PwaConfig and its component dataclasses.

These tests do not touch the filesystem except where pytest's tmp_path
fixture is used (PwaPaths.make_dirs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pwa_tools.config import PwaConfig, PwaInputs, PwaPaths


# ============ PwaPaths ============


def test_paths_from_watershed_derives_full_layout(tmp_path: Path) -> None:
    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    assert paths.base_data == tmp_path
    assert paths.watershed == tmp_path / "cypress_river"
    assert paths.hydrocon == tmp_path / "cypress_river" / "HydroConditioning"
    assert paths.hydrocon_raw == paths.hydrocon / "Raw"
    assert paths.hydrocon_interim == paths.hydrocon / "Interim"
    assert paths.hydrocon_processed == paths.hydrocon / "Processed"


def test_paths_from_watershed_does_not_touch_filesystem(tmp_path: Path) -> None:
    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    assert not paths.watershed.exists()


def test_paths_make_dirs_creates_full_structure(tmp_path: Path) -> None:
    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    paths.make_dirs()
    assert paths.hydrocon_raw.is_dir()
    assert paths.hydrocon_interim.is_dir()
    assert paths.hydrocon_processed.is_dir()


def test_paths_make_dirs_is_idempotent(tmp_path: Path) -> None:
    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    paths.make_dirs()
    paths.make_dirs()  # second call must not raise


def test_paths_are_frozen(tmp_path: Path) -> None:
    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    with pytest.raises(Exception):  # FrozenInstanceError
        paths.base_data = Path("/elsewhere")  # type: ignore[misc]


def test_paths_coerces_str_base_data_to_path(tmp_path: Path) -> None:
    paths = PwaPaths.from_watershed(str(tmp_path), "cypress_river")
    assert isinstance(paths.base_data, Path)


# ============ PwaInputs ============


def _valid_inputs(**overrides) -> PwaInputs:
    """Helper: build a PwaInputs with sensible defaults, overriding selected fields."""
    defaults = dict(
        clrh_filename="finalcat_info_v1-0",
        lidar_filenames=["Pembina_LiDAR_DEM"],
        nhn_filename="NHN_05MH000_3_0_HD_SLWATER_1",
        crs_string="EPSG:3158",
        culvert_filename=None,
    )
    defaults.update(overrides)
    return PwaInputs(**defaults)


def test_inputs_multiple_lidar_rasters_false_for_single() -> None:
    assert _valid_inputs(lidar_filenames=["only_one"]).multiple_lidar_rasters is False


def test_inputs_multiple_lidar_rasters_true_for_many() -> None:
    assert _valid_inputs(lidar_filenames=["a", "b"]).multiple_lidar_rasters is True


def test_inputs_rejects_empty_clrh_filename() -> None:
    with pytest.raises(ValueError, match="clrh_filename"):
        _valid_inputs(clrh_filename="")


def test_inputs_rejects_empty_nhn_filename() -> None:
    with pytest.raises(ValueError, match="nhn_filename"):
        _valid_inputs(nhn_filename="")


def test_inputs_rejects_empty_lidar_filenames() -> None:
    with pytest.raises(ValueError, match="lidar_filenames"):
        _valid_inputs(lidar_filenames=[])


def test_inputs_rejects_invalid_crs_string() -> None:
    with pytest.raises(ValueError, match="EPSG"):
        _valid_inputs(crs_string="3158")  # missing prefix


def test_inputs_accepts_lowercase_epsg() -> None:
    """Pyproj is case-insensitive for the EPSG prefix; mirror that tolerance."""
    inputs = _valid_inputs(crs_string="epsg:3158")
    assert inputs.crs_string == "epsg:3158"


def test_inputs_culvert_filename_defaults_to_none() -> None:
    inputs = PwaInputs(
        clrh_filename="x", lidar_filenames=["y"], nhn_filename="z", crs_string="EPSG:1"
    )
    assert inputs.culvert_filename is None


# ============ PwaConfig ============


def _config_dict(tmp_path: Path, **overrides) -> dict:
    base = {
        "watershed_name": "cypress_river",
        "base_data_dir": str(tmp_path),
        "inputs": {
            "clrh_filename": "finalcat_info_v1-0",
            "lidar_filenames": ["Pembina_LiDAR_DEM"],
            "nhn_filename": "NHN_05MH000_3_0_HD_SLWATER_1",
            "crs_string": "EPSG:3158",
        },
    }
    base.update(overrides)
    return base


def test_config_from_dict_minimal(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    assert config.watershed_name == "cypress_river"
    assert config.paths.watershed == tmp_path / "cypress_river"
    assert config.inputs.clrh_filename == "finalcat_info_v1-0"
    assert config.inputs.multiple_lidar_rasters is False


def test_config_from_dict_coerces_str_lidar_to_list(tmp_path: Path) -> None:
    """YAML may have lidar_filenames as a single string for the single-file case."""
    data = _config_dict(tmp_path)
    data["inputs"]["lidar_filenames"] = "single_file"  # str, not list
    config = PwaConfig.from_dict(data)
    assert config.inputs.lidar_filenames == ["single_file"]
    assert config.inputs.multiple_lidar_rasters is False


def test_config_from_dict_treats_empty_culvert_as_none(tmp_path: Path) -> None:
    """Legacy state code used '' as the absence sentinel for culvert; normalize to None."""
    data = _config_dict(tmp_path)
    data["inputs"]["culvert_filename"] = ""
    config = PwaConfig.from_dict(data)
    assert config.inputs.culvert_filename is None


def test_config_rejects_empty_watershed_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="watershed_name"):
        PwaConfig.from_dict(_config_dict(tmp_path, watershed_name=""))


def test_config_from_yaml_roundtrip(tmp_path: Path) -> None:
    config_yaml = tmp_path / "pwa_config.yml"
    config_yaml.write_text(
        f"""
watershed_name: cypress_river
base_data_dir: {tmp_path}
inputs:
  clrh_filename: finalcat_info_v1-0
  lidar_filenames:
    - Pembina_LiDAR_DEM
  nhn_filename: NHN_05MH000_3_0_HD_SLWATER_1
  culvert_filename: null
  crs_string: EPSG:3158
"""
    )
    config = PwaConfig.from_yaml(config_yaml)
    assert config.watershed_name == "cypress_river"
    assert config.inputs.multiple_lidar_rasters is False
    assert config.inputs.culvert_filename is None


def test_config_is_frozen(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    with pytest.raises(Exception):  # FrozenInstanceError
        config.watershed_name = "renamed"  # type: ignore[misc]
