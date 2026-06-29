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


def test_clean_interim_removes_stale_files_and_subdirs(tmp_path: Path) -> None:
    """Every Step 0 run starts with an empty Interim/ — leftover files
    and subdirectories from a prior run get wiped, not silently shadowed
    by the new run's outputs."""
    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    paths.make_dirs()
    (paths.hydrocon_interim / "stale.tif").write_text("old raster")
    sub = paths.hydrocon_interim / "scratch"
    sub.mkdir()
    (sub / "nested.tif").write_text("nested junk")

    paths.clean_interim()

    assert paths.hydrocon_interim.is_dir()
    assert list(paths.hydrocon_interim.iterdir()) == []


def test_clean_interim_is_noop_when_dir_missing(tmp_path: Path) -> None:
    """First-ever run has no Interim/ yet — clean_interim must not raise.
    Callers run make_dirs() separately to materialize the layout."""
    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    # No make_dirs() — Interim/ does not exist yet.
    paths.clean_interim()  # must not raise
    assert not paths.hydrocon_interim.exists()


def test_clean_interim_refuses_path_not_named_interim(tmp_path: Path) -> None:
    """Safety net: if a caller hand-builds a PwaPaths with the wrong
    hydrocon_interim, clean_interim must refuse rather than rm -rf the
    contents of an unrelated directory."""
    from dataclasses import replace

    paths = PwaPaths.from_watershed(tmp_path, "cypress_river")
    bad_target = tmp_path / "NotInterim"
    bad_target.mkdir()
    (bad_target / "precious.txt").write_text("do not delete me")
    bad_paths = replace(paths, hydrocon_interim=bad_target)

    with pytest.raises(ValueError, match="not 'Interim'"):
        bad_paths.clean_interim()
    # Contents untouched.
    assert (bad_target / "precious.txt").read_text() == "do not delete me"


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


def test_config_output_res_m_defaults_to_5(tmp_path: Path) -> None:
    """output_res_m should default to 5.0 when omitted from the config dict."""
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    assert config.output_res_m == 5.0


def test_config_output_res_m_loaded_from_dict(tmp_path: Path) -> None:
    """A custom output_res_m value supplied in the config dict should be stored as-is."""
    config = PwaConfig.from_dict(_config_dict(tmp_path, output_res_m=3.0))
    assert config.output_res_m == 3.0


def test_config_output_res_m_roundtrips_via_to_dict(tmp_path: Path) -> None:
    """output_res_m must survive a to_dict() → from_dict() round-trip."""
    config = PwaConfig.from_dict(_config_dict(tmp_path, output_res_m=10.0))
    assert PwaConfig.from_dict(config.to_dict()).output_res_m == 10.0


# ============ Input-file validation ============


def _stage_inputs(config: PwaConfig, *, omit: tuple[str, ...] = ()) -> None:
    """Create empty placeholder files for every expected input.

    Pass *omit* names (basenames without extension) to skip particular files
    so we can assert validate_inputs_exist raises for missing ones.
    """
    config.paths.make_dirs()
    for path in config.expected_input_files():
        if path.stem in omit:
            continue
        path.touch()


def test_expected_input_files_lists_all(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    files = config.expected_input_files()
    raw = config.paths.hydrocon_raw
    assert raw / "finalcat_info_v1-0.shp" in files
    assert raw / "Pembina_LiDAR_DEM.tif" in files
    assert raw / "NHN_05MH000_3_0_HD_SLWATER_1.shp" in files
    # No culvert filename → no culvert path
    assert all("culvert" not in str(p).lower() for p in files)


def test_expected_input_files_includes_culvert_when_set(tmp_path: Path) -> None:
    data = _config_dict(tmp_path)
    data["inputs"]["culvert_filename"] = "my_culverts"
    config = PwaConfig.from_dict(data)
    files = config.expected_input_files()
    assert config.paths.hydrocon_raw / "my_culverts.shp" in files


def test_expected_input_files_lists_every_lidar_in_multi_raster_mode(tmp_path: Path) -> None:
    data = _config_dict(tmp_path)
    data["inputs"]["lidar_filenames"] = ["tile_a", "tile_b", "tile_c"]
    config = PwaConfig.from_dict(data)
    files = config.expected_input_files()
    assert config.paths.hydrocon_raw / "tile_a.tif" in files
    assert config.paths.hydrocon_raw / "tile_b.tif" in files
    assert config.paths.hydrocon_raw / "tile_c.tif" in files


def test_validate_inputs_exist_passes_when_all_present(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    _stage_inputs(config)
    config.validate_inputs_exist()  # should not raise


def test_validate_inputs_exist_raises_listing_all_missing(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    _stage_inputs(config, omit=("finalcat_info_v1-0", "Pembina_LiDAR_DEM"))
    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_inputs_exist()
    msg = str(exc_info.value)
    assert "finalcat_info_v1-0.shp" in msg
    assert "Pembina_LiDAR_DEM.tif" in msg
    # NHN was staged → not in the missing list
    assert "NHN_05MH000_3_0_HD_SLWATER_1.shp" not in msg


def test_validate_inputs_exist_raises_when_directory_absent(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    # Don't make_dirs — the entire raw directory is absent.
    with pytest.raises(FileNotFoundError, match="missing"):
        config.validate_inputs_exist()


def test_validate_inputs_exist_lists_sibling_watersheds_when_dir_missing(
    tmp_path: Path,
) -> None:
    """When the watershed dir is absent, surface the siblings present in
    base_data_dir so case/spelling mismatches are obvious to the user."""
    # Stage a couple of unrelated sibling directories
    (tmp_path / "Grassmere-test-run").mkdir()  # the user's actual on-disk name
    (tmp_path / "manning_canal").mkdir()
    (tmp_path / "some_file.txt").touch()  # not a directory — should be filtered out

    config = PwaConfig.from_dict(_config_dict(tmp_path))  # watershed_name=cypress_river
    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_inputs_exist()

    msg = str(exc_info.value)
    assert "Grassmere-test-run" in msg
    assert "manning_canal" in msg
    assert "some_file.txt" not in msg  # files filtered out, only dirs listed
    # The original missing-path is still surfaced
    assert "cypress_river" in msg


def test_validate_inputs_exist_handles_missing_base_data_dir(tmp_path: Path) -> None:
    """If base_data_dir itself doesn't exist, fail without crashing on the
    sibling-listing code."""
    nonexistent = tmp_path / "does_not_exist"
    config = PwaConfig.from_dict(
        _config_dict(tmp_path, base_data_dir=str(nonexistent))
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_inputs_exist()
    # Should mention the base_data_dir issue, not crash with a different error
    msg = str(exc_info.value)
    assert "missing" in msg or "does not exist" in msg.lower()


def test_validate_inputs_exist_per_file_message_unchanged_when_dir_present(
    tmp_path: Path,
) -> None:
    """When the Raw/ dir does exist, keep the original per-file error format —
    don't switch to the sibling-listing branch."""
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    config.paths.make_dirs()  # Raw/ exists, but no input files inside
    with pytest.raises(FileNotFoundError) as exc_info:
        config.validate_inputs_exist()
    msg = str(exc_info.value)
    # Per-file format mentions specific filenames
    assert "finalcat_info_v1-0.shp" in msg
    # And does NOT include the sibling-listing header
    assert "Available directories" not in msg


# ============ to_dict / to_yaml round-trip ============


def test_to_dict_round_trip(tmp_path: Path) -> None:
    original = PwaConfig.from_dict(_config_dict(tmp_path))
    rebuilt = PwaConfig.from_dict(original.to_dict())
    assert rebuilt == original


def test_to_dict_with_culvert_round_trip(tmp_path: Path) -> None:
    data = _config_dict(tmp_path)
    data["inputs"]["culvert_filename"] = "my_culverts"
    original = PwaConfig.from_dict(data)
    rebuilt = PwaConfig.from_dict(original.to_dict())
    assert rebuilt == original
    assert rebuilt.inputs.culvert_filename == "my_culverts"


def test_to_dict_with_multiple_lidar_round_trip(tmp_path: Path) -> None:
    data = _config_dict(tmp_path)
    data["inputs"]["lidar_filenames"] = ["tile_a", "tile_b"]
    original = PwaConfig.from_dict(data)
    rebuilt = PwaConfig.from_dict(original.to_dict())
    assert rebuilt == original
    assert rebuilt.inputs.multiple_lidar_rasters is True


def test_to_yaml_writes_human_readable_file(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    out = tmp_path / "out" / "pwa_config.yml"
    written = config.to_yaml(out)
    assert written == out
    assert out.is_file()
    text = out.read_text()
    # Block style, key order preserved (watershed_name first).
    assert text.startswith("watershed_name:")
    assert "lidar_filenames:\n- " in text or "lidar_filenames:\n  - " in text
    # No python tags.
    assert "!!python" not in text


def test_to_yaml_round_trip(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    out = tmp_path / "pwa_config.yml"
    config.to_yaml(out)
    rebuilt = PwaConfig.from_yaml(out)
    assert rebuilt == config


def test_to_yaml_creates_parent_dir(tmp_path: Path) -> None:
    config = PwaConfig.from_dict(_config_dict(tmp_path))
    out = tmp_path / "deeply" / "nested" / "pwa_config.yml"
    config.to_yaml(out)
    assert out.is_file()
