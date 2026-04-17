"""Unit tests for the WhiteboxTools wrapper (_wbt.py).

These tests verify error-checking behavior and working-directory restoration
without invoking real WBT binaries. WBT initialization is mocked.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pwa_tools._wbt import WBTError, check_wbt, wbt_session


# ============ check_wbt ============


def test_check_wbt_passes_on_zero() -> None:
    check_wbt(0, "clip")  # should not raise


def test_check_wbt_raises_on_nonzero() -> None:
    with pytest.raises(WBTError, match="fill_burn"):
        check_wbt(1, "fill_burn")


def test_check_wbt_raises_on_cancelled() -> None:
    with pytest.raises(WBTError, match="clip"):
        check_wbt(2, "clip")


# ============ WBTError ============


def test_wbt_error_attributes() -> None:
    err = WBTError("zonal_statistics", 1)
    assert err.tool_name == "zonal_statistics"
    assert err.return_code == 1
    assert "zonal_statistics" in str(err)
    assert "1" in str(err)


# ============ wbt_session ============


@patch("pwa_tools._wbt._get_wbt_class")
def test_wbt_session_restores_cwd_on_normal_exit(mock_get_class: MagicMock) -> None:
    """Working directory must be restored after a clean context-manager exit."""
    mock_wbt = MagicMock()
    mock_get_class.return_value = lambda: mock_wbt

    cwd_before = os.getcwd()
    with wbt_session() as wbt:
        # Simulate what WBT.run_tool does internally: chdir to exe_path
        os.chdir("/tmp")
        assert os.getcwd() == "/private/tmp" or os.getcwd() == "/tmp"

    assert os.getcwd() == cwd_before


@patch("pwa_tools._wbt._get_wbt_class")
def test_wbt_session_restores_cwd_on_exception(mock_get_class: MagicMock) -> None:
    """Working directory must be restored even if an exception fires inside."""
    mock_wbt = MagicMock()
    mock_get_class.return_value = lambda: mock_wbt

    cwd_before = os.getcwd()
    with pytest.raises(WBTError):
        with wbt_session() as wbt:
            os.chdir("/tmp")
            raise WBTError("test_tool", 1)

    assert os.getcwd() == cwd_before


@patch("pwa_tools._wbt._get_wbt_class")
def test_wbt_session_yields_wbt_instance(mock_get_class: MagicMock) -> None:
    mock_wbt = MagicMock()
    mock_get_class.return_value = lambda: mock_wbt

    with wbt_session() as wbt:
        assert wbt is mock_wbt


# ============ Integration: check_wbt + wbt_session ============


@patch("pwa_tools._wbt._get_wbt_class")
def test_typical_usage_pattern(mock_get_class: MagicMock) -> None:
    """Simulate the intended caller pattern: session + checked calls."""
    mock_wbt = MagicMock()
    mock_wbt.clip.return_value = 0
    mock_wbt.fill_burn.return_value = 1  # simulated failure
    mock_get_class.return_value = lambda: mock_wbt

    cwd_before = os.getcwd()
    with pytest.raises(WBTError, match="fill_burn"):
        with wbt_session() as wbt:
            check_wbt(wbt.clip(i="a.shp", clip="b.shp", output="c.shp"), "clip")
            check_wbt(wbt.fill_burn(dem="d.tif", streams="e.shp", output="f.tif"), "fill_burn")

    # cwd restored despite exception
    assert os.getcwd() == cwd_before
    # clip was called, fill_burn was called
    mock_wbt.clip.assert_called_once()
    mock_wbt.fill_burn.assert_called_once()
