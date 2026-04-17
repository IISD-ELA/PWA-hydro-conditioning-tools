"""Shared pytest fixtures for pwa-tools tests.

Real fixtures (synthetic DEMs, shapefiles, NetCDF) will be added as we extract
modules during Phase 2. This file establishes the convention up front.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to tests/fixtures/ for tests that need committed sample data."""
    return Path(__file__).parent / "fixtures"
