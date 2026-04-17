"""Shared fixtures for regression tests.

Regression tests need access to Thomas's reference dataset, which lives outside
any of the three repos at ``<project_root>/data/``. We discover it via:

  1. ``PWA_TEST_DATA_DIR`` environment variable (overrides everything)
  2. Walking up from this file until we find a ``data/`` dir containing
     ``HydroConditioning/``

If neither works, regression tests are skipped — devs without the ~6 GB dataset
can still run the rest of the suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def grassmere_data_dir() -> Path:
    """Absolute path to the grassmere reference data directory."""
    env_path = os.environ.get("PWA_TEST_DATA_DIR")
    if env_path:
        path = Path(env_path)
        if not path.is_dir():
            pytest.fail(f"PWA_TEST_DATA_DIR={env_path!r} does not exist")
        return path

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if candidate.is_dir() and (candidate / "HydroConditioning").is_dir():
            return candidate

    pytest.skip(
        "Reference dataset not found. Place Thomas's grassmere data at "
        "<project_root>/data/ or set PWA_TEST_DATA_DIR to its location."
    )


@pytest.fixture(scope="session")
def baseline_manifest() -> dict:
    """Load the committed baseline manifest as a dict."""
    manifest_path = Path(__file__).parent / "grassmere-baseline.json"
    if not manifest_path.exists():
        pytest.skip(
            f"Baseline manifest not found at {manifest_path}. "
            "Run `python tests/regression/generate_baseline.py` to create it."
        )
    return json.loads(manifest_path.read_text())
