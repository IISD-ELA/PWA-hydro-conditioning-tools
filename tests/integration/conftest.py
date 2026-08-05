"""Fixtures for pwa_tools integration tests — real binaries / real data.

Local-only: GitHub CI runs ``pytest tests/unit`` and never enters this
directory. The deeper Step 0 science (WhiteboxTools + gdalwarp end-to-end on
real grassmere data) already lives in ``tests/regression/``; the integration
test here covers the one path regression doesn't — driving Step 0 through its
**module CLI** (``python -m pwa_tools.run_step0``) on a real config.

Inputs are discovered from the environment so no machine-specific path is baked
into the test:

* ``PWA_STEP0_CONFIG`` — path to a filled-in ``pwa_config.yml`` (generate one
  with ``pwa-init-hydrocondition`` / ``python -m pwa_tools.init_config``).

Skips cleanly when unset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def step0_config_path() -> Path:
    """Path to a real ``pwa_config.yml`` from ``$PWA_STEP0_CONFIG``, or skip."""
    config_path = os.environ.get("PWA_STEP0_CONFIG")
    if not config_path:
        pytest.skip(
            "Set PWA_STEP0_CONFIG to a filled-in pwa_config.yml "
            "(see `python -m pwa_tools.init_config`) to run the Step 0 CLI e2e."
        )
    path = Path(config_path)
    if not path.is_file():
        pytest.fail(f"PWA_STEP0_CONFIG={config_path!r} is not a file")
    return path
