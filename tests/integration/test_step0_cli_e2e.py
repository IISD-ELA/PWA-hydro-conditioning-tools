"""End-to-end: drive Step 0 through its module CLI on a real config.

Why this exists
---------------
``tests/regression/`` already runs the Step 0 science (WhiteboxTools + gdalwarp)
end-to-end by calling the functions directly. What it does **not** exercise is
the user-facing entry point: argument parsing, config loading via
``PwaConfig.from_yaml``, the missing-config exit code, and the
``run_step0``-from-``__main__`` wiring — all on real data. A broken CLI wrapper
(or a regression in entry-point plumbing) would pass every regression test yet
break every user.

Run with::

    PWA_STEP0_CONFIG=/path/to/pwa_config.yml pytest tests/integration

Marked ``integration`` + ``slow`` (a real Step 0 run is minutes, not seconds)
and ``regression`` (needs real watershed inputs). Skips without the config.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.regression
@pytest.mark.slow
def test_run_step0_module_cli_succeeds(step0_config_path):
    """``python -m pwa_tools.run_step0 --config <real>`` exits 0 on real data."""
    completed = subprocess.run(
        [sys.executable, "-m", "pwa_tools.run_step0", "--config", str(step0_config_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"Step 0 CLI failed (exit {completed.returncode}).\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    # The wrapper prints the two headline outputs on success.
    assert "Depression depths" in completed.stdout


def test_run_step0_module_cli_reports_missing_config():
    """A no-binary, no-data smoke check of the CLI's error path: a missing
    config yields a non-zero exit and a helpful message. Runs everywhere, so
    it's the one integration-dir test that doesn't need real inputs."""
    completed = subprocess.run(
        [sys.executable, "-m", "pwa_tools.run_step0", "--config", "definitely_missing.yml"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "Config file not found" in completed.stderr
