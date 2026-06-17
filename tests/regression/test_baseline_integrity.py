"""Verify that Thomas's reference dataset on disk matches the committed baseline.

This test does NOT exercise our code — it confirms that the source-of-truth
dataset hasn't been accidentally modified since we baselined it. If a hash
mismatch fires, either:

  1. Someone edited a file in ``data/`` (recover it from S3 / Thomas)
  2. Thomas sent us an updated dataset (regenerate baseline + commit)

Pure stdlib — no geopandas/rasterio needed. Marked ``regression`` for
filterability; runs by default if the dataset is on disk and skips
automatically (via the ``grassmere_data_dir`` fixture) if not. Devs without
the ~6 GB dataset see ``SKIPPED``; devs with it see ``PASSED``.

To deliberately skip regression on a fast dev loop::

    pytest -m "not regression"

To run regression-only::

    pytest -m regression
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


CHUNK_SIZE = 1024 * 1024  # 1 MB — large enough to be efficient, small enough to bound memory


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(block)
    return h.hexdigest()


@pytest.mark.regression
def test_step0_processed_files_match_baseline(
    grassmere_data_dir: Path, baseline_manifest: dict
) -> None:
    """Step 0 outputs in data/HydroConditioning/Processed/ match committed hashes."""
    failures = []
    for relpath, expected in baseline_manifest["files"]["step0_processed"].items():
        full = grassmere_data_dir / relpath
        if not full.exists():
            failures.append(f"  MISSING: {relpath}")
            continue
        actual_size = full.stat().st_size
        if actual_size != expected["size_bytes"]:
            failures.append(
                f"  SIZE MISMATCH: {relpath} "
                f"(expected {expected['size_bytes']:,}, got {actual_size:,})"
            )
            continue
        actual_hash = _sha256_of(full)
        if actual_hash != expected["sha256"]:
            failures.append(
                f"  HASH MISMATCH: {relpath}\n"
                f"    expected: {expected['sha256']}\n"
                f"    got:      {actual_hash}"
            )

    if failures:
        pytest.fail(
            "Step 0 reference data drift detected:\n"
            + "\n".join(failures)
            + "\n\nIf this is intentional (Thomas sent updated data), regenerate "
            "the baseline:\n"
            "  python tests/regression/generate_baseline.py"
        )


@pytest.mark.regression
def test_step12_processed_files_match_baseline(
    grassmere_data_dir: Path, baseline_manifest: dict
) -> None:
    """Step 1–2 outputs in data/Raven/Processed/ match committed hashes."""
    failures = []
    for relpath, expected in baseline_manifest["files"]["step12_processed"].items():
        full = grassmere_data_dir / relpath
        if not full.exists():
            failures.append(f"  MISSING: {relpath}")
            continue
        actual_size = full.stat().st_size
        if actual_size != expected["size_bytes"]:
            failures.append(
                f"  SIZE MISMATCH: {relpath} "
                f"(expected {expected['size_bytes']:,}, got {actual_size:,})"
            )
            continue
        actual_hash = _sha256_of(full)
        if actual_hash != expected["sha256"]:
            failures.append(
                f"  HASH MISMATCH: {relpath}\n"
                f"    expected: {expected['sha256']}\n"
                f"    got:      {actual_hash}"
            )

    if failures:
        pytest.fail(
            "Step 1–2 reference data drift detected:\n"
            + "\n".join(failures)
            + "\n\nIf this is intentional (Thomas sent updated data), regenerate "
            "the baseline:\n"
            "  python tests/regression/generate_baseline.py"
        )
