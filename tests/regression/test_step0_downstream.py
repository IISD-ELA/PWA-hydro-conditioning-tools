"""Step 0 downstream-half regression test — placeholder until the refactor lands.

Why this exists as a skip:
  The current pwa_tools/__init__.py is the 1,139-line god file with global
  state, 8 input() prompts, and unchecked subprocess calls. Driving it from
  pytest end-to-end requires either (a) heroic monkeypatching of input(), or
  (b) waiting for the Phase 1/2 refactor that replaces state with PwaConfig
  and removes the input() calls. (b) is much cheaper.

Why "downstream half":
  Thomas's dataset is missing the raw LiDAR .tif inputs to Step 0's
  merge_rasters function. We cannot regenerate the merge from scratch.
  However, the merged DEM (Interim/merged_average_dem_filled.tif) IS present,
  so we can re-run the rest of Step 0 starting from that artifact:

      [merged DEM]  ←── starting point (have)
            │
            ▼
      resample → clip → fill_burn → calc_depression_depths
            │
            ▼
      [CLRH_basins_depression_depths.shp]  ←── compare to baseline

What this test will do once enabled:
  1. Build a PwaConfig pointing at data/grassmere/ (or wherever we relocate)
  2. Run the refactored Step 0 downstream functions in sequence
  3. Hash the output shapefile + raster
  4. Compare against tests/regression/grassmere-baseline.json
  5. If hashes drift, fall back to numeric comparison of Deps_Depth column
     against baseline samples (within tolerance — GDAL version differences
     can produce identical data with slightly different file headers)

Until enabled, the baseline integrity test in test_baseline_integrity.py
provides the safety net by ensuring Thomas's reference data hasn't been
modified locally.
"""

from __future__ import annotations

import pytest


@pytest.mark.regression
@pytest.mark.skip(
    reason=(
        "Blocked on Phase 1/2 refactor: the current god-file pwa_tools requires "
        "interactive input() prompts and global state mutation, neither of which "
        "is reasonably driveable from pytest. Will be enabled once Step 0 "
        "functions accept a PwaConfig and the runner is decoupled from input(). "
        "See conftest.py and bug-tracker.md for the prerequisites."
    )
)
def test_step0_downstream_regenerates_depression_depths() -> None:
    """Re-run Step 0 from the merged DEM and confirm outputs match baseline."""
    raise NotImplementedError(
        "See module docstring for the implementation plan. "
        "Wire up after Phase 2 extracts the depression module."
    )
