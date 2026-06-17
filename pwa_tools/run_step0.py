"""CLI: ``python -m pwa_tools.run_step0 --config pwa_config.yml``.

Two equivalent ways to drive the Step 0 hydro-conditioning pipeline:

* **From the shell** (one-shot, after ``init_config``)::

    python -m pwa_tools.run_step0 --config pwa_config.yml
    python -m pwa_tools.run_step0 --config pwa_config.yml --wetlands
    python -m pwa_tools.run_step0 --log-level DEBUG

* **From Python (notebook or script)**::

    from pwa_tools.config import PwaConfig
    from pwa_tools.runner import run_step0
    config = PwaConfig.from_yaml("pwa_config.yml")
    result = run_step0(config)
    print(result.depression_depths, result.depression_raster)

The CLI is a thin wrapper around :func:`pwa_tools.runner.run_step0`. It
mirrors the entry-point pattern used by ``pwa_raven.run_nc_processing``
and ``pwa_raven.run_raven_inputs`` so users have one mental model across
all pipeline steps. The ``hydro_condition_v2.py`` script in the
``PWA-hydro-conditioning-main`` repo is a backwards-compatibility shim
that calls into this module.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from pwa_tools.config import PwaConfig
from pwa_tools.runner import Step0Result, run_step0

__all__ = ["main", "run_step0", "Step0Result", "PwaConfig"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pwa-hydrocondition",
        description="Run the PWA Step 0 hydro-conditioning pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pwa_config.yml"),
        help="Path to the pwa_config.yml file (default: ./pwa_config.yml).",
    )
    parser.add_argument(
        "--wetlands",
        action="store_true",
        help="Also generate the wetland polygons shapefile (not required for Raven).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )

    if not args.config.is_file():
        print(
            f"Config file not found: {args.config}\n"
            "Generate one with: python -m pwa_tools.init_config pwa_config.yml",
            file=sys.stderr,
        )
        return 1

    config = PwaConfig.from_yaml(args.config)
    result = run_step0(config, generate_wetlands=args.wetlands)

    print()
    print(f"Depression depths : {result.depression_depths}")
    print(f"Depression raster : {result.depression_raster}")
    if result.wetlands is not None:
        print(f"Wetland polygons  : {result.wetlands}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
