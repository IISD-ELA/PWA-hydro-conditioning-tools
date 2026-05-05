"""CLI: ``python -m pwa_tools.init_config <output.yml>``.

Bridges the legacy interactive UX (input() prompts) with the new
declarative ``pwa_config.yml`` workflow consumed by
:func:`pwa_tools.runner.run_step0`. Run once per watershed; the
resulting YAML can be edited, version-controlled, and re-used.

Example::

    $ python -m pwa_tools.init_config pwa_config.yml
    Enter the name of your watershed (e.g. cypress_river): grassmere
    Enter hydrofabric shapefile filename (e.g. finalcat_info_v1-0): ...
    ...
    Wrote pwa_config.yml
    $ python hydro_condition_v2.py  # consumes pwa_config.yml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from pwa_tools.ui import prompt_for_config

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pwa-init-config",
        description=(
            "Interactively build a pwa_config.yml suitable for "
            "pwa_tools.runner.run_step0."
        ),
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Where to write the YAML file.",
    )
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=None,
        help="Root data directory. Defaults to ./Data/ relative to cwd.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output_path if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(message)s",
    )

    if args.output_path.exists() and not args.force:
        print(
            f"Refusing to overwrite existing file: {args.output_path}\n"
            "Pass --force to override.",
            file=sys.stderr,
        )
        return 1

    config = prompt_for_config(base_data_dir=args.base_data_dir)
    written = config.to_yaml(args.output_path)
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
