"""WhiteboxTools wrapper with error checking and safe working-directory handling.

Provides:
  - :func:`wbt_session`: context manager that initializes WBT, yields it, and
    restores the working directory even if a WBT call crashes.
  - :func:`check_wbt`: validates WBT return codes and raises on failure.
  - :class:`WBTError`: exception for WBT tool failures.

Replaces the scattered WBT init + ``os.chdir`` patterns in the god file:
  - ``clip_nhn_to_watershed``  (lines 468-495)
  - ``gen_depressions_raster`` (lines 593-638)
  - ``calc_depression_depths`` (lines 659-733)

All three functions repeat the same boilerplate:
  1. ``this_dir = os.path.dirname(os.path.abspath(__file__))``
  2. ``original_dir = os.getcwd()``
  3. ``wbt = WhiteboxTools(); wbt.set_whitebox_dir(...)``
  4. call wbt tool (discard return value)
  5. ``os.chdir(original_dir)``  # without try/finally

This module centralizes that into one context manager with proper cleanup.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


def _get_wbt_class():
    """Import WhiteboxTools from the best available source.

    Priority:
      1. Bundled ``pwa_tools/WBT/whitebox_tools.py`` (existing vendor, Windows binaries)
      2. ``whitebox`` pip package (cross-platform, auto-downloads correct binary)
      3. Raise with clear install instructions

    The bundled and pip versions share the same upstream codebase (Dr. John
    Lindsay, MIT license) and expose the same ``run_tool`` API.
    """
    try:
        from .WBT.whitebox_tools import WhiteboxTools
        return WhiteboxTools
    except (ImportError, OSError):
        pass

    try:
        from whitebox import WhiteboxTools  # type: ignore[import-untyped]
        return WhiteboxTools
    except ImportError:
        raise ImportError(
            "WhiteboxTools not found. Either:\n"
            "  1. Install the 'whitebox' pip package:  pip install whitebox\n"
            "  2. Or ensure pwa_tools/WBT/ contains the correct platform binary.\n"
            "See project-review/local-run-guide.md for platform-specific setup."
        )


class WBTError(RuntimeError):
    """Raised when a WhiteboxTools operation returns a non-zero exit code."""

    def __init__(self, tool_name: str, return_code: int) -> None:
        self.tool_name = tool_name
        self.return_code = return_code
        super().__init__(
            f"WhiteboxTools '{tool_name}' failed (return code {return_code}). "
            f"Check the WhiteboxTools output above for details."
        )


def check_wbt(return_code: int, tool_name: str) -> None:
    """Validate a WhiteboxTools return code; raise :class:`WBTError` on failure.

    WhiteboxTools ``run_tool`` returns:
      - 0: success
      - 1: error (details sent to callback / stdout)
      - 2: cancelled by user

    Usage::

        ret = wbt.fill_burn(dem=..., streams=..., output=...)
        check_wbt(ret, "fill_burn")
    """
    if return_code != 0:
        raise WBTError(tool_name, return_code)


@contextmanager
def wbt_session() -> Generator:
    """Context manager: initialize WBT, yield it, restore working directory.

    ``WhiteboxTools.run_tool()`` internally calls ``os.chdir(self.exe_path)``.
    Without try/finally, a crash mid-tool leaves the process in the WBT
    binary directory and breaks all subsequent relative-path operations.
    This context manager guarantees the caller's working directory is
    restored on exit — normal or exceptional.

    Usage::

        from pwa_tools._wbt import wbt_session, check_wbt

        with wbt_session() as wbt:
            check_wbt(wbt.clip(i=..., clip=..., output=...), "clip")
            check_wbt(wbt.fill_burn(dem=..., streams=..., output=...), "fill_burn")
        # cwd is restored here, even if WBTError or any other exception fired
    """
    original_dir = os.getcwd()
    try:
        WBTClass = _get_wbt_class()
        wbt = WBTClass()

        # Point WBT at the bundled binary directory when using the bundled version.
        # The pip-installed whitebox package manages its own binary location, so
        # set_whitebox_dir is only needed for the vendor path.
        this_dir = Path(__file__).resolve().parent
        wbt_dir = this_dir / "WBT"
        if wbt_dir.is_dir():
            wbt.set_whitebox_dir(str(wbt_dir))

        yield wbt
    finally:
        os.chdir(original_dir)
