"""User-facing CLI prompts — isolates all ``input()`` calls from the library.

Every other module in pwa_tools is free of interactive prompts.
The runner chooses between :func:`prompt_for_config` (interactive) and
:meth:`PwaConfig.from_yaml <pwa_tools.config.PwaConfig.from_yaml>` (automated).

Replaces ``snake_case`` (lines 51-56), ``hydrocon_usr_input`` (lines 58-101),
and the interactive portion of ``project_setup`` (lines 206-240) from the
god file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pwa_tools.config import PwaConfig


def snake_case(s: str) -> str:
    """Convert a string to snake_case (spaces and hyphens become underscores)."""
    return s.replace(" ", "_").replace("-", "_").lower()


def _prompt_string(description: str, default: str | None = None) -> str:
    """Prompt for a string value with optional default. Loops until non-empty."""
    suffix = f" (default: {default})" if default else ""
    user_input = input(f"Enter {description}{suffix}: ").strip()
    if user_input == "" and default is not None:
        print(f"No {description} provided. Default applied ('{default}').")
        return default
    while user_input == "":
        user_input = input(f"A value is required. Enter {description}: ").strip()
    return snake_case(user_input)


def _prompt_filename(description: str, default: str | None = None) -> str | list[str]:
    """Prompt for a filename (no extension). Supports comma-separated lists for LiDAR.

    Returns a single string for most inputs, or a list of strings if the user
    enters comma-separated values for a LiDAR DEM prompt.
    """
    prompt = f"Enter {description} filename (e.g., '{default}'). "
    if "LiDAR" in description:
        prompt += "If entering multiple raster files, separate with commas: "

    filename = input(prompt)

    if "LiDAR" in description and "," in filename:
        filenames = [f.strip() for f in filename.split(",") if f.strip()]
        # BUG-010 fix: original `filenames if filenames` was truthy even
        # for ["", "", ""]. Filter empty entries out and re-check.
        if filenames:
            return filenames
        if default is not None:
            print(f"No {description} provided. Default applied ('{default}').")
            return default
        # Fall through to the single-file prompt loop below.
        filename = ""

    filename = filename.strip()
    if "." in filename:
        filename = input("Please do not include file extension. " + prompt).strip()
    if filename == "" and default is not None:
        print(f"No {description} provided. Default applied ('{default}').")
        return default
    while filename == "":
        filename = input(f"A filename is required. " + prompt).strip()
    return filename


def prompt_for_config(
    base_data_dir: Path | None = None,
    defaults: dict[str, Any] | None = None,
) -> PwaConfig:
    """Interactively collect all Step 0 configuration from the user.

    This replaces ``project_setup()`` from the god file. Returns a fully
    validated, frozen :class:`PwaConfig`.

    Parameters
    ----------
    base_data_dir
        Where input data lives. Defaults to ``./Data/`` relative to cwd.
    defaults
        Optional dict of default values for prompts. Keys:
        ``watershed``, ``clrh``, ``lidar``, ``nhn``, ``culvert``, ``crs``.
    """
    if defaults is None:
        defaults = {}

    base_data_dir = Path(base_data_dir) if base_data_dir else Path.cwd() / "Data"

    watershed_name = _prompt_string(
        "the name of your watershed",
        defaults.get("watershed", "cypress_river"),
    )

    clrh = _prompt_filename(
        "hydrofabric shapefile",
        defaults.get("clrh", "finalcat_info_v1-0"),
    )

    lidar = _prompt_filename(
        "LiDAR DEM raster",
        defaults.get("lidar", "Pembina_LiDAR_DEM"),
    )

    nhn = _prompt_filename(
        "NHN streams shapefile",
        defaults.get("nhn", "NHN_05MH000_3_0_HD_SLWATER_1"),
    )

    culvert = _prompt_filename(
        "Culvert lines shapefile (optional)",
        defaults.get("culvert", ""),
    )

    crs = input(
        f"Preferred coordinate reference system "
        f"(EPSG Code, default {defaults.get('crs', 'EPSG:3158')}): "
    ).strip()
    if not crs:
        crs = defaults.get("crs", "EPSG:3158")

    # Normalize inputs for PwaConfig
    lidar_filenames = lidar if isinstance(lidar, list) else [lidar]
    culvert_filename = culvert if culvert else None

    return PwaConfig.from_dict({
        "watershed_name": watershed_name,
        "base_data_dir": str(base_data_dir),
        "inputs": {
            "clrh_filename": clrh,
            "lidar_filenames": lidar_filenames,
            "nhn_filename": nhn,
            "crs_string": crs,
            "culvert_filename": culvert_filename,
        },
    })
