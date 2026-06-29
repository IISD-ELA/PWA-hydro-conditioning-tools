"""PwaConfig dataclass — replacement for the global ``state`` singleton.

This module stands up alongside the existing ``state`` singleton during the
Phase 0/2 cleanup. It is not yet wired into the existing Step 0 functions —
those still read ``pwa_tools.state.*``. Migration happens in Phase 2 as each
function is extracted into its new module home.

See ``project-review/state-field-map.md`` for the analysis driving this design.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

_EPSG_PATTERN = re.compile(r"^EPSG:\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class PwaPaths:
    """Filesystem paths derived from a base data directory + watershed name.

    Pure config: construction does not touch the filesystem. Call
    :meth:`make_dirs` to materialize the directory structure on disk.

    Replaces these ``state.*`` fields: ``BS_DATA_PATH``, ``WATERSHED_PATH``,
    ``HYDROCON_PATH``, ``HYDROCON_RAW_PATH``, ``HYDROCON_INTERIM_PATH``,
    ``HYDROCON_PROCESSED_PATH``.
    """

    base_data: Path
    watershed: Path
    hydrocon: Path
    hydrocon_raw: Path
    hydrocon_interim: Path
    hydrocon_processed: Path

    @classmethod
    def from_watershed(cls, base_data: Path, watershed_name: str) -> "PwaPaths":
        """Derive the standard PWA directory layout from base + name. No I/O."""
        base_data = Path(base_data)
        watershed = base_data / watershed_name
        hydrocon = watershed / "HydroConditioning"
        return cls(
            base_data=base_data,
            watershed=watershed,
            hydrocon=hydrocon,
            hydrocon_raw=hydrocon / "Raw",
            hydrocon_interim=hydrocon / "Interim",
            hydrocon_processed=hydrocon / "Processed",
        )

    def make_dirs(self) -> None:
        """Create the directory structure on disk. Idempotent."""
        for path in (self.hydrocon_raw, self.hydrocon_interim, self.hydrocon_processed):
            path.mkdir(parents=True, exist_ok=True)

    def clean_interim(self) -> None:
        """Empty the ``Interim/`` directory at the start of a Step 0 run.

        Clean-first idempotency model: every Step 0 run starts with an
        empty ``Interim/`` so stale files from a prior — possibly
        partial — run can't be misread as outputs of the new run.
        ``Processed/`` is intentionally untouched.

        Safety: refuses to clean a path whose directory name isn't
        ``Interim``, defending against a misconfigured :class:`PwaPaths`
        from blowing away the wrong tree. No-op if ``Interim/`` doesn't
        exist — callers that want a guaranteed-empty directory on return
        should call :meth:`make_dirs` afterwards (or before).
        """
        target = self.hydrocon_interim
        if target.name != "Interim":
            raise ValueError(
                f"refusing to clean {target} — its directory name is not 'Interim'"
            )
        if not target.exists():
            return
        for item in target.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info("Cleaned Interim/: %s", target)


@dataclass(frozen=True)
class PwaInputs:
    """User-supplied input filenames and projection.

    Replaces these ``state.*`` fields: ``CLRH_FILENAME``, ``LIDAR_FILENAME``,
    ``NHN_FILENAME``, ``CULVERT_FILENAME``, ``crs_string``. The previously
    stored ``MULTIPLE_LIDAR_RASTERS`` becomes a derived property.
    """

    clrh_filename: str
    lidar_filenames: list[str]
    nhn_filename: str
    crs_string: str
    culvert_filename: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.clrh_filename:
            raise ValueError("clrh_filename is required")
        if not self.nhn_filename:
            raise ValueError("nhn_filename is required")
        if not self.lidar_filenames:
            raise ValueError("lidar_filenames must contain at least one filename")
        if not _EPSG_PATTERN.match(self.crs_string):
            raise ValueError(
                f"crs_string must match 'EPSG:NNNN' pattern, got: {self.crs_string!r}"
            )

    @property
    def multiple_lidar_rasters(self) -> bool:
        """True if more than one LiDAR raster needs merging."""
        return len(self.lidar_filenames) > 1


@dataclass(frozen=True)
class PwaConfig:
    """Top-level PWA Step 0 configuration. Frozen and fully validated.

    Replaces the global ``pwa_tools.state`` singleton.
    """

    watershed_name: str
    paths: PwaPaths
    inputs: PwaInputs
    output_res_m: float = 5.0

    def __post_init__(self) -> None:
        if not self.watershed_name:
            raise ValueError("watershed_name is required")
        if self.output_res_m <= 0:
            raise ValueError(f"output_res_m must be positive, got: {self.output_res_m}")

    @classmethod
    def from_dict(cls, data: dict) -> "PwaConfig":
        """Build from a plain dict — used by tests, programmatic callers, and
        as the intermediate step inside :meth:`from_yaml`."""
        watershed_name = data["watershed_name"]
        paths = PwaPaths.from_watershed(Path(data["base_data_dir"]), watershed_name)

        inputs_data = data["inputs"]

        # Coerce single-string lidar to single-element list for uniformity.
        # The legacy state.LIDAR_FILENAME accepted str OR list[str]; we always
        # store list[str] internally.
        lidar = inputs_data["lidar_filenames"]
        if isinstance(lidar, str):
            lidar = [lidar]

        # Treat missing/empty culvert as None — the legacy code used "" as the
        # absence sentinel. Normalize at the boundary.
        culvert = inputs_data.get("culvert_filename") or None

        inputs = PwaInputs(
            clrh_filename=inputs_data["clrh_filename"],
            lidar_filenames=lidar,
            nhn_filename=inputs_data["nhn_filename"],
            crs_string=inputs_data["crs_string"],
            culvert_filename=culvert,
        )
        output_res_m = float(data.get("output_res_m", 5.0))
        return cls(watershed_name=watershed_name, paths=paths, inputs=inputs, output_res_m=output_res_m)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PwaConfig":
        """Load configuration from a YAML file. Replaces the input() prompts."""
        import yaml  # local import keeps module-level deps light

        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def expected_input_files(self) -> list[Path]:
        """Paths the pipeline expects to find on disk before run_step0 starts.

        Returns the resolved paths to the CLRH shapefile, every LiDAR raster,
        the NHN shapefile, and (if specified) the culvert shapefile. Sidecars
        like ``.dbf``/``.shx`` are not enumerated — geopandas will fail with a
        clear error if they're missing alongside a present ``.shp``.
        """
        raw = self.paths.hydrocon_raw
        files: list[Path] = []
        files.append(raw / f"{self.inputs.clrh_filename}.shp")
        for name in self.inputs.lidar_filenames:
            files.append(raw / f"{name}.tif")
        files.append(raw / f"{self.inputs.nhn_filename}.shp")
        if self.inputs.culvert_filename:
            files.append(raw / f"{self.inputs.culvert_filename}.shp")
        return files

    def validate_inputs_exist(self) -> None:
        """Fail fast if expected input files are missing.

        Two branches: if the entire watershed Raw/ directory is absent
        (the typical "watershed_name typo" case), surface sibling
        directories under base_data_dir so the user can spot a case or
        spelling mismatch at a glance. Otherwise, list the specific
        missing files individually.
        """
        raw = self.paths.hydrocon_raw
        if not raw.is_dir():
            siblings = []
            base = self.paths.base_data
            if base.is_dir():
                siblings = sorted(p.name for p in base.iterdir() if p.is_dir())

            msg = (
                "Step 0 cannot start; the watershed input directory is "
                f"missing:\n  {raw}\n"
            )
            if siblings:
                sibling_list = "\n  - ".join(siblings)
                msg += (
                    f"\nAvailable directories in {base}:\n"
                    f"  - {sibling_list}\n"
                    "\nIf one of these is the watershed you meant, either "
                    "update 'watershed_name' in your config or rename the "
                    "on-disk directory to match."
                )
            else:
                msg += (
                    f"\nThe parent directory {base} doesn't exist or has no "
                    "subdirectories. Check 'base_data_dir' in your config."
                )
            raise FileNotFoundError(msg)

        missing = [p for p in self.expected_input_files() if not p.is_file()]
        if missing:
            bullet_list = "\n  - ".join(str(p) for p in missing)
            raise FileNotFoundError(
                "Step 0 cannot start; the following expected input files are "
                f"missing:\n  - {bullet_list}\n"
                f"Place them in {raw} (or fix the "
                "filenames in your config) and re-run."
            )

    def to_dict(self) -> dict:
        """Serialize back to the same dict shape :meth:`from_dict` consumes.

        Round-trip property: ``PwaConfig.from_dict(c.to_dict()) == c`` (with
        the caveat that ``base_data_dir`` is reconstructed from
        ``paths.base_data``, so the watershed-name suffix is stripped from
        the path tail consistently with ``from_dict``'s expectation).
        """
        return {
            "watershed_name": self.watershed_name,
            "base_data_dir": str(self.paths.base_data),
            "output_res_m": self.output_res_m,
            "inputs": {
                "clrh_filename": self.inputs.clrh_filename,
                "lidar_filenames": list(self.inputs.lidar_filenames),
                "nhn_filename": self.inputs.nhn_filename,
                "culvert_filename": self.inputs.culvert_filename,
                "crs_string": self.inputs.crs_string,
            },
        }

    def to_yaml(self, path: Path | str) -> Path:
        """Write this config to *path* as YAML. Returns the resolved path.

        Output is human-readable: keys preserved in insertion order, lists
        rendered block-style (``- name``), no Python tags. Parent directory
        is created if needed.
        """
        import yaml  # local import keeps module-level deps light

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                self.to_dict(),
                sort_keys=False,
                default_flow_style=False,
            )
        )
        return path
