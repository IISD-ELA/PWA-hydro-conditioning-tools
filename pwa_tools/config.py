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
    processing_res_m: float = 2.0  # intermediate LiDAR resampling resolution
    # Wetland filter thresholds — override in config YAML to suit the dataset.
    # Defaults match the module-level constants in pwa_tools.wetlands.
    depth_llim: float = 0.1    # minimum depression depth (m)
    area_llim: float = 4000.0  # minimum wetland area (m²)
    volume_llim: float = 30.0  # minimum wetland storage volume (m³)
    # Optional shared DEM directory — LiDAR files are read directly from here
    # and are never staged/moved into hydrocon_raw.
    dem_source_dir: Optional[Path] = None
    # Set to False to skip step 12 (depression-depth summary per subbasin).
    summarize_by_subbasin: bool = True

    def __post_init__(self) -> None:
        if not self.watershed_name:
            raise ValueError("watershed_name is required")
        if self.output_res_m <= 0:
            raise ValueError(f"output_res_m must be positive, got: {self.output_res_m}")
        if self.processing_res_m <= 0:
            raise ValueError(f"processing_res_m must be positive, got: {self.processing_res_m}")

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
        processing_res_m = float(data.get("processing_res_m", 2.0))
        depth_llim = float(data.get("depth_llim", 0.1))
        area_llim = float(data.get("area_llim", 4000.0))
        volume_llim = float(data.get("volume_llim", 30.0))
        dem_source_dir_raw = data.get("dem_source_dir")
        dem_source_dir = Path(dem_source_dir_raw) if dem_source_dir_raw else None
        summarize_by_subbasin = bool(data.get("summarize_by_subbasin", True))
        return cls(
            watershed_name=watershed_name,
            paths=paths,
            inputs=inputs,
            output_res_m=output_res_m,
            processing_res_m=processing_res_m,
            depth_llim=depth_llim,
            area_llim=area_llim,
            volume_llim=volume_llim,
            dem_source_dir=dem_source_dir,
            summarize_by_subbasin=summarize_by_subbasin,
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PwaConfig":
        """Load configuration from a YAML file. Replaces the input() prompts."""
        import yaml  # local import keeps module-level deps light

        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @property
    def lidar_dir(self) -> Path:
        """Source directory for LiDAR ``.tif`` files.

        Returns ``dem_source_dir`` when specified; otherwise falls back to
        ``paths.hydrocon_raw``.  LiDAR files are always read directly from
        this directory and are never moved or copied into the watershed
        hierarchy.
        """
        if self.dem_source_dir is not None:
            return self.dem_source_dir
        return self.paths.hydrocon_raw

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
            files.append(self.lidar_dir / f"{name}.tif")
        files.append(raw / f"{self.inputs.nhn_filename}.shp")
        if self.inputs.culvert_filename:
            files.append(raw / f"{self.inputs.culvert_filename}.shp")
        return files

    def _input_search_dirs(self) -> list[Path]:
        """Ordered list of directories to search when staging raw inputs.

        Search order (each directory is checked flat — not recursively):
        1. ``hydrocon_raw`` — files already in place, nothing to move.
        2. Every subdirectory of the watershed folder (at any depth) whose
           **name** contains ``"condition"`` or ``"data"`` (lower-cased for
           comparison).
        3. The watershed folder root.
        4. ``base_data_dir`` root.
        """
        dirs: list[Path] = [self.paths.hydrocon_raw]
        watershed = self.paths.watershed
        if watershed.is_dir():
            for d in sorted(watershed.rglob("*")):
                if d.is_dir():
                    name_lower = d.name.lower()
                    if "condition" in name_lower or "data" in name_lower:
                        if d not in dirs:
                            dirs.append(d)
        if watershed not in dirs:
            dirs.append(watershed)
        if self.paths.base_data not in dirs:
            dirs.append(self.paths.base_data)
        return dirs

    def stage_inputs(self) -> None:
        """Search the data hierarchy for raw inputs and move them into Raw/.

        For each expected input file the method walks the search directories
        returned by :meth:`_input_search_dirs`.  Files already present in
        ``hydrocon_raw`` are left untouched.  Files found elsewhere are moved
        to ``hydrocon_raw``; for shapefiles (``.shp``) every sidecar sharing
        the same stem (e.g. ``.dbf``, ``.shx``, ``.prj``, ``.cpg``) is moved
        alongside.

        Raises :exc:`FileNotFoundError` if the watershed directory does not
        exist (surfaces sibling directories so the user can spot a typo) or if
        one or more expected files cannot be located anywhere in the hierarchy.
        """
        watershed = self.paths.watershed
        if not watershed.is_dir():
            siblings = []
            base = self.paths.base_data
            if base.is_dir():
                siblings = sorted(p.name for p in base.iterdir() if p.is_dir())
            msg = (
                "Step 0 cannot start; the watershed directory is missing:"
                f"\n  {watershed}\n"
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

        self.paths.make_dirs()

        # When dem_source_dir is configured, validate it early — fail fast
        # before any expensive pipeline work.
        if self.dem_source_dir is not None:
            if not self.dem_source_dir.is_dir():
                logger.error(
                    "dem_source_dir does not exist: %s", self.dem_source_dir
                )
                raise FileNotFoundError(
                    f"dem_source_dir does not exist: {self.dem_source_dir}\n"
                    "Check the 'dem_source_dir' path in your config."
                )
            missing_lidar = [
                f"{name}.tif"
                for name in self.inputs.lidar_filenames
                if not (self.dem_source_dir / f"{name}.tif").is_file()
            ]
            if missing_lidar:
                bullet_list = "\n  - ".join(missing_lidar)
                logger.error(
                    "LiDAR files missing from dem_source_dir %s: %s",
                    self.dem_source_dir, missing_lidar,
                )
                raise FileNotFoundError(
                    "Step 0 cannot start; the following LiDAR files are missing "
                    f"from dem_source_dir ({self.dem_source_dir}):\n  - {bullet_list}\n"
                    "Place them in dem_source_dir or fix the filenames in your config."
                )

        raw = self.paths.hydrocon_raw
        search_dirs = self._input_search_dirs()
        missing: list[str] = []

        for expected in self.expected_input_files():
            if expected.is_file():
                continue  # already staged

            found: Path | None = None
            for search_dir in search_dirs:
                candidate = search_dir / expected.name
                if candidate.is_file():
                    found = candidate
                    break

            if found is None:
                missing.append(expected.name)
                continue

            # Move the primary file to Raw/
            dest = raw / expected.name
            shutil.move(str(found), dest)
            logger.info("Staged %s -> %s", found, dest)

            # For shapefiles, carry sidecars from the same source directory
            if expected.suffix.lower() == ".shp":
                stem = expected.stem
                for sidecar in found.parent.glob(f"{stem}.*"):
                    if sidecar.suffix.lower() != ".shp" and sidecar.is_file():
                        shutil.move(str(sidecar), raw / sidecar.name)
                        logger.info("Staged sidecar %s -> %s", sidecar, raw / sidecar.name)

        if missing:
            bullet_list = "\n  - ".join(missing)
            raise FileNotFoundError(
                "Step 0 cannot start; the following input files could not be "
                f"found anywhere in the search hierarchy:\n  - {bullet_list}\n"
                f"Place them under {watershed} (or fix the filenames in your "
                "config) and re-run."
            )

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
            "processing_res_m": self.processing_res_m,
            "depth_llim": self.depth_llim,
            "area_llim": self.area_llim,
            "volume_llim": self.volume_llim,
            "dem_source_dir": str(self.dem_source_dir) if self.dem_source_dir is not None else None,
            "summarize_by_subbasin": self.summarize_by_subbasin,
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
