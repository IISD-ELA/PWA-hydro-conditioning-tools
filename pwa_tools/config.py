"""PwaConfig dataclass — replacement for the global ``state`` singleton.

This module stands up alongside the existing ``state`` singleton during the
Phase 0/2 cleanup. It is not yet wired into the existing Step 0 functions —
those still read ``pwa_tools.state.*``. Migration happens in Phase 2 as each
function is extracted into its new module home.

See ``project-review/state-field-map.md`` for the analysis driving this design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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

    def __post_init__(self) -> None:
        if not self.watershed_name:
            raise ValueError("watershed_name is required")

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
        return cls(watershed_name=watershed_name, paths=paths, inputs=inputs)

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

        Checks every path returned by :meth:`expected_input_files`. Raises
        :class:`FileNotFoundError` listing all missing paths in one message
        rather than failing on the first one — saves the user three round
        trips when their input directory is partially populated.
        """
        missing = [p for p in self.expected_input_files() if not p.is_file()]
        if missing:
            bullet_list = "\n  - ".join(str(p) for p in missing)
            raise FileNotFoundError(
                "Step 0 cannot start; the following expected input files are "
                f"missing:\n  - {bullet_list}\n"
                f"Place them in {self.paths.hydrocon_raw} (or fix the "
                "filenames in your config) and re-run."
            )
