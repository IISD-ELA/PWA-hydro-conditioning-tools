"""Step 0 pipeline orchestration — the hydro-conditioning workflow.

Replaces ``PWA-hydro-conditioning-main/hydro_condition.py`` with a function
that accepts a :class:`~pwa_tools.config.PwaConfig` and calls the extracted
domain modules in sequence. No global state, no ``input()`` calls.

Usage::

    from pwa_tools.config import PwaConfig
    from pwa_tools.runner import run_step0

    config = PwaConfig.from_yaml("pwa_config.yml")
    result = run_step0(config)
    print(result.depression_depths)

Or interactively::

    from pwa_tools.ui import prompt_for_config
    from pwa_tools.runner import run_step0

    config = prompt_for_config()
    result = run_step0(config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pwa_tools.config import PwaConfig
from pwa_tools.depression import calc_depression_depths, gen_depressions_raster
from pwa_tools.io.raster import (
    clip_lidar_to_shapefile,
    merge_rasters,
    resample_lidar_raster,
)
from pwa_tools.io.shapefile import read_shapefile
from pwa_tools.projection import project_subbasins_to_lidar, project_subbasins_to_nhn
from pwa_tools.streams import clip_nhn_to_watershed
from pwa_tools.wetlands import gen_wetland_polygons

logger = logging.getLogger(__name__)


@dataclass
class Step0Result:
    """Paths to the key outputs of a Step 0 run."""

    depression_depths: Path
    depression_raster: Path
    wetlands: Optional[Path] = None


def run_step0(config: PwaConfig, generate_wetlands: bool = False) -> Step0Result:
    """Run the full Step 0 hydro-conditioning pipeline.

    Parameters
    ----------
    config
        Validated pipeline configuration (paths + input filenames + CRS).
    generate_wetlands
        If True, also generate the wetland polygons shapefile (not required
        for Raven but useful for analysis).

    Returns
    -------
    Step0Result
        Paths to the depression depths shapefile, depression raster, and
        optionally the wetlands shapefile.
    """
    logger.info("Starting Step 0 for watershed '%s'", config.watershed_name)

    # 0. Fail fast if expected input files are missing — saves the user
    #    from a 5-minute LiDAR resample crashing on a missing shapefile.
    config.validate_inputs_exist()

    # 1. Create directory structure
    config.paths.make_dirs()

    # 1b. Wipe Interim/ so stale files from a prior (possibly partial)
    #     run can't be misread as outputs of this one. Clean-first
    #     idempotency model; Processed/ is intentionally left alone.
    config.paths.clean_interim()

    # 2. Load CLRH watershed shapefile
    clrh_path = config.paths.hydrocon_raw / f"{config.inputs.clrh_filename}.shp"
    clrh_gdf = read_shapefile(clrh_path, target_crs=config.inputs.crs_string)
    logger.info("Loaded CLRH subbasins: %d features", len(clrh_gdf))

    # 3. LiDAR DEM — merge if multiple rasters, otherwise use raw
    if config.inputs.multiple_lidar_rasters:
        lidar_path = merge_rasters(
            config.inputs.lidar_filenames,
            clrh_gdf,
            config.paths.hydrocon_raw,
            config.paths.hydrocon_interim,
        )
    else:
        lidar_path = config.paths.hydrocon_raw / f"{config.inputs.lidar_filenames[0]}.tif"

    # 4. Project subbasins to LiDAR CRS
    clrh_projected, lidar_crs, crs_tag, clrh_proj_path = project_subbasins_to_lidar(
        clrh_gdf,
        config.inputs.clrh_filename,
        lidar_path,
        config.paths.hydrocon_interim,
    )

    # 5. Clip LiDAR to watershed (merge_rasters already clips if multi-raster)
    if not config.inputs.multiple_lidar_rasters:
        lidar_clipped = clip_lidar_to_shapefile(
            clrh_projected, lidar_path, config.paths.hydrocon_interim,
        )
    else:
        lidar_clipped = lidar_path

    # 6. Resample to 5m resolution
    lidar_resampled = resample_lidar_raster(lidar_clipped, resolution_m=5)

    # 7. Load NHN streams shapefile
    nhn_path = config.paths.hydrocon_raw / f"{config.inputs.nhn_filename}.shp"
    nhn_gdf = read_shapefile(nhn_path, target_crs=config.inputs.crs_string)

    # 8. Project subbasins to NHN CRS (for clipping)
    _nhn_projected_gdf, nhn_proj_path = project_subbasins_to_nhn(
        nhn_gdf, clrh_gdf, config.inputs.clrh_filename,
        config.paths.hydrocon_interim,
    )

    # 9. Clip NHN to watershed + optional culvert append
    nhn_clipped_path = clip_nhn_to_watershed(
        config.inputs.nhn_filename,
        nhn_proj_path,
        lidar_crs,
        crs_tag,
        config.paths.hydrocon_raw,
        config.paths.hydrocon_interim,
        culvert_filename=config.inputs.culvert_filename,
        culvert_target_crs=config.inputs.crs_string,
    )

    # 10. Generate depression raster
    depression_output = (
        config.paths.hydrocon_processed
        / f"{lidar_resampled.stem}_FillBurn_Deps_Corr.tif"
    )
    depression_raster = gen_depressions_raster(
        lidar_resampled, nhn_clipped_path, depression_output,
    )

    # 11. Optional: generate wetland polygons
    wetlands_path = None
    if generate_wetlands:
        wetlands_path, _wetlands_gdf = gen_wetland_polygons(
            depression_raster, config.paths.hydrocon_processed,
        )

    # 12. Calculate depression depths per subbasin
    depression_depths = calc_depression_depths(
        clrh_proj_path,
        config.watershed_name,
        depression_raster,
        clrh_projected,
        config.paths.hydrocon_processed,
    )

    logger.info("Step 0 complete. Depression depths → %s", depression_depths.name)
    return Step0Result(
        depression_depths=depression_depths,
        depression_raster=depression_raster,
        wetlands=wetlands_path,
    )
