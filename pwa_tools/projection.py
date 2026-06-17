"""CRS projection — reproject subbasins to match NHN streams or LiDAR DEMs.

Replaces ``project_crs_subbasins_to_nhn`` (lines 289-335) and
``project_subbasins_to_lidar`` (lines 338-377) from the god file.

Changes: pathlib paths, explicit output_dir param, no state reads/writes,
logging instead of print.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import rasterio

logger = logging.getLogger(__name__)


def _sanitize_crs_for_filename(crs) -> str:
    """Convert a CRS to a short alphanumeric string safe for filenames.

    Used to suffix projected shapefiles with their CRS (e.g.
    ``finalcat_info_v1-0_projected_EPSG3158``).
    """
    alnum = "".join(c for c in str(crs) if c.isalnum())
    return alnum[:10] if len(alnum) > 10 else alnum


def project_subbasins_to_nhn(
    nhn_gdf: gpd.GeoDataFrame,
    subbasins_gdf: gpd.GeoDataFrame,
    subbasins_filename: str,
    output_dir: Path,
) -> tuple[gpd.GeoDataFrame, Path]:
    """Reproject *subbasins_gdf* to match *nhn_gdf*'s CRS. Write result to *output_dir*.

    Returns (projected_gdf, output_shapefile_path).
    """
    output_dir = Path(output_dir)
    target_crs = nhn_gdf.crs
    projected = subbasins_gdf.to_crs(target_crs)

    crs_tag = _sanitize_crs_for_filename(target_crs)
    output_path = output_dir / f"{subbasins_filename}_projected_{crs_tag}.shp"
    projected.to_file(output_path)

    logger.info(
        "Projected %s to %s → %s",
        subbasins_filename, target_crs, output_path.name,
    )
    return projected, output_path


def project_subbasins_to_lidar(
    gdf: gpd.GeoDataFrame,
    gdf_filename: str,
    lidar_path: Path,
    output_dir: Path,
) -> tuple[gpd.GeoDataFrame, object, str, Path]:
    """Reproject *gdf* to match the CRS of *lidar_path*. Write result to *output_dir*.

    Returns (projected_gdf, lidar_crs, crs_alnum_tag, output_shapefile_path).
    The extra return values are consumed by downstream functions that need
    the LiDAR CRS for further reprojection.
    """
    lidar_path = Path(lidar_path)
    output_dir = Path(output_dir)

    with rasterio.open(lidar_path) as src:
        lidar_crs = src.crs

    projected = gdf.to_crs(lidar_crs)

    crs_tag = _sanitize_crs_for_filename(lidar_crs)
    output_path = output_dir / f"{gdf_filename}_projected_{crs_tag}.shp"
    projected.to_file(output_path)

    logger.info(
        "Projected %s to %s → %s",
        gdf_filename, lidar_crs, output_path.name,
    )
    return projected, lidar_crs, crs_tag, output_path
