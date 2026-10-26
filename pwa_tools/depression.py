"""Depression analysis — generate depression rasters and calculate depths per subbasin.

Replaces ``gen_depressions_raster`` (lines 587-646) and
``calc_depression_depths`` (lines 649-738) from the god file.

Key fixes:
  - ``os.chdir`` without try/finally → ``wbt_session`` context manager
  - 5 unchecked WBT calls → ``check_wbt`` on every call
  - Unchecked ``subprocess.run(["gdalwarp", ...])`` → ``check=True``
  - File handle leak in original (rasterio.open without context manager for
    bounds extraction) → proper context manager
  - All paths are ``pathlib.Path``
  - No global ``state`` reads/writes
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

from pwa_tools._wbt import check_wbt, wbt_session

logger = logging.getLogger(__name__)


def gen_depressions_raster(
    lidar_resampled_path: Path,
    nhn_projected_path: Path,
    output_path: Path,
) -> Path:
    """Generate a depression-depths raster via WBT fill-burn + correction.

    Steps:
      1. Burn streams into DEM (``fill_burn``)
      2. Subtract raw DEM from burned DEM (``raster_calculator``)
      3. Remove stray negative artefacts (``conditional_evaluation``)

    Parameters
    ----------
    lidar_resampled_path
        Clipped + resampled LiDAR DEM (``.tif``).
    nhn_projected_path
        Clipped + projected NHN streams shapefile (``.shp``).
    output_path
        Where to write the corrected depressions raster (``.tif``).

    Returns
    -------
    Path
        *output_path* (same value passed in, for chaining convenience).
    """
    lidar_resampled_path = Path(lidar_resampled_path)
    nhn_projected_path = Path(nhn_projected_path)
    output_path = Path(output_path)

    stem = lidar_resampled_path.stem
    parent = lidar_resampled_path.parent
    hc_path = parent / f"{stem}_HC.tif"
    fill_burn_path = parent / f"{stem}_FillBurn.tif"
    deps_path = parent / f"{stem}_FillBurn_Deps.tif"
    nhn_raster_path = parent / f"{nhn_projected_path.stem}.tif"

    ## Burn the streamlines into the DEM with a specified burn depth
    
       
    nhn_gdf = gpd.read_file(nhn_projected_path)

    # Get the transform, shape, and CRS from the reference DEM
    with rasterio.open(lidar_resampled_path) as ref:
        transform = ref.transform
        out_shape = (ref.height, ref.width)
        crs = ref.crs

    # Generate a rasterized version of the NHN streams for burning into the DEM
    burned = rasterio.features.rasterize(
        [(geom, 1) for geom in nhn_gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    profile = {
    "driver": "GTiff",
    "height": out_shape[0],
    "width": out_shape[1],
    "count": 1,
    "dtype": "uint8",
    "crs": crs,
    "transform": transform,
    "nodata": 0,
    }

    with rasterio.open(nhn_raster_path, "w", **profile) as dst:
        dst.write(burned, 1)

    # Apply the NHN burn to the DEM by subtracting a fixed burn depth
    with rasterio.open(lidar_resampled_path) as dem, rasterio.open(nhn_raster_path) as nhn:
        dem_data = dem.read(1).astype('float32')
        nhn_data = nhn.read(1)
        profile = dem.profile
        burn_depth = 10
        dem_data[(nhn_data > 0) & nhn_data != nhn.nodata] -= burn_depth

    profile.update(dtype="float32")
    with rasterio.open(hc_path, "w", **profile) as dst:
        dst.write(dem_data, 1)

    # Fill depressions while "fixing flats" to better resolve flow accumulation

    with wbt_session() as wbt:
        check_wbt(
            wbt.fill_depressions(
                dem=str(hc_path),
                fix_flats = True,
                output=str(fill_burn_path),
            ),
            "fill_depressions",
        )

        # Subtract the original DEM to get only depression depths
        check_wbt(
            wbt.raster_calculator(
                output=str(deps_path),
                statement=f"'{fill_burn_path}' - '{lidar_resampled_path}'",
            ),
            "raster_calculator",
        )

        # Remove any artificial negative values from the depression raster
        check_wbt(
            wbt.conditional_evaluation(
                i=str(deps_path),
                output=str(output_path),
                statement="value < 0.0",
                true=0.0,
                false=str(deps_path),
            ),
            "conditional_evaluation",
        )

    logger.info("Generated depressions raster → %s", output_path.name)
    return output_path, fill_burn_path


def calc_depression_depths(
    clrh_proj_lidar_path: Path,
    watershed_name: str,
    depressions_raster_path: Path,
    clrh_gdf: gpd.GeoDataFrame,
    processed_dir: Path,
    resolution_m: float = 5.0,
) -> Path:
    """Calculate per-subbasin depression depths from zonal statistics.

    Steps:
      1. Rasterize CLRH subbasin polygons (``vector_polygons_to_raster``)
      2. Align subbasin raster to depression raster (gdalwarp)
      3. Compute zonal statistics (``zonal_statistics``)
      4. Derive Deps_Depth_mm and Deps_Vol_m3 columns
      5. Write enriched shapefile

    Parameters
    ----------
    clrh_proj_lidar_path
        Path to the CLRH subbasins shapefile projected to LiDAR CRS (``.shp``).
    watershed_name
        Used to name the zonal stats output file.
    depressions_raster_path
        Path to the corrected depressions raster (``.tif``).
    clrh_gdf
        Subbasins GeoDataFrame (will be enriched with depth/volume columns).
    processed_dir
        Output directory for final products.

    Returns
    -------
    Path
        Path to the output depression-depths shapefile.
    """
    clrh_proj_lidar_path = Path(clrh_proj_lidar_path)
    depressions_raster_path = Path(depressions_raster_path)
    processed_dir = Path(processed_dir)

    clrh_raster_path = clrh_proj_lidar_path.parent / f"{clrh_proj_lidar_path.stem}_raster.tif"
    clrh_aligned_path = clrh_proj_lidar_path.parent / f"{clrh_proj_lidar_path.stem}_raster_aligned.tif"
    zonal_stats_path = processed_dir / f"ZonalStats_{watershed_name}.html"

    # 1. Rasterize subbasin polygons
    with wbt_session() as wbt:
        check_wbt(
            wbt.vector_polygons_to_raster(
                i=str(clrh_proj_lidar_path),
                output=str(clrh_raster_path),
                field="FID",
                nodata=True,
                cell_size=resolution_m,
            ),
            "vector_polygons_to_raster",
        )

    # 2. Align subbasin raster to depression raster extents (only if CRS differs)
    with rasterio.open(depressions_raster_path) as dep_src:
        dep_crs = dep_src.crs
        bounds = dep_src.bounds

    with rasterio.open(clrh_raster_path) as clrh_src:
        clrh_crs = clrh_src.crs

    if dep_crs != clrh_crs:
        logger.info("CRS mismatch — reprojecting subbasin raster to match depressions raster")
        subprocess.run(
            [
                "gdalwarp",
                "-r", "near",
                "-overwrite",
                "-tr", str(resolution_m), str(resolution_m),
                "-te",
                str(bounds.left), str(bounds.bottom),
                str(bounds.right), str(bounds.top),
                str(clrh_raster_path),
                str(clrh_aligned_path),
            ],
            check=True,
        )
    else:
        clrh_aligned_path = clrh_raster_path

    # 3. Zonal statistics
    with wbt_session() as wbt:
        check_wbt(
            wbt.zonal_statistics(
                i=str(depressions_raster_path),
                features=str(clrh_aligned_path),
                stat="total",
                out_table=str(zonal_stats_path),
            ),
            "zonal_statistics",
        )

    # 4. Derive depression depth and volume columns
    zonal_stats_df = pd.read_html(str(zonal_stats_path), flavor="bs4")[0]
    clrh_gdf = clrh_gdf.copy()
    clrh_gdf["Deps_Depth_mm"] = zonal_stats_df["Mean"] * 1000
    clrh_gdf["Deps_Vol_m3"] = clrh_gdf["Deps_Depth_mm"] * clrh_gdf["BasArea"]

    # 5. Write output shapefile
    output_path = processed_dir / "CLRH_basins_depression_depths.shp"
    clrh_gdf.to_file(output_path)

    logger.info("Depression depths → %s (%d subbasins)", output_path.name, len(clrh_gdf))
    return output_path
