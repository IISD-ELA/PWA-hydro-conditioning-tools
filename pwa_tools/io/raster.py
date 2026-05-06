"""Raster I/O — resample, clip, and merge LiDAR DEM rasters.

Replaces ``resample_lidar_raster`` (lines 265-286),
``clip_lidar_to_shapefile`` (lines 415-456), and ``merge_rasters``
(lines 843-1115) from the god file.

Key changes from originals:
  - All paths are ``pathlib.Path`` (no string concatenation with ``+``)
  - ``subprocess.run`` uses ``check=True`` — gdalwarp failures now raise
    ``subprocess.CalledProcessError`` instead of silently continuing
  - ``gdal.BuildVRT`` / ``gdal.Warp`` return values are checked
  - No global ``state`` reads — callers pass directories explicitly
  - ``fill_nodata_gaps`` and ``get_raster_resolution`` promoted to module-level
    from nested helpers inside ``merge_rasters``
  - ``save_state()`` / ``state.LAST_FUNCTION_RUN`` writes removed
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from rasterio.mask import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


def resample_lidar_raster(input_path: Path, resolution_m: float) -> Path:
    """Resample a raster to *resolution_m* using gdalwarp with cubic interpolation.

    The output file is written next to the input with a ``_resample_{N}m``
    suffix. Returns the full output path.

    Raises
    ------
    subprocess.CalledProcessError
        If gdalwarp exits with a non-zero return code.
    FileNotFoundError
        If *input_path* does not exist.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raster not found: {input_path}")

    output_path = input_path.parent / f"{input_path.stem}_resample_{resolution_m}m{input_path.suffix}"

    subprocess.run(
        [
            "gdalwarp",
            "-overwrite",
            "-tr", str(resolution_m), str(resolution_m),
            "-r", "cubic",
            str(input_path),
            str(output_path),
        ],
        check=True,
    )
    logger.info("Resampled %s → %s at %sm resolution", input_path.name, output_path.name, resolution_m)
    return output_path


def clip_lidar_to_shapefile(
    gdf: gpd.GeoDataFrame,
    input_path: Path,
    output_dir: Path,
) -> Path:
    """Clip a raster to the extent of a GeoDataFrame's geometries.

    The output file is written to *output_dir* with a ``_clip`` suffix.
    The GeoDataFrame must already be projected to match the raster's CRS.

    Returns the full path to the clipped raster.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    shapes = [mapping(geom) for geom in gdf.geometry]

    with rasterio.open(input_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
    })

    output_path = output_dir / f"{input_path.stem}_clip{input_path.suffix}"

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    logger.info("Clipped %s → %s", input_path.name, output_path.name)
    return output_path


def get_raster_resolution(path: Path) -> float:
    """Return the finest pixel dimension (in CRS units) of a raster."""
    with rasterio.open(path) as src:
        return min(abs(src.res[0]), abs(src.res[1]))


def fill_nodata_gaps(
    input_path: Path,
    output_path: Path,
    buffer_px: int = 50,
) -> None:
    """Fill nodata pixels using nearest-neighbor interpolation.

    Processes the raster block-by-block with a *buffer_px*-pixel overlap
    to handle gap edges correctly without loading the full raster into memory.
    """
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999
        profile.update(nodata=nodata)

        with rasterio.open(output_path, "w", **profile) as dst:
            for _ji, window in src.block_windows(1):
                row_off, col_off = window.row_off, window.col_off
                height, width = window.height, window.width

                buffered_window = Window(
                    max(0, col_off - buffer_px),
                    max(0, row_off - buffer_px),
                    min(width + 2 * buffer_px, src.width - max(0, col_off - buffer_px)),
                    min(height + 2 * buffer_px, src.height - max(0, row_off - buffer_px)),
                )

                data = src.read(1, window=buffered_window)
                nodata_mask = data == nodata

                if not np.any(nodata_mask):
                    # Trim buffer back to original block before writing
                    r_start = row_off - buffered_window.row_off
                    c_start = col_off - buffered_window.col_off
                    dst.write(data[r_start:r_start + height, c_start:c_start + width], 1, window=window)
                    continue

                inds = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)
                filled = data.copy()
                filled[nodata_mask] = data[tuple(inds[:, nodata_mask])]

                r_start = row_off - buffered_window.row_off
                c_start = col_off - buffered_window.col_off
                dst.write(filled[r_start:r_start + height, c_start:c_start + width], 1, window=window)

    logger.info("Filled nodata gaps: %s → %s", input_path.name, output_path.name)


def merge_rasters(
    lidar_filenames: list[str],
    gdf: gpd.GeoDataFrame,
    raw_dir: Path,
    interim_dir: Path,
) -> Path:
    """Merge multiple LiDAR DEMs into one aligned, gap-filled raster.

    Steps:
      1. Clip each raster to the watershed boundary
      2. Find the highest-resolution raster's CRS
      3. Reproject all clipped rasters to that CRS
      4. Merge via GDAL VRT + Warp (averaging overlapping areas)
      5. Fill nodata gaps via nearest-neighbor interpolation

    Parameters
    ----------
    lidar_filenames
        Basenames (no extension) of the LiDAR ``.tif`` files in *raw_dir*.
    gdf
        Watershed boundary GeoDataFrame used for clipping.
    raw_dir
        Directory containing the raw LiDAR ``.tif`` files.
    interim_dir
        Directory for intermediate outputs (clipped, reprojected, merged).

    Returns
    -------
    Path
        Full path to the final filled, merged raster.

    Raises
    ------
    RuntimeError
        If ``gdal.BuildVRT`` or ``gdal.Warp`` fails.
    """
    from osgeo import gdal  # local import — GDAL is heavy

    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)

    # 1. Clip each raster to the watershed boundary
    for filename in lidar_filenames:
        input_path = raw_dir / f"{filename}.tif"

        with rasterio.open(input_path) as src:
            input_crs = src.crs

        gdf_projected = gdf.to_crs(input_crs)
        shapes = [mapping(geom) for geom in gdf_projected.geometry]

        with rasterio.open(input_path) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": src.nodata,
        })

        clip_path = interim_dir / f"{filename}_clip.tif"
        with rasterio.open(clip_path, "w", **out_meta) as dest:
            dest.write(out_image)

    logger.info("Clipped %d rasters to watershed boundary", len(lidar_filenames))

    # 2. Find the highest-resolution raster's CRS
    clipped_paths = sorted(interim_dir.glob("*_clip.tif"))
    highest_res_path = min(clipped_paths, key=get_raster_resolution)
    highest_res = get_raster_resolution(highest_res_path)

    with rasterio.open(highest_res_path) as ref_src:
        target_crs = ref_src.crs

    # 3. Reproject all to the highest-resolution CRS
    reprojected_paths: list[str] = []

    for clip_path in clipped_paths:
        with rasterio.open(clip_path) as src:
            if src.crs != target_crs:
                out_path = interim_dir / f"{clip_path.stem}_reprojected.tif"
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds,
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                })
                with rasterio.open(out_path, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.bilinear,
                            dst_nodata=src.nodata,
                        )
                reprojected_paths.append(str(out_path))
            else:
                reprojected_paths.append(str(clip_path))

    logger.info("Reprojected %d rasters to %s", len(reprojected_paths), target_crs)

    # 4. Merge via GDAL VRT + Warp
    merged_name = "merged_average_dem"
    out_path_merged = interim_dir / f"{merged_name}.tif"
    out_path_vrt = interim_dir / "merged_virtual_raster.vrt"

    vrt = gdal.BuildVRT(str(out_path_vrt), reprojected_paths)
    if vrt is None:
        raise RuntimeError(f"gdal.BuildVRT failed for {reprojected_paths}")

    result = gdal.Warp(
        str(out_path_merged),
        vrt,
        xRes=highest_res,
        yRes=highest_res,
        resampleAlg="average",
        options=gdal.WarpOptions(
            format="GTiff",
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
        ),
    )
    if result is None:
        raise RuntimeError(f"gdal.Warp failed writing {out_path_merged}")
    vrt = None  # close VRT

    logger.info("Merged %d rasters → %s", len(reprojected_paths), out_path_merged.name)

    # 5. Fill nodata gaps
    filled_path = interim_dir / f"{merged_name}_filled.tif"
    fill_nodata_gaps(out_path_merged, filled_path, buffer_px=50)

    return filled_path
