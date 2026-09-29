"""Wetland polygon generation from depression rasters.

Replaces ``gen_wetland_polygons`` (lines 741-839) from the god file.

Standalone module — no WBT, no subprocess, no cross-module dependencies.
Pure rasterio + numpy + scipy analysis.

Changes: pathlib paths, explicit output directory, no state reads/writes,
logging instead of print.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pwa_tools._wbt import wbt_session, check_wbt
import rasterio
import rasterio.features
from scipy.ndimage import label, maximum as label_maximum
from shapely.geometry import shape as shapely_shape

logger = logging.getLogger(__name__)

# Thresholds for filtering insignificant wetlands
_DEPTH_THRESHOLD_M = 0.1   # ignore depressions shallower than 10 cm
_AREA_THRESHOLD_M2 = 4000    # minimum wetland area
_VOLUME_THRESHOLD_M3 = 30   # minimum wetland storage volume



def gen_direction_raster(
    dem_path: Path,
    output_dir: Path,
) -> Path:
    """Generate a D8 flow-direction raster from an elevation DEM via WBT ``d8_pointer``.

    Parameters
    ----------
    dem_path
        Elevation DEM raster, float-valued (``.tif``).
    output_dir
        Directory where to write the D8 pointer raster (``.tif``).

    Returns
    -------
    Path
        Full path to the generated D8 pointer raster (inside *output_dir*).
    """
    dem_path = Path(dem_path)
    stem = dem_path.stem
    output_path = Path(output_dir) / (stem + "_d8.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wbt_session() as wbt:
        check_wbt(
            wbt.d8_pointer(
                dem=str(dem_path),
                output=str(output_path),
            ),
            "d8_pointer",
        )

    logger.info("Generated D8 direction raster → %s", output_path.name)
    return output_path

def gen_accum_raster(
    direction_path: Path,
    output_dir: Path,
) -> Path:
    """Generate a flow-accumulation raster from a D8 pointer via WBT ``d8_flow_accumulation``.

    Parameters
    ----------
    direction_path
        D8 flow-direction (pointer) raster, e.g. from ``gen_direction_raster`` (``.tif``).
    output_dir
        Directory where to write the flow-accumulation raster (``.tif``).

    Returns
    -------
    Path
        *output_path* (same value passed in, for chaining convenience).
    """
    direction_path = Path(direction_path)
    stem = direction_path.stem
    output_path = Path(output_dir) / (stem + "_accum.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wbt_session() as wbt:
        check_wbt(
            wbt.d8_flow_accumulation(
                i=str(direction_path),
                output=str(output_path),
                pntr=True,  # input is already a D8 pointer, not a raw DEM
            ),
            "d8_flow_accumulation",
        )

    logger.info("Generated flow accumulation raster → %s", output_path.name)
    return output_path


def gen_wetland_polygons(
    depressions_raster_path: Path,
    output_dir: Path,
    *,
    depth_llim: float = _DEPTH_THRESHOLD_M,
    area_llim: float = _AREA_THRESHOLD_M2,
    volume_llim: float = _VOLUME_THRESHOLD_M3,
    accum_raster_path: Path | None = None,
) -> tuple[Path, gpd.GeoDataFrame]:
    """Generate wetland polygons with area/volume/depth statistics from a depression raster.

    Steps:
      1. Threshold the depression raster at ``depth_llim``
         (default ``_DEPTH_THRESHOLD_M`` = 0.1 m / 10 cm; ignores shallower depressions)
      2. Label connected components (8-connectivity)
      3. Compute per-wetland area, total storage volume, and median depth
      4. Filter by area >= ``area_llim`` (default ``_AREA_THRESHOLD_M2`` = 4000 m²)
         and volume >= ``volume_llim`` (default ``_VOLUME_THRESHOLD_M3`` = 30 m³)
      5. Vectorize remaining polygons
      6. Write stats CSV + shapefile

    The three threshold parameters are optional keyword-only arguments whose
    defaults are the module-level constants ``_DEPTH_THRESHOLD_M``,
    ``_AREA_THRESHOLD_M2``, and ``_VOLUME_THRESHOLD_M3``.  Pass explicit
    values (e.g. from a :class:`~pwa_tools.config.PwaConfig`) to override
    them for a specific dataset without editing the source.

    ``accum_raster_path`` is optional. When given, it must share the
    depression raster's grid (same shape). The max flow-accumulation value
    per wetland is used to add a ``wet_cat_m2`` column estimating the
    contributing catchment area.

    See ``test_wetlands_docstring_cites_live_constants``.

    Returns (shapefile_path, wetlands_gdf).
    """
    depressions_raster_path = Path(depressions_raster_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(depressions_raster_path) as src:
        depression_data = src.read(1)
        transform = src.transform
        pixel_area = abs(transform[0] * transform[4])
        nodata = src.nodata
        crs = src.crs

    # Mask nodata and select depressions above threshold
    valid_mask = (depression_data > depth_llim) & (depression_data != nodata)

    # Label connected depressions (wetlands)
    structure = np.ones((3, 3), dtype=int) # Specify 8-connectivity for labeling (diagonal and orthogonal neighbors)
    labeled_array, num_features = label(valid_mask, structure=structure) # Create labels for each "wetland" (connected pixels in the mask)
    del valid_mask

    flat_labels = labeled_array.ravel() # Flatten
    flat_depths = depression_data.ravel() # Flatten

    # Per-wetland statistics
    counts = np.bincount(flat_labels)
    areas_m2 = counts * pixel_area
    volume_sums = np.bincount(flat_labels, weights=flat_depths)
    volumes_m3 = volume_sums * pixel_area

    # Median depth per wetland
    medians = np.full(num_features + 1, np.nan)
    depths_by_label: dict[int, list[float]] = defaultdict(list)
    for label_val, depth in zip(flat_labels, flat_depths):
        if label_val == 0:
            continue
        depths_by_label[label_val].append(depth)
    for label_val, depth_list in depths_by_label.items():
        medians[label_val] = np.median(depth_list)
    del flat_labels, flat_depths, depression_data

    # Build statistics dataframe
    stats_df = pd.DataFrame({
        "wetland_id": np.arange(len(areas_m2)),
        "area_m2": areas_m2,
        "volume_m3": volumes_m3,
        "med_dep_m": medians,
    })

    # Filter out background + small features
    stats_df = stats_df.query(
        f"wetland_id != 0 and area_m2 >= {area_llim} "
        f"and volume_m3 >= {volume_llim}"
    )

    # Write stats CSV
    stats_csv = output_dir / "Wetlands_Stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    # Vectorize remaining wetland polygons
    valid_ids = set(stats_df["wetland_id"])
    valid_mask_vec = np.isin(labeled_array, list(valid_ids))

    # Zonal stats: max flow accumulation per wetland → estimated contributing catchment area
    if accum_raster_path is not None:
        accum_raster_path = Path(accum_raster_path)
        with rasterio.open(accum_raster_path) as accum_src:
            accum_data = accum_src.read(1)
            accum_pixel_area = abs(accum_src.transform[0] * accum_src.transform[4])

        if accum_data.shape != labeled_array.shape:
            raise ValueError(
                "accum_raster_path must share the depression raster's grid "
                f"(got {accum_data.shape}, expected {labeled_array.shape})"
            )

        wetland_ids = stats_df["wetland_id"].tolist()
        max_accum = label_maximum(accum_data, labeled_array, index=wetland_ids)
        stats_df["wet_cat_m2"] = np.asarray(max_accum) * accum_pixel_area
        del accum_data

        # For wetlands where the catchment area is less than the wetland area, set the catchment area to null
        stats_df.loc[stats_df["wet_cat_m2"] < stats_df["area_m2"], "wet_cat_m2"] = np.nan

    polygons = []
    labels_out = []
    for geom, val in rasterio.features.shapes(
        labeled_array.astype(np.int32),
        mask=valid_mask_vec,
        transform=transform,
    ):
        if val in valid_ids:
            polygons.append(shapely_shape(geom))
            labels_out.append(val)

    gdf = gpd.GeoDataFrame(
        {"wetland_id": labels_out, "geometry": polygons}, crs=crs,
    )
    gdf = gdf.merge(stats_df, on="wetland_id")
    # Merge polygon pieces sharing a label (8-conn label, 4-conn vectorisation) into MultiPolygons
    gdf = gdf.dissolve(by="wetland_id", aggfunc="first").reset_index()

    # Imperial unit columns (for final output only; all processing remains metric)
    gdf["area_ac"]   = gdf["area_m2"]       / 4046.8564224
    gdf["vol_acft"]  = gdf["volume_m3"]      / 1233.4818375
    gdf["med_dep_ft"] = gdf["med_dep_m"] * 3.28084

    if "wet_cat_m2" in gdf.columns:
        gdf["wet_cat_ac"] = gdf["wet_cat_m2"] / 4046.8564224
        gdf["ip_ratio"] = gdf["wet_cat_m2"] / gdf["area_m2"]

    # Write shapefile (skip when no wetlands pass the filter)
    output_shapefile = output_dir / "Wetlands_Polygons_with_Stats.shp"
    if not gdf.empty:
        gdf.to_file(output_shapefile)
        logger.info(
            "Generated %d wetland polygons → %s",
            len(gdf), output_shapefile.name,
        )
    else:
        logger.info(
            "No wetland polygons met the filter thresholds; shapefile not written."
        )
    return output_shapefile, gdf
