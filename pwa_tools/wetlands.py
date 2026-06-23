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
import rasterio
import rasterio.features
from scipy.ndimage import label
from shapely.geometry import shape as shapely_shape

logger = logging.getLogger(__name__)

# Thresholds for filtering insignificant wetlands
_DEPTH_THRESHOLD_M = 0.1   # ignore depressions shallower than 10 cm
_AREA_THRESHOLD_M2 = 4000    # minimum wetland area
_VOLUME_THRESHOLD_M3 = 30   # minimum wetland storage volume


def gen_wetland_polygons(
    depressions_raster_path: Path,
    output_dir: Path,
) -> tuple[Path, gpd.GeoDataFrame]:
    """Generate wetland polygons with area/volume/depth statistics from a depression raster.

    Steps:
      1. Threshold the depression raster at ``_DEPTH_THRESHOLD_M``
         (currently 0.1 m = 10 cm; ignores shallower depressions)
      2. Label connected components (8-connectivity)
      3. Compute per-wetland area, total storage volume, and median depth
      4. Filter by area >= ``_AREA_THRESHOLD_M2`` (currently 4000 m²)
         and volume >= ``_VOLUME_THRESHOLD_M3`` (currently 30 m³)
      5. Vectorize remaining polygons
      6. Write stats CSV + shapefile

    Threshold constants live at module level so the live values are
    the source of truth; the step descriptions above reference the
    constants by name (not by literal value) to avoid the docstring
    silently drifting from the implementation when a default
    changes. See ``test_wetlands_docstring_cites_live_constants``.

    Returns (shapefile_path, wetlands_gdf).
    """
    depressions_raster_path = Path(depressions_raster_path)
    output_dir = Path(output_dir)

    with rasterio.open(depressions_raster_path) as src:
        depression_data = src.read(1)
        transform = src.transform
        pixel_area = abs(transform[0] * transform[4])
        nodata = src.nodata
        crs = src.crs

    # Mask nodata and select depressions above threshold
    valid_mask = (depression_data > _DEPTH_THRESHOLD_M) & (depression_data != nodata)

    # Label connected depressions (wetlands)
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(valid_mask, structure=structure)

    flat_labels = labeled_array.ravel()
    flat_depths = depression_data.ravel()

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

    # Build statistics dataframe
    stats_df = pd.DataFrame({
        "wetland_id": np.arange(len(areas_m2)),
        "area_m2": areas_m2,
        "volume_m3": volumes_m3,
        "median_depth_m": medians,
    })

    # Filter out background + small features
    stats_df = stats_df.query(
        f"wetland_id != 0 and area_m2 >= {_AREA_THRESHOLD_M2} "
        f"and volume_m3 >= {_VOLUME_THRESHOLD_M3}"
    )

    # Write stats CSV
    stats_csv = output_dir / "Wetlands_Stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    # Vectorize remaining wetland polygons
    valid_ids = set(stats_df["wetland_id"])
    valid_mask_vec = np.isin(labeled_array, list(valid_ids))

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

    # Write shapefile
    output_shapefile = output_dir / "Wetlands_Polygons_with_Stats.shp"
    gdf.to_file(output_shapefile)

    logger.info(
        "Generated %d wetland polygons → %s",
        len(gdf), output_shapefile.name,
    )
    return output_shapefile, gdf
