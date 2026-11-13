"""Wetland polygon and catchment-geometry generation from depression rasters.

Replaces ``gen_wetland_polygons`` (lines 741-839) from the god file.

Uses WhiteboxTools (via ``pwa_tools._wbt``) for D8/flow-accumulation/
catchment delineation, and ``pwa_tools.io.shapefile`` for vector I/O;
otherwise pure rasterio + numpy + scipy analysis.

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
from pwa_tools.io.shapefile import read_shapefile
import rasterio
import rasterio.features
import rasterio.transform
from scipy.ndimage import label, maximum as label_maximum
from shapely.geometry import shape as shapely_shape

logger = logging.getLogger(__name__)

# Thresholds for filtering insignificant wetlands
_DEPTH_THRESHOLD_M = 0.1   # ignore depressions shallower than 10 cm
_AREA_THRESHOLD_M2 = 4000    # minimum wetland area
_VOLUME_THRESHOLD_M3 = 30   # minimum wetland storage volume

# Maximum distance (map units, typically metres) WBT snap_pour_points may
# move a user-marked point to reach the nearest high-flow-accumulation cell.
_SNAP_DIST_M = 20.0



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


def gen_wetland_catchment_polygons(
    points_path: Path,
    wetlands_path: Path,
    direction_raster_path: Path,
    accum_raster_path: Path,
    output_dir: Path,
    *,
    id_field: str = "wetland_id",
    snap_dist: float = _SNAP_DIST_M,
) -> tuple[Path, gpd.GeoDataFrame]:
    """Delineate full contributing-catchment polygons for user-selected wetlands.

    Computing a catchment polygon for every wetland in a watershed is
    expensive at the scale of hundreds or thousands of wetlands.
    ``gen_wetland_polygons`` already provides a cheap per-wetland catchment
    *area* estimate (``wet_cat_m2``) from the max flow-accumulation value.
    This function instead delineates real catchment *geometry*, but only for
    wetlands a user has flagged by placing a point — typically after
    reviewing the wetland polygons shapefile in a GIS.

    Points need no attributes: each point's wetland identity is borrowed
    from the nearest polygon in *wetlands_path*, as long as that polygon is
    within *snap_dist* (wetlands produced by :func:`gen_wetland_polygons`
    never overlap, so within that radius "nearest" is always an unambiguous
    match). A point with no wetland polygon within *snap_dist* is assigned a
    new negative ``id_field`` value instead (``-1``, ``-2``, ...) rather than
    silently inheriting a distant wetland's identity — negative ids are
    never produced by :func:`gen_wetland_polygons`, so they're unambiguous
    when reviewing the output in a GIS.

    Steps:
      1. Read the points and wetland polygons, reprojecting both to the
         D8 raster's CRS.
      2. Assign each point the ``id_field`` value of its nearest wetland
         polygon, within *snap_dist*; points with no such polygon get a new
         negative id.
      3. Snap each point onto the nearest high-flow-accumulation cell with
         WBT ``snap_pour_points`` (forgives imprecise manual placement).
      4. Delineate the full upstream contributing area for every snapped
         point in one WBT ``watershed`` call.
      5. Vectorize the watershed raster and label each polygon by sampling
         the raster at its originating point's (snapped) location —
         independent of however WhiteboxTools numbers pour points
         internally.

    Parameters
    ----------
    points_path
        User-marked wetland-of-interest points (``.shp`` or any
        geopandas-readable format). No attributes required.
    wetlands_path
        Wetland polygons shapefile with an ``id_field`` column, e.g. the
        output of :func:`gen_wetland_polygons`.
    direction_raster_path
        D8 pointer raster, e.g. from :func:`gen_direction_raster`.
    accum_raster_path
        Flow-accumulation raster on the same grid as *direction_raster_path*,
        e.g. from :func:`gen_accum_raster`.
    output_dir
        Directory for all outputs (intermediate pour points, the watershed
        raster, and the final catchment-polygons shapefile).
    id_field
        Column in *wetlands_path* identifying each wetland. Carried through
        to the output catchment polygons.
    snap_dist
        Maximum distance (map units) a point may be from the nearest wetland
        polygon to borrow its ``id_field`` (see above), and the maximum
        distance WBT ``snap_pour_points`` may move a point to reach the
        nearest high-flow-accumulation cell. Defaults to the module-level
        constant ``_SNAP_DIST_M``.

    Returns
    -------
    tuple[Path, gpd.GeoDataFrame]
        Path to the written catchment-polygons shapefile, and the
        corresponding GeoDataFrame (``id_field``, ``cat_ar_m2``,
        ``cat_ar_ac``, geometry). Points whose snapped location produces
        no catchment (nodata or outside the raster extent) are logged and
        excluded, not fatal.
    """
    direction_raster_path = Path(direction_raster_path)
    accum_raster_path = Path(accum_raster_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(direction_raster_path) as d8_src:
        raster_crs = d8_src.crs
        d8_shape = d8_src.shape

    with rasterio.open(accum_raster_path) as accum_src:
        accum_shape = accum_src.shape

    if d8_shape != accum_shape:
        raise ValueError(
            "direction_raster_path and accum_raster_path must share the same "
            f"grid (got {d8_shape}, expected {accum_shape})"
        )

    raster_crs_wkt = raster_crs.to_wkt()
    points_gdf = read_shapefile(points_path, target_crs=raster_crs_wkt)
    wetlands_gdf = read_shapefile(wetlands_path, target_crs=raster_crs_wkt)

    if id_field not in wetlands_gdf.columns:
        raise ValueError(
            f"wetlands_path is missing the '{id_field}' column needed to "
            "label catchment polygons."
        )
    if points_gdf.empty:
        raise ValueError("points_path contains no points to delineate catchments for.")
    if wetlands_gdf.empty:
        raise ValueError("wetlands_path contains no wetland polygons to match points against.")

    # Borrow each point's wetland identity from the nearest wetland polygon,
    # but only within snap_dist. Wetlands produced by gen_wetland_polygons
    # are spatially distinct (non-overlapping), so within that radius
    # "nearest" is always an unambiguous match.
    points_gdf = gpd.sjoin_nearest(
        points_gdf[["geometry"]],
        wetlands_gdf[[id_field, "geometry"]],
        how="left",
        max_distance=snap_dist,
    )
    points_gdf = points_gdf[~points_gdf.index.duplicated(keep="first")]
    points_gdf = points_gdf[["geometry", id_field]].reset_index(drop=True)

    # Points with no wetland polygon within snap_dist get a new negative id
    # instead of silently inheriting a distant wetland's identity.
    unmatched = points_gdf[id_field].isna()
    if unmatched.any():
        new_ids = -np.arange(1, int(unmatched.sum()) + 1)
        points_gdf.loc[unmatched, id_field] = new_ids
        logger.warning(
            "%d point(s) had no wetland polygon within %.1f m — assigned "
            "new catchment id(s): %s.",
            int(unmatched.sum()), snap_dist, sorted(new_ids.tolist()),
        )
    points_gdf[id_field] = points_gdf[id_field].astype(wetlands_gdf[id_field].dtype)

    raw_points_path = output_dir / "_wetland_catchment_pour_points.shp"
    points_gdf.to_file(raw_points_path)

    snapped_points_path = output_dir / "_wetland_catchment_pour_points_snapped.shp"
    watershed_raster_path = output_dir / "_wetland_catchment_basins.tif"

    with wbt_session() as wbt:
        check_wbt(
            wbt.snap_pour_points(
                pour_pts=str(raw_points_path),
                flow_accum=str(accum_raster_path),
                output=str(snapped_points_path),
                snap_dist=snap_dist,
            ),
            "snap_pour_points",
        )
        check_wbt(
            wbt.watershed(
                d8_pntr=str(direction_raster_path),
                pour_pts=str(snapped_points_path),
                output=str(watershed_raster_path),
            ),
            "watershed",
        )

    snapped_gdf = gpd.read_file(snapped_points_path)

    with rasterio.open(watershed_raster_path) as ws_src:
        watershed_data = ws_src.read(1)
        ws_transform = ws_src.transform
        ws_nodata = ws_src.nodata
        ws_crs = ws_src.crs

    # Map each raster "basin" value back to the id_field of the point it was
    # delineated from, by sampling the raster at that point's (snapped)
    # location. Robust to however WhiteboxTools numbers pour points internally.
    value_to_id: dict[int, object] = {}
    for wid, geom in zip(points_gdf[id_field], snapped_gdf.geometry):
        row, col = rasterio.transform.rowcol(ws_transform, geom.x, geom.y)
        if not (0 <= row < watershed_data.shape[0] and 0 <= col < watershed_data.shape[1]):
            logger.warning(
                "Snapped pour point for wetland %s fell outside the watershed "
                "raster — skipped.", wid,
            )
            continue
        value = watershed_data[row, col]
        if ws_nodata is not None and value == ws_nodata:
            logger.warning(
                "Snapped pour point for wetland %s produced no catchment "
                "(nodata) — skipped.", wid,
            )
            continue
        value_to_id[int(value)] = wid

    valid_values = set(value_to_id)
    mask = (
        np.isin(watershed_data, list(valid_values))
        if valid_values
        else np.zeros_like(watershed_data, dtype=bool)
    )

    polygons = []
    ids = []
    for geom, val in rasterio.features.shapes(
        watershed_data.astype(np.int32), mask=mask, transform=ws_transform,
    ):
        if val in valid_values:
            polygons.append(shapely_shape(geom))
            ids.append(value_to_id[val])

    gdf = gpd.GeoDataFrame({id_field: ids, "geometry": polygons}, crs=ws_crs)
    if not gdf.empty:
        gdf = gdf.dissolve(by=id_field, aggfunc="first").reset_index()
        # Field names kept <= 10 chars for ESRI Shapefile/ArcGIS compatibility
        # ("catchment_m2"/"catchment_ac" both truncate to the indistinguishable
        # "catchment_" in ArcGIS).
        gdf["cat_ar_m2"] = gdf.geometry.area
        gdf["cat_ar_ac"] = gdf["cat_ar_m2"] / 4046.8564224

    output_path = output_dir / "Wetland_Catchment_Polygons.shp"
    if not gdf.empty:
        gdf.to_file(output_path)
        logger.info(
            "Generated %d wetland catchment polygon(s) → %s",
            len(gdf), output_path.name,
        )
    else:
        logger.info(
            "No wetland catchments could be delineated; shapefile not written."
        )
    return output_path, gdf
