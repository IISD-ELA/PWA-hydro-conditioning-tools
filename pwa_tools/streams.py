"""Stream network processing — clip, extend, and combine NHN + culvert lines.

Replaces ``extend_line`` (lines 383-412), ``clip_nhn_to_watershed``
(lines 459-536), and ``append_culvert_lines`` (lines 538-584) from
the god file.

Key fixes:
  - ``os.chdir`` without try/finally → replaced by ``wbt_session`` context manager
  - WBT ``clip`` return value unchecked → now uses ``check_wbt``
  - ``state.log`` dead writes removed
  - ``read_shapefile`` call uses the new ``io.shapefile`` module
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from pwa_tools._wbt import check_wbt, wbt_session
from pwa_tools.io.shapefile import read_shapefile

logger = logging.getLogger(__name__)


def extend_line(line_geom, extension_distance: float):
    """Extend a LineString by *extension_distance* on both ends.

    Non-LineString geometries and single-point lines are returned unchanged.
    This is a pure function — no I/O, no side effects.
    """
    if line_geom.geom_type != "LineString":
        return line_geom

    coords = list(line_geom.coords)
    if len(coords) < 2:
        return line_geom

    start = np.array(coords[0])
    second = np.array(coords[1])
    start_dir = start - second
    start_dir /= np.linalg.norm(start_dir)
    new_start = start + start_dir * extension_distance

    end = np.array(coords[-1])
    second_last = np.array(coords[-2])
    end_dir = end - second_last
    end_dir /= np.linalg.norm(end_dir)
    new_end = end + end_dir * extension_distance

    return LineString([tuple(new_start)] + coords + [tuple(new_end)])


def append_culvert_lines(
    channels_gdf: gpd.GeoDataFrame,
    culvert_gdf: gpd.GeoDataFrame,
    output_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Append culvert lines to the NHN channels GeoDataFrame.

    Culverts are reprojected to match channels if needed, then extended
    by 2 m on each end (projected CRS only). Returns the combined GDF.
    If *output_path* is given, also writes the result to shapefile.
    """
    if culvert_gdf.crs != channels_gdf.crs:
        culvert_gdf = culvert_gdf.to_crs(channels_gdf.crs)

    if culvert_gdf.crs.is_projected:
        culvert_gdf = culvert_gdf.copy()
        culvert_gdf["geometry"] = culvert_gdf["geometry"].apply(
            lambda geom: extend_line(geom, 2.0)
        )
    else:
        logger.warning("Culvert CRS is not projected — skipping 2m line extension")

    burn_lines_gdf = pd.concat(
        [channels_gdf[["geometry"]], culvert_gdf[["geometry"]]],
        ignore_index=True,
    )
    burn_lines_gdf["source"] = (
        ["NHN"] * len(channels_gdf) + ["Culvert"] * len(culvert_gdf)
    )

    if output_path is not None:
        burn_lines_gdf.to_file(output_path)
        logger.info(
            "Wrote burn lines (%d NHN + %d culvert) → %s",
            len(channels_gdf), len(culvert_gdf), Path(output_path).name,
        )

    return burn_lines_gdf


def clip_nhn_to_watershed(
    nhn_filename: str,
    clrh_proj_nhn_path: Path,
    input_dem_crs,
    input_dem_crs_alnum: str,
    raw_dir: Path,
    interim_dir: Path,
    culvert_filename: str | None = None,
    culvert_target_crs: str | None = None,
) -> Path:
    """Clip NHN streams to the watershed, optionally append culverts, reproject to DEM CRS.

    Uses the ``wbt_session`` context manager for safe WhiteboxTools calls.

    Parameters
    ----------
    nhn_filename
        Basename (no extension) of the NHN shapefile in *raw_dir*.
    clrh_proj_nhn_path
        Path to the CLRH subbasins shapefile already projected to NHN CRS
        (used as the clip polygon). Include ``.shp`` extension.
    input_dem_crs
        CRS of the target DEM (pyproj CRS object or EPSG string).
    input_dem_crs_alnum
        Short alphanumeric CRS tag for filename suffixing.
    raw_dir, interim_dir
        Directories for input and intermediate files.
    culvert_filename
        If provided, basename of the culvert shapefile in *raw_dir*.
    culvert_target_crs
        CRS to reproject culvert shapefile to before appending (typically
        the project CRS from PwaConfig).

    Returns
    -------
    Path
        Path to the clipped + projected burn-lines shapefile (no extension
        in the stem — ``.shp`` is the file on disk).
    """
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    clrh_proj_nhn_path = Path(clrh_proj_nhn_path)

    nhn_input = raw_dir / f"{nhn_filename}.shp"
    nhn_clipped = raw_dir / f"{nhn_filename}.shp"  # WBT overwrites in-place

    with wbt_session() as wbt:
        check_wbt(
            wbt.clip(
                i=str(nhn_input),
                clip=str(clrh_proj_nhn_path),
                output=str(nhn_clipped),
            ),
            "clip",
        )

    nhn_gdf_clip = gpd.read_file(nhn_clipped)

    # Optionally append culvert lines
    if culvert_filename:
        culvert_path = raw_dir / f"{culvert_filename}.shp"
        culvert_gdf = read_shapefile(culvert_path, target_crs=culvert_target_crs)
        burn_lines_path = interim_dir / "burn_lines.shp"
        burn_lines_gdf = append_culvert_lines(
            nhn_gdf_clip, culvert_gdf, output_path=burn_lines_path,
        )
    else:
        burn_lines_gdf = nhn_gdf_clip

    # Project to DEM CRS
    burn_lines_projected = burn_lines_gdf.to_crs(input_dem_crs)

    output_path = interim_dir / f"{nhn_filename}_clip_projected_{input_dem_crs_alnum}.shp"
    burn_lines_projected.to_file(output_path)

    logger.info("Clipped + projected NHN → %s", output_path.name)
    return output_path
