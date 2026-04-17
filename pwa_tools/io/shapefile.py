"""Shapefile I/O — read and optionally reproject shapefiles.

Replaces ``read_shapefile()`` from the god file (``__init__.py:247-262``).

Key changes from the original:
  - Accepts a ``pathlib.Path`` filepath instead of ``directory + filename + '.shp'``
    string concatenation.
  - Accepts ``target_crs`` as an explicit parameter instead of reading the global
    ``state.crs_string``. Callers pass ``config.inputs.crs_string`` from
    :class:`~pwa_tools.config.PwaConfig`.
  - Fixes BUG-001: the original crashes with ``UnboundLocalError`` when the
    shapefile's CRS already matches the project CRS — neither ``if`` nor ``elif``
    branch fires, leaving ``shape_out`` unassigned. The new code returns the
    GeoDataFrame as-is when no reprojection is needed.
  - Removes the dead ``state.log`` write (assigned 4 times, read 0 times).
  - Validates that the file exists before attempting to read.
  - Validates that the shapefile has a CRS before attempting to reproject.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def read_shapefile(
    filepath: Path,
    target_crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Read a shapefile, optionally reprojecting to *target_crs*.

    Parameters
    ----------
    filepath
        Full path to the ``.shp`` file (or any format geopandas can read).
        Strings are coerced to ``pathlib.Path``.
    target_crs
        If provided, the GeoDataFrame is reprojected when the file's native
        CRS differs. If ``None``, the GeoDataFrame is returned with its
        original CRS.

    Returns
    -------
    gpd.GeoDataFrame

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist on disk.
    ValueError
        If *target_crs* is specified but the shapefile has no CRS defined.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Shapefile not found: {filepath}")

    gdf = gpd.read_file(filepath)

    if target_crs is not None:
        if gdf.crs is None:
            raise ValueError(
                f"Cannot reproject {filepath.name}: shapefile has no CRS defined. "
                f"Set the CRS first with gdf.set_crs(...)."
            )
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

    return gdf
