# System
import os
import sys
import fileinput
from pathlib import Path
import shutil
from glob import glob
import tempfile
import subprocess


# Analysis
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon, MultiPoint, mapping
from shapely.geometry import shape as shapely_shape
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from bs4 import BeautifulSoup
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.windows import Window
# do not use pip to install gdal, use conda instead (conda install -c conda-forge gdal)
from osgeo import gdal


# Visualization
import matplotlib.pyplot as plt
import rasterio.plot


# # Whitebox
# from WBT.whitebox_tools import WhiteboxTools


#======================================CLASSES========================================
class hydrocon_usr_input:
    def string(self, description, default_value=None):
        """
        Prompt the user for a string input with a description.
        If no input is provided, return the default value if specified.
        """
        prompt = f"Enter {description} (default: {default_value}): "
        user_input = input(prompt).strip()
        if user_input == "" and default_value is not None:
            print(f"No {description} provided. Default applied ('{default_value}').")
            return default_value
        elif user_input == "":
            while user_input == "":
                user_input = input(f"A value is required. " + prompt).strip()
        return user_input
    
    def file(self, description, default_value=None):
        """
        Prompt the user for a file name input with a description.
        If no input is provided, return the default value if specified.
        """
        prompt = f"Enter {description} filename (e.g., '{default_value}'). " + \
                "If entering multiple raster files, separate them with commas (e.g., raster1,raster2,raster3) "
        if description == "LiDAR DEM raster":
            filename = [f.strip() for f in input(prompt).split(",")]
        else:
            filename = input(prompt).strip()
            if "." in filename:
                filename = input("Please do not include file extension in the name. " + prompt).strip()
            elif default_value is None and filename == "":
                while filename == "":
                    filename = input(f"A file name is required. " + prompt).strip()
            elif filename == "":
                filename = default_value
                print(f"No {description} name provided. Default applied ('{default_value}').")
        return filename
    

#=======================================FUNCTIONS========================================
def read_shapefile(filename: str, directory: str):
    shapefile = gpd.read_file(directory + \
                              filename + \
                              ".shp")
    return shapefile


def set_directory_structure():
    """
    Creates and organizes the directory structure for hydro conditioning.
    Moves files to the appropriate directories.
    """
    # Ask user to input watershed name (default is "Manning")
    WATERSHED_NAME = hydrocon_usr_input().string("watershed", "Manning")

    # Path for the parent directory of the user's current script
    CURRENT_PATH = str(Path.cwd())

    # Path for base raw data folder
    BS_DATA_PATH = CURRENT_PATH + r"/Data/"
    
    # Specify path for watershed folder
    WATERSHED_PATH = BS_DATA_PATH + WATERSHED_NAME

    # Path for hydro-conditioning folder
    HYDROCON_PATH = WATERSHED_PATH + r"/HydroConditioning"
    
    # Create watershed folder in specified path
    os.makedirs(HYDROCON_PATH, 
                exist_ok=True) # Do nothing if already exists

    # Specify paths for hydro-conditioning subfolders
    HYDROCON_RAW_PATH = HYDROCON_PATH + r"/Raw/"
    HYDROCON_INTERIM_PATH = HYDROCON_PATH + r"/Interim/"
    HYDROCON_PROCESSED_PATH = HYDROCON_PATH + r"/Processed/"
    
    # Create watershed subfolders in specified paths
    SUBFOLDERS_LIST = [HYDROCON_RAW_PATH,
                       HYDROCON_INTERIM_PATH,
                       HYDROCON_PROCESSED_PATH]
    for sub in SUBFOLDERS_LIST:
        os.makedirs(sub, exist_ok=True)

    # Specify source and destination folders before moving files
    src = Path(BS_DATA_PATH)
    dst = Path(HYDROCON_RAW_PATH)
    
    # Move files in base data folder to watershed folder
    for file in src.iterdir():
         if file.is_file():
            destination_file = dst / file.name
            if destination_file.exists():
                print(f"Skipped (already exists): {file.name}")
                continue
            shutil.move(str(file), destination_file)

    # Relevant variables that the user will want in the main script
    dict = {"WATERSHED_NAME": WATERSHED_NAME,
            "BS_DATA_PATH": BS_DATA_PATH,
            "WATERSHED_PATH": WATERSHED_PATH,
            "HYDROCON_PATH": HYDROCON_PATH,
            "HYDROCON_RAW_PATH": HYDROCON_RAW_PATH,
            "HYDROCON_INTERIM_PATH": HYDROCON_INTERIM_PATH,
            "HYDROCON_PROCESSED_PATH": HYDROCON_PROCESSED_PATH}

    return dict


def merge_rasters(lidar_files: str, directory_dict: dict)
    # Turn string input into list
    LIDAR_FILENAMES_LIST = [f for f in lidar_files]

    # Clip each raster to the subbasins shapefile for memory efficiency
    for file in LIDAR_FILENAMES_LIST:
        # 1. Project shapefile to match raster CRS

        # Open raster file and read its CRS
        with rasterio.open(dict["HYDROCON_RAW_PATH"] + \
                        file + \
                        ".tif") as src:
            input_DEM_crs = src.crs
        
        # Project subbasins data to match raster CRS
        clrh_gdf_projected = clrh_gdf.to_crs(input_DEM_crs)

        # Check if shapefile projection aligns with DEM projection
        is_correctly_projected_clrh = (input_DEM_crs == clrh_gdf_projected.crs)

        
        # 2. Clip raster to watershed boundary

        # Convert projected subbasins data to GeoJSON-like format
        shapes = [mapping(geom) for geom in clrh_gdf_projected.geometry]

        # Mask (clip) the input DEM file
        with rasterio.open(dict["HYDROCON_RAW_PATH"] + \
                            file + \
                            ".tif") as src:
            nodata_value = src.nodata
            out_image, out_transform = mask(src, 
                                            shapes, 
                                            crop=True)
            out_meta = src.meta.copy()

        # Update the copied metadata to match the clipped raster's dimensions and transform
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw",              # apply LZW compression (for smaller file size)
            "tiled": True,                  # enable tiling (for faster access)
            "blockxsize": 256,              # set block size for tiling (for faster access)
            "blockysize": 256,              # set block size for tiling (for faster access)
            "nodata": nodata_value          # preserve nodata values
        })

        # Clipped DEM file name with path
        LIDAR_CLIPPED_FILE = dict["HYDROCON_INTERIM_PATH"] + \
                                file + \
                                "_clip"

        # Write clipped DEM data into file
        with rasterio.open(LIDAR_CLIPPED_FILE + ".tif",
                        "w", 
                        **out_meta) as dest:
            dest.write(out_image)

                  
