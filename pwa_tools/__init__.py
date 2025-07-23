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


def set_directory_structure():
    """
    Creates and organizes the directory structure for hydro conditioning.
    Moves files to the appropriate directories.
    """
    # Ask user to input watershed name
    WATERSHED_NAME = hydrocon_usr_input().string("watershed", "Manning")

    # Path for the parent directory of the user's current script
    CURRENT_PATH = str(Path.cwd())
    print("CURRENT_PATH: ", CURRENT_PATH)

    # Path for base raw data folder
    BS_DATA_PATH = CURRENT_PATH + r"/Data/"
    print("BS_DATA_PATH: ", BS_DATA_PATH)
    
    # Specify path for watershed folder
    WATERSHED_PATH = BS_DATA_PATH + WATERSHED_NAME
    print("WATERSHED_PATH: ", WATERSHED_PATH)

    # Path for hydro-conditioning folder
    HYDROCON_PATH = WATERSHED_PATH + r"/HydroConditioning"
    print("HYDROCON_PATH: ", HYDROCON_PATH)
    
    # Create watershed folder in specified path
    os.makedirs(HYDROCON_PATH, 
                exist_ok=True) # Do nothing if already exists

    # Specify paths for hydro-conditioning subfolders
    DD_RAW_PATH = HYDROCON_PATH + r"/Raw/"
    DD_INTERIM_PATH = HYDROCON_PATH + r"/Interim/"
    DD_PROCESSED_PATH = HYDROCON_PATH + r"/Processed/"
    
    # Create watershed subfolders in specified paths
    SUBFOLDERS_LIST = [DD_RAW_PATH,
                       DD_INTERIM_PATH,
                       DD_PROCESSED_PATH]
    for sub in SUBFOLDERS_LIST:
        os.makedirs(sub, exist_ok=True)

    # Specify source and destination folders before moving files
    src = Path(BS_DATA_PATH)
    dst = Path(DD_RAW_PATH)
    
    # Move files in base data folder to watershed folder
    for file in src.iterdir():
         if file.is_file():
            destination_file = dst / file.name
            if destination_file.exists():
                print(f"Skipped (already exists): {file.name}")
                continue
            shutil.move(str(file), destination_file)
