# Sytem
import pickle
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
from scipy.ndimage import distance_transform_edt, label
import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.mask import mask
from bs4 import BeautifulSoup
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.windows import Window
import collections
from collections import defaultdict
# do not use pip to install gdal, use conda instead (conda install -c conda-forge gdal)
from osgeo import gdal


# Visualization
import matplotlib.pyplot as plt
import rasterio.plot


# Whitebox
from .WBT.whitebox_tools import WhiteboxTools  


#======================================SETUP========================================

#-------------------------------USER INPUT CLASS AND FUNCTIONS--------------------------------
def snake_case(string):
    """
    Converts a given string to snake_case format.
    """
    # Replace spaces and hyphens with underscores, and convert to lowercase
    return string.replace(" ", "_").replace("-", "_").lower()

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
        out_str = snake_case(user_input)
        return out_str
    
    def file(self, description, default_value=None):
        """
        Prompt the user for a file name input with a description.
        If no input is provided, return the default value if specified.
        """
        prompt = f"Enter {description} filename (e.g., '{default_value}'). " + \
                "If entering multiple raster files, separate them with commas (e.g., raster1,raster2,raster3) "
        filename = input(prompt)
        if description == "LiDAR DEM raster" and "," in filename:
            filename = [f.strip() for f in filename.split(",")]
            if default_value is None and filename == "":
                while filename == "":
                    filename = input(f"A file name is required. " + prompt).strip()
            elif filename == "":
                filename = default_value
                print(f"No {description} name provided. Default applied ('{default_value}').")
        else:
            filename = filename.strip()
            if "." in filename:
                filename = input("Please do not include file extension in the name. " + prompt).strip()
            elif default_value is None and filename == "":
                while filename == "":
                    filename = input(f"A file name is required. " + prompt).strip()
            elif filename == "":
                filename = default_value
                print(f"No {description} name provided. Default applied ('{default_value}').")
        return filename


#---------------------------------PROJECT SETUP CLASS AND FUNCTIONS--------------------------------
class project_state:
    def __init__(self):
        self.WATERSHED_NAME = None
        self.BS_DATA_PATH = None
        self.WATERSHED_PATH = None
        self.HYDROCON_PATH = None
        self.HYDROCON_RAW_PATH = None
        self.HYDROCON_INTERIM_PATH = None
        self.HYDROCON_PROCESSED_PATH = None
        self.CLRH_FILENAME = None
        self.LIDAR_FILENAME = None
        self.NHN_FILENAME = None
        self.MULTIPLE_LIDAR_RASTERS = None
        self.RECOVERY_PATH = None
        self.LAST_FUNCTION_RUN = None

# Global Project State Instance
state = project_state()

def save_state():
    # Store state as pickle file for use in main script
    if state.RECOVERY_PATH is not None:
        pickle_file_path = Path(state.RECOVERY_PATH) / "state.pkl"
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved to: {state.RECOVERY_PATH}")
    else:
        print("Recovery mode not enabled. Directory information not saved to pickle file.")

    print(f"Project setup complete. Data organized in directory: {state.WATERSHED_PATH}")

def set_directory_structure(default_watershed = "watershed_name", data_dir = None, recovery_mode = False):
    """
    Creates and organizes the directory structure for hydro conditioning.
    Moves files to the appropriate directories.

-------    Parameters:
    data_dir (str, optional): The relative path to the base raw data folder. If None, it defaults to the current working directory + "/Data/".
    recovery_mode (bool, optional): If True, functions in this project will store outputs as pkl files in the interim data folder, and this function  
        will load the directory dictionary from a pkl file in the interim data folder instead of creating a new directory structure.  
        This is for testing purposes to ensure that the same directory structure is used across different sessions.
    """
    
    # Ask user to input watershed name (default is "Manning")
    state.WATERSHED_NAME = hydrocon_usr_input().string("Provide the name of your watershed", default_watershed)

    # Path for the parent directory of the user's current script
    CURRENT_PATH = str(Path.cwd())

    # Path for base raw data folder
    if data_dir is not None:
        state.BS_DATA_PATH = CURRENT_PATH + data_dir
    else:
        state.BS_DATA_PATH = CURRENT_PATH + r"/Data/"
    
    # Specify path for watershed folder
    state.WATERSHED_PATH = state.BS_DATA_PATH + state.WATERSHED_NAME

    # Path for hydro-conditioning folder
    state.HYDROCON_PATH = state.WATERSHED_PATH + r"/HydroConditioning"
    
    # Create watershed folder in specified path
    os.makedirs(state.HYDROCON_PATH, 
                exist_ok=True) # Do nothing if already exists

    # Specify paths for hydro-conditioning subfolders
    state.HYDROCON_RAW_PATH = state.HYDROCON_PATH + r"/Raw/"
    state.HYDROCON_INTERIM_PATH = state.HYDROCON_PATH + r"/Interim/"
    state.HYDROCON_PROCESSED_PATH = state.HYDROCON_PATH + r"/Processed/"
    
    # Create watershed subfolders in specified paths
    SUBFOLDERS_LIST = [state.HYDROCON_RAW_PATH,
                       state.HYDROCON_INTERIM_PATH,
                       state.HYDROCON_PROCESSED_PATH]
    for sub in SUBFOLDERS_LIST:
        os.makedirs(sub, exist_ok=True)

    # Specify source and destination folders before moving files
    src = Path(state.BS_DATA_PATH)
    dst = Path(state.HYDROCON_RAW_PATH)
    
    # Move files in base data folder to watershed folder
    for file in src.iterdir():
         if file.is_file():
            destination_file = dst / file.name
            if destination_file.exists():
                print(f"Skipped (already exists): {file.name}")
                continue
            shutil.move(str(file), destination_file)

        
    if recovery_mode:
        state.RECOVERY_PATH = state.HYDROCON_INTERIM_PATH

    # Ending this and other functions with save_state() to store directory information in a pickle file for use in the main script
    state.LAST_FUNCTION_RUN = "set_directory_structure"
    save_state()

    return state.__dict__


def project_setup(watershed_default = "watershed_name", delineation_default = "finalcat_info_v1-0", lidar_default = "sr_dem_cgvd28", channels_default = "NHN_05OE000_5_0_HD_SLWATER_1", data_dir = None, recovery_mode = False):
    
    
    # Organize data folders and files and store relevant directory information in a dictionary
    # (user will be prompted to enter the watershed name to name the working directory after)
    set_directory_structure(default_watershed = watershed_default, data_dir = data_dir, recovery_mode = recovery_mode)


    # Name of watershed shapefile from CLRH hydrofabrics zip file (.shp)
    state.CLRH_FILENAME = hydrocon_usr_input().file("hydrofabric shapefile", 
                                                delineation_default) # This is the default for Manning Canal


    # Name of LiDAR DEM raster from LiDAR DEM zip file (.tif)
    # This can be multiple files, separated by commas (e.g., Boyne river requires multiple rasters))
    # Multiple rasters will be merged into one in latter steps
    state.LIDAR_FILENAME = hydrocon_usr_input().file("LiDAR DEM raster", 
                                                lidar_default) # This is the default for Manning Canal


    # Name of streams shapefile from NHN streams zip file (.shp)
    state.NHN_FILENAME = hydrocon_usr_input().file("NHN streams shapefile", 
                                                channels_default) # This is the default for Manning Canal


    # Boolean object to indicate if there are multiple LiDAR DEM rasters
    state.MULTIPLE_LIDAR_RASTERS = True if (isinstance(state.LIDAR_FILENAME, list) and len(state.LIDAR_FILENAME)) > 1 else False

    # Ending this and other functions with save_state() to store directory information in a pickle file for use in the main script
    state.LAST_FUNCTION_RUN = "project_setup"
    save_state()

    

#=======================================FUNCTIONS========================================
def read_shapefile(filename: str, directory: str):
    shapefile = gpd.read_file(directory + \
                              filename + \
                              ".shp")
    return shapefile



def resample_lidar_raster(lidar_file, resolution_m):
    print("Starting resample_lidar_raster()...")
    
    resolution_units = "m"
    LIDAR_RESAMPLED_FILE = lidar_file + \
                        f"_resample_{resolution_m}{resolution_units}"

    subprocess.run([
        "gdalwarp",
        "-tr", str(resolution_m), str(resolution_m),
        "-r", "cubic",
        lidar_file + ".tif",
        LIDAR_RESAMPLED_FILE + ".tif"
    ])

    print(f"Inside resample_lidar_raster(): The resampled file has been written to {LIDAR_RESAMPLED_FILE}.")
    print("resample_lidar_raster() has ended.")

    state.LAST_FUNCTION_RUN = "resample_lidar_raster"
    save_state()
    
    return LIDAR_RESAMPLED_FILE


def clip_lidar_to_shapefile(projected_gdf,
                           lidar_filename, lidar_directory):
    print("Starting clip_lidar_to_shapefile()...")
    # Convert projected subbasins data to GeoJSON-like format
    shapes = [mapping(geom) for geom in projected_gdf.geometry]

    # Mask (clip) the input DEM file
    with rasterio.open(lidar_directory + \
                        lidar_filename + \
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
        "transform": out_transform
    })

    # Clipped DEM file name with path
    LIDAR_CLIPPED_FILE = state.HYDROCON_INTERIM_PATH + \
                         lidar_filename + \
                            "_clip"

    # Write clipped DEM data into file
    with rasterio.open(LIDAR_CLIPPED_FILE + ".tif",
                    "w", 
                    **out_meta) as dest:
        dest.write(out_image)
                        
    print(f"Inside clip_lidar_to_shapefile(): the clipped file has been written to {LIDAR_CLIPPED_FILE}.")
    print("clip_lidar_to_shapefile() has ended.")

    state.LAST_FUNCTION_RUN = "clip_lidar_to_shapefile"
    save_state()

    return LIDAR_CLIPPED_FILE


def clip_nhn_to_watershed(nhn_filename, 
                          clrh_proj_nhn_file,
                          input_DEM_crs,
                          input_DEM_crs_alnum):
    
    print("Starting clip_nhn_to_watershed()...")


    # Get path to the folder where THIS file lives (required to access WhiteboxTools)
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Save working directory so we can return to it later
    original_dir = os.getcwd()

    # Initialize whitebox tools object
    wbt = WhiteboxTools()

    # Set whitebox directory
    wbt_dir = os.path.join(this_dir, "WBT")
    wbt.set_whitebox_dir(wbt_dir)

    # Clipped NHN shapefile name with PATH
    state.NHN_CLIPPED_FILE = state.HYDROCON_INTERIM_PATH + \
                        nhn_filename + \
                        "_clip"

    # Clip NHN streams shapefile to watershed
    wbt.clip(
    i=state.HYDROCON_RAW_PATH+nhn_filename+".shp",
    clip=clrh_proj_nhn_file+".shp",
    output=state.NHN_CLIPPED_FILE+".shp"
    )

    # Return to original working directory
    os.chdir(original_dir)

    # Load NHN streams shapefile
    nhn_gdf_clip = gpd.read_file(state.NHN_CLIPPED_FILE + \
                                ".shp")


    # Project clipped NHN shapefile to match DEM
    nhn_gdf_clip_projected_lidar = nhn_gdf_clip.to_crs(input_DEM_crs)

    # Clipped and projected NHN shapefile name with path
    state.NHN_CLIPPED_PROJECTED_LIDAR_FILE = state.HYDROCON_INTERIM_PATH + \
                                        nhn_filename + \
                                        f"_clip_projected_{input_DEM_crs_alnum}"

    # Write clipped and projected NHN shapefile
    nhn_gdf_clip_projected_lidar.to_file(state.NHN_CLIPPED_PROJECTED_LIDAR_FILE + \
                                        ".shp")

    # Check if shapefile projection aligns with DEM projection
    is_correctly_projected_nhn_lidar = (input_DEM_crs == nhn_gdf_clip_projected_lidar.crs)

    # Print results
    print("Inside clip_nhn_to_watershed(): NHN shapefile projection is aligned with DEM projection: ", 
        is_correctly_projected_nhn_lidar)
    print("clip_nhn_to_watershed() has ended.")
    print("clip_nhn_to_watershed has ended.")

    state.LAST_FUNCTION_RUN = "clip_nhn_to_watershed"
    save_state()

    return state.NHN_CLIPPED_PROJECTED_LIDAR_FILE


def gen_depressions_raster(lidar_filename,
                            lidar_clipped_resampled_file,
                            nhn_clipped_projected_lidar_file,
                            resolution_m):
    print("Starting gen_depressions_raster()...")

    # Get path to the folder where THIS file lives (required to access WhiteboxTools)
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Save working directory so we can return to it later
    original_dir = os.getcwd()

    # Initialize whitebox tools object
    wbt = WhiteboxTools()

    # Set whitebox directory
    wbt_dir = os.path.join(this_dir, "WBT")
    wbt.set_whitebox_dir(wbt_dir)

    # Burn streams into DEM and fill
    wbt.fill_burn(
        dem=lidar_clipped_resampled_file+".tif",
        streams=nhn_clipped_projected_lidar_file+".shp",
        output=lidar_clipped_resampled_file+"_FillBurn"+".tif"
    )


    # Subtract raw DEM from filled DEM
    first_statement_operand = lidar_clipped_resampled_file+"_FillBurn"+".tif"
    second_statement_operand = lidar_clipped_resampled_file+".tif"
    wbt.raster_calculator(
        output = lidar_clipped_resampled_file+"_FillBurn_Deps"+".tif",
        statement = f"'{first_statement_operand}' - '{second_statement_operand}'"  
    )


    # Depressions raster file with path
    state.DEPRESSIONS_RASTER_FILE = state.HYDROCON_PROCESSED_PATH + lidar_filename + \
                            f"_clip_resample_{resolution_m}m_FillBurn_Deps_Corr"


    # Remove stray burn lines
    wbt.conditional_evaluation(
        i=lidar_clipped_resampled_file+"_FillBurn_Deps"+".tif",
        output=state.DEPRESSIONS_RASTER_FILE+".tif",
        statement="value < 0.0",
        true=0.0,
        false=lidar_clipped_resampled_file+"_FillBurn_Deps"+".tif"
    )

    # Return to original working directory
    os.chdir(original_dir)

    print("Inside gen_depressions_raster(): The depressions raster has been generated and saved to: ",
          state.DEPRESSIONS_RASTER_FILE + ".tif")
    
    state.LAST_FUNCTION_RUN = "gen_depressions_raster"
    save_state()
    
    return state.DEPRESSIONS_RASTER_FILE


def calc_depression_depths(clrh_proj_lidar_file,
                           watershed_name,
                           depressions_raster_file,
                           clrh_gdf_projected_lidar):
    print("Starting calc_depression_depths()...")

    # CLRH subbasins raster file name with path 
    state.CLRH_RASTER_FILE = clrh_proj_lidar_file + "_raster"

    # Get path to the folder where THIS file lives (required to access WhiteboxTools)
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Save working directory so we can return to it later
    original_dir = os.getcwd()

    # Initialize whitebox tools object
    wbt = WhiteboxTools()

    # Set whitebox directory
    wbt_dir = os.path.join(this_dir, "WBT")
    wbt.set_whitebox_dir(wbt_dir)

    # # Convert clrh subbasin polygons to raster
    wbt.vector_polygons_to_raster(
        i=clrh_proj_lidar_file+".shp",
        output=state.CLRH_RASTER_FILE+".tif",
        field = "FID",
        nodata = True,
        cell_size = 5.0
    )

    # Open clrh subbasins raster file
    with rasterio.open(state.CLRH_RASTER_FILE+".tif") as src:
        target_crs = 26914
        profile = src.profile
        profile.update(crs=CRS.from_epsg(target_crs))
    
    # Zonal stats file name with path
    state.ZONAL_STATS_FILE = Path(state.HYDROCON_PROCESSED_PATH, 
                    f"ZonalStats_{watershed_name}.html").resolve()


    # Make sure that the hydrofabric raster is aligned with the depression raster
    subprocess.run([
        "gdalwarp",
        "-r", "near",               
        "-overwrite",    
        "-tr", "5", "5",            
        "-te",                     
        str(rasterio.open(depressions_raster_file + ".tif").bounds.left),
        str(rasterio.open(depressions_raster_file + ".tif").bounds.bottom),
        str(rasterio.open(depressions_raster_file + ".tif").bounds.right),
        str(rasterio.open(depressions_raster_file + ".tif").bounds.top),
        state.CLRH_RASTER_FILE + ".tif",
        state.CLRH_RASTER_FILE + "_aligned" + ".tif"
    ])


    # Calculate zonal statistics for depression raster
    wbt.zonal_statistics(
        i=depressions_raster_file + ".tif",
        features=state.CLRH_RASTER_FILE + "_aligned" + ".tif",
        stat="total",
        out_table=state.ZONAL_STATS_FILE
    )

    # Read zonal stats html as pandas df
    zonal_stats_df = pd.read_html(state.ZONAL_STATS_FILE,
                                flavor='bs4')[0]

    # Calculate depression depths (mm) and add field to gdf
    clrh_gdf_projected_lidar['Deps_Depth_mm'] = zonal_stats_df['Mean'] * 1000

    # Calculate depression volumes (m3) and add field to gdf
    clrh_gdf_projected_lidar['Deps_Vol_m3'] = clrh_gdf_projected_lidar['Deps_Depth_mm']*clrh_gdf_projected_lidar['BasArea']
    
    # Depression depths file name with path
    state.DEPRESSION_DEPTHS_FILE = state.HYDROCON_PROCESSED_PATH + \
                            "CLRH_basins_depression_depths"

    # Write geodataframe to file for use in raven input file creation
    clrh_gdf_projected_lidar.to_file(state.DEPRESSION_DEPTHS_FILE + ".shp")

    # Return to original working directory
    os.chdir(original_dir)

    state.LAST_FUNCTION_RUN = "calc_depression_depths"
    save_state()

    return state.DEPRESSION_DEPTHS_FILE


def gen_wetland_polygons(depressions_raster_file):
    print("Starting gen_wetland_polygons()...")

    # Load the depression depths raster
    with rasterio.open(depressions_raster_file + ".tif") as src:
        depression_data = src.read(1)  # Read first layer (only has one layer)
        transform = src.transform
        pixel_area = abs(transform[0] * transform[4])
        nodata = src.nodata

    # Mask nodata and select depressions
    depth_threshold = 0.05  # ignore anything below 5 cm (for efficiency later)
    valid_mask = (depression_data > depth_threshold) & (depression_data != nodata)

    # Label all connected depressions > 0 (wetlands)
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(valid_mask, structure=structure)

    # Flatten arrays
    flat_labels = labeled_array.ravel()
    flat_depths = depression_data.ravel()

    # Get pixel counts per label (per wetland)
    counts = np.bincount(flat_labels)
    areas_m2 = counts * pixel_area

    # Total storage volume per label (per wetland)
    volume_sums = np.bincount(flat_labels, weights=flat_depths)
    volumes_m3 = volume_sums * pixel_area

    # Preallocate median array (this is so that we can avoid for loops)
    medians = np.full(num_features + 1, np.nan)

    depths_by_label = defaultdict(list)
    for label_val, depth in zip(flat_labels, flat_depths):
        if label_val == 0:
            continue
        depths_by_label[label_val].append(depth)

    for label_val, depth_list in depths_by_label.items():
        medians[label_val] = np.median(depth_list)


    # Build dataframe
    wetlands_df_stats = pd.DataFrame({
        'wetland_id': np.arange(len(areas_m2)),
        'area_m2': areas_m2,
        'volume_m3': volumes_m3,
        'median_depth_m': medians
    })


    # Filter out background (label 0), small areas, and small volumes
    area_threshold = 100  # m2
    volume_threshold = 30  # m3
    wetlands_df_stats = wetlands_df_stats.query(f"wetland_id != 0 and area_m2 >= {area_threshold} and volume_m3 >= {volume_threshold}")


    # Write stats to CSV
    state.WETLANDS_STATS_CSV = state.HYDROCON_PROCESSED_PATH + "Wetlands_Stats.csv"
    wetlands_df_stats.to_csv(state.WETLANDS_STATS_CSV, index=False) 

    # Create mask of valid wetland IDs (from filtered df_stats)
    valid_ids = set(wetlands_df_stats["wetland_id"])
    valid_mask = np.isin(labeled_array, list(valid_ids))

    # Extract shapes (polygons) from the labeled array
    polygons = []
    labels = []

    with rasterio.open(depressions_raster_file + ".tif") as src:  # or any raster for transform/CRS
        transform = src.transform
        crs = src.crs

        for geom, val in rasterio.features.shapes(
            labeled_array.astype(np.int32),
            mask=valid_mask,
            transform=transform
        ):
            if val in valid_ids:
                polygons.append(shapely_shape(geom))
                labels.append(val)
    
    # Make GeoDataFrame
    gdf = gpd.GeoDataFrame({'wetland_id': labels, 'geometry': polygons}, crs=crs)

    # Join with statistics table
    gdf = gdf.merge(wetlands_df_stats, on='wetland_id')

    # Write results to shapefile
    state.WETLANDS_POLYGONS_SHAPEFILE = state.HYDROCON_PROCESSED_PATH + "Wetlands_Polygons_with_Stats.shp"
    gdf.to_file(state.WETLANDS_POLYGONS_SHAPEFILE)

    print("Inside gen_wetland_polygons(): Wetland polygons have been generated and saved to: ", state.WETLANDS_POLYGONS_SHAPEFILE)

    state.LAST_FUNCTION_RUN = "gen_wetland_polygons"
    save_state()

    return state.WETLANDS_POLYGONS_SHAPEFILE, gdf


def project_crs_subbasins_to_nhn(nhn_gdf, 
                             subbasins_gdf,
                             subbasins_filename):
    print("Starting project_crs_subbasins_to_nhn()...")

    # CRS for NHN shapefile
    input_NHN_crs = nhn_gdf.crs


    # Project subbasins data to match streams data
    subbasins_gdf_projected_nhn = subbasins_gdf.to_crs(input_NHN_crs)


    # Remove any non-alphanumeric characters from crs name
    input_NHN_crs_alnum = ''.join(c for c in \
                                str(input_NHN_crs) if c.isalnum())
    if len(input_NHN_crs_alnum) > 10:
        # truncate to first 10 characters to avoid errors due to long file names
        input_NHN_crs_alnum = input_NHN_crs_alnum[:10]


    # Projected subbasins shapefile name with path
    state.SUBBASINS_PROJ_NHN_FILE = state.HYDROCON_INTERIM_PATH + \
                            subbasins_filename + \
                            f"_projected_{input_NHN_crs_alnum}" 


    # Write projected subbasins data to shapefile
    subbasins_gdf_projected_nhn.to_file(state.SUBBASINS_PROJ_NHN_FILE + \
                                ".shp")


    # Check if shapefile projection aligns with DEM projection
    is_correctly_projected_clrh_nhn = (input_NHN_crs == subbasins_gdf_projected_nhn.crs)
                            

    # Print results
    print("Inside project_subbasins_to_nhn(): Shapefile projection is aligned with NHN projection: ", 
        is_correctly_projected_clrh_nhn)
    print(f"Inside project_subbasins_to_nhn(): The projected shapefile has been written to {state.SUBBASINS_PROJ_NHN_FILE}.")
    print("project_crs_subbasins_to_nhn() has ended.")

    state.LAST_FUNCTION_RUN = "project_crs_subbasins_to_nhn"
    save_state()

    # Return projected subbasins shapefile
    return subbasins_gdf_projected_nhn, state.SUBBASINS_PROJ_NHN_FILE


def project_subbasins_to_lidar(gdf, gdf_filename,
                                   lidar_filename, lidar_directory):
    print("Starting project_crs_subbasins_to_lidar()...")
                                       
    with rasterio.open(lidar_directory + \
                       lidar_filename + \
                       ".tif") as src:
        input_DEM_crs = src.crs
    
    # Project subbasins data to match DEM
    clrh_gdf_projected_lidar = gdf.to_crs(input_DEM_crs)
    
    # Remove any non-alphanumeric characters from crs name
    input_DEM_crs_alnum = ''.join(c for c in \
                                  str(input_DEM_crs) if c.isalnum())
    if len(input_DEM_crs_alnum) > 10:
        # truncate to first 10 characters to avoid errors due to long file names
        input_DEM_crs_alnum = input_DEM_crs_alnum[:10]  
    
    # Projected subbasins shapefile name with path
    state.CLRH_PROJ_LIDAR_FILE = state.HYDROCON_INTERIM_PATH + \
                            gdf_filename + \
                            f"_projected_{input_DEM_crs_alnum}"
    
    # Write projected subbasins data to shapefile
    clrh_gdf_projected_lidar.to_file(state.CLRH_PROJ_LIDAR_FILE + ".shp")
    
    # Check if shapefile projection aligns with DEM projection
    is_correctly_projected_clrh_lidar = (input_DEM_crs == clrh_gdf_projected_lidar.crs)         
    
    # Print results
    print("Inside project_crs_subbasins_to_lidar(): Shapefile projection is aligned with DEM projection: ", 
          is_correctly_projected_clrh_lidar)
    print(f"Inside project_crs_subbasins_to_lidar(): The projected shapefile has been written to {state.CLRH_PROJ_LIDAR_FILE}.")
    print("project_crs_subbasins_to_lidar() has ended.")

    state.LAST_FUNCTION_RUN = "project_crs_subbasins_to_lidar"
    save_state()

    return clrh_gdf_projected_lidar, input_DEM_crs, input_DEM_crs_alnum, state.CLRH_PROJ_LIDAR_FILE


def merge_rasters(lidar_files, gdf):
    print("Starting merge_rasters()....")
    
    # Turn string input into list
    LIDAR_FILENAMES_LIST = [f for f in lidar_files]

    # 1. Clip each raster to the subbasins shapefile for memory efficiency
    for file in LIDAR_FILENAMES_LIST:
        # 1. Project shapefile to match raster CRS

        # Open raster file and read its CRS
        with rasterio.open(state.HYDROCON_RAW_PATH + \
                        file + \
                        ".tif") as src:
            input_DEM_crs = src.crs
        
        # Project subbasins data to match raster CRS
        clrh_gdf_projected = gdf.to_crs(input_DEM_crs)

        # Check if shapefile projection aligns with DEM projection
        # is_correctly_projected_clrh = (input_DEM_crs == clrh_gdf_projected.crs)

        # 2. Clip raster to watershed boundary

        # Convert projected subbasins data to GeoJSON-like format
        shapes = [mapping(geom) for geom in clrh_gdf_projected.geometry]

        # Mask (clip) the input DEM file
        with rasterio.open(state.HYDROCON_RAW_PATH + \
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
        state.LIDAR_CLIPPED_FILE = state.HYDROCON_INTERIM_PATH + \
                                file + \
                                "_clip"

        # Write clipped DEM data into file
        with rasterio.open(state.LIDAR_CLIPPED_FILE + ".tif",
                        "w", 
                        **out_meta) as dest:
            dest.write(out_image)

    print("Inside merge_rasters(): Each raster has been clipped to subbasins shapefile.")

    # 2. Find the CRS for the highest resolution raster
    
    # Record the paths of all clipped raster files
    clipped_raster_paths = glob(f"{state.HYDROCON_INTERIM_PATH}*_clip.tif")


    # Function to get the resolution of a raster file
    def get_resolution(path):
        with rasterio.open(path) as src:
            return min(abs(src.res[0]), abs(src.res[1]))
        

    # Record the path for the highest-resolution raster
    highest_res_path = min(clipped_raster_paths,
                        key=get_resolution)


    # Record the resolution of the highest-resolution raster
    highest_res_resolution = get_resolution(highest_res_path)


    # Record the CRS for the highest resolution raster
    with rasterio.open(highest_res_path) as ref_src:
        target_crs = ref_src.crs

    print("Inside merge_rasters(): The CRS for the highest resolution raster has been found.")

    # 3. Project all rasters to the CRS of the highest resolution raster

    # Initialize an empty list to store reprojected raster paths
    reprojected_paths = []

    # Reproject rasters to the crs of the highest-resolution raster
    for path in clipped_raster_paths:

        # Output file name
        out_filename = os.path.splitext(os.path.basename(path))[0] + \
                                "_reprojected.tif"
        # Output file path
        out_path = os.path.join(state.HYDROCON_INTERIM_PATH, out_filename)
        
        with rasterio.open(path) as src:
            # Check if the raster's CRS is different from the target CRS
            if src.crs != target_crs:
                transform, width, height = calculate_default_transform(
                    src.crs, # current raster's CRS
                    target_crs, # target CRS (CRS of the highest-resolution raster)
                    src.width, src.height, *src.bounds # bounds of the current raster
                    )

                # Update metadata for the output raster
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    "compress": "lzw",              # apply LZW compression (for smaller file size)
                    "tiled": True,                  # enable tiling (for faster access)
                    "blockxsize": 256,              # set block size for tiling (for faster access)
                    "blockysize": 256               # set block size for tiling (for faster access)
                })

                # Reproject and write the raster
                with rasterio.open(out_path, 'w', **kwargs) as dst:
                    # Loop through each band in the source raster
                    for i in range(1, src.count + 1): 
                        reproject(
                            source=rasterio.band(src, i), # the input band to reproject
                            destination=rasterio.band(dst, i), # where to write the reprojected band
                            src_transform=src.transform, # how to interpret the source raster's coordinates
                            src_crs=src.crs, # source raster's CRS
                            dst_transform=transform, # how to interpret the destination raster's coordinates
                            dst_crs=target_crs, # target CRS
                            resampling=Resampling.bilinear, # resampling method
                            dst_nodata=src.nodata
                        )
                # Append the reprojected raster path to the list
                reprojected_paths.append(out_path)

                # Print confirmation message
                print(f"Raster {path} reprojected to {target_crs} and saved as {out_path}.")
            else:
                # Print message if no reprojection is needed
                print(f"Raster {path} is already in the target CRS. No reprojection needed.")

                # Append the original path to the list of reprojected paths
                reprojected_paths.append(path)
                
    print("Inside merge_rasters(): All rasters have been projected to the CRS of the highest-resolution raster.")
    
    # 4. Merge all rasters together

    # Path for the merged output raster
    state.LIDAR_FILENAME_NEW = "merged_average_dem"
    out_path_merged = os.path.join(state.HYDROCON_INTERIM_PATH, state.LIDAR_FILENAME_NEW + ".tif")
    out_path_merged_vrt = os.path.join(state.HYDROCON_INTERIM_PATH, "merged_virtual_raster.vrt")


    # Create a visual raster that includes all input rasters 
    # (virtual rasters are not written to disk, so more memory efficient!)
    vrt = gdal.BuildVRT(out_path_merged_vrt, reprojected_paths)


    # Now use gdal.Warp to resample and average overlapping areas
    gdal.Warp(
        out_path_merged,
        vrt,
        xRes=highest_res_resolution,
        yRes=highest_res_resolution,
        resampleAlg='average', # average elevation for overlapping areas
        options=gdal.WarpOptions(
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 
                            'TILED=YES', 
                            'BLOCKXSIZE=256', 
                            'BLOCKYSIZE=256'] # to limit memory usage and improve performance
        )
    )

    # Close the virtual raster to free up space
    vrt = None

    print("Inside merge_rasters(): Input rasters have been merged. Next step is to fill in the gaps.")
    
    # 5. Fill in gaps in merged raster

    # Path to the merged raster (input for filling nodata values)
    input_raster = out_path_merged
    # Path to the filled output raster
    output_raster = os.path.join(state.HYDROCON_INTERIM_PATH, state.LIDAR_FILENAME_NEW + "_filled.tif")


    def fill_with_buffer(src, dst, band_index=1, nodata_val=-9999, buffer=20):
        """
        Fill nodata values in a raster using local nearest-neighbor interpolation with a buffer.

        Parameters:
        ------------------------------
        - src: Source rasterio dataset
        - dst: Destination rasterio dataset
        - band_index: Index of the band to process (default is 1)
        - nodata_val: Value representing nodata in the raster (default is -9999)
        - buffer: Size of the buffer around nodata pixels to consider for interpolation (default is 20 pixels)
        """

        # Process block by block (otherwise will run out of memory for large rasters)
        # ji is the block index (row, column) and window is the Window object
        for ji, window in src.block_windows(band_index):
            

            # Extract starting row and column offsets and block dimensions
            row_off, col_off = window.row_off, window.col_off
            height, width = window.height, window.width


            # Define a larger window around the block (otherwise will not be able to fill nodata values at the edges)
            buffered_window = Window(
                max(0, col_off - buffer),
                max(0, row_off - buffer), 
                min(width + 2 * buffer, src.width - col_off + buffer), 
                min(height + 2 * buffer, src.height - row_off + buffer)
            )


            # Read buffered window
            data = src.read(band_index, window=buffered_window)
            mask = data == nodata_val

            # Skip blocks without gaps for efficiency
            if not np.any(mask):
                dst.write(data[buffer:-buffer, buffer:-buffer], band_index, window=window)
                continue


            # Perform local nearest-neighbor interpolation (distance transform gives index of nearest non-nodata pixel)
            inds = distance_transform_edt(mask, 
                                        return_distances=False, 
                                        return_indices=True)
            

            # Make a copy of the data to fill nodata values
            filled = data.copy()


            # Replace nodata values with nearest neighbor values
            filled[mask] = data[tuple(inds[:, mask])]


            # Extract central (original) window and write it (trim back to original block size to avoid overlap)
            filled_central = filled[buffer:buffer+height, buffer:buffer+width]
            dst.write(filled_central, band_index, window=window)

    # Open the input raster and create a new output raster with the same profile
    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -9999
        profile.update(nodata=nodata)

        # Update the profile for the output raster
        with rasterio.open(output_raster, 'w', **profile) as dst:
            fill_with_buffer(src, dst, band_index=1, nodata_val=nodata, buffer=50)


    # Rename lidar filename variable
    LIDAR_FILENAME_NEW = LIDAR_FILENAME_NEW + "_filled"

    print("Inside merge_rasters(): The gaps have been filled.")
    print("merge_rasters() has ended.")
    return LIDAR_FILENAME_NEW
                  

#--------------------RECOVERY FUNCTIONS------------------------

def recover_state(recovery_path = None):

    ## Recover dictionary from pickle file (for testing purposes)
    if "state" not in globals():
        if recovery_path is None:
            recovery_path = input("State data not found in current session. Enter the recovery folder path to load from pickle file.")
        pickle_file_path = Path(recovery_path) / "state.pkl"
        
        try:
            with open(pickle_file_path, 'rb') as f:
                state = pickle.load(f)

                # Promote the dictionary and its contents to global scope so that it can be accessed in the main script
                globals()["state"] = state

            print(f"State data loaded from: {pickle_file_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found at: {pickle_file_path}")
        
    

