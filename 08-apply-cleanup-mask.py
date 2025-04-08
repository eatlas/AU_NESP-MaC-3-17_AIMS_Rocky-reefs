"""
This script applies a clean up mask (data/in/clean-up/AU-clean-up-mask.shp) to the raw rocky reef data (working/07/raw-rocky-reef.shp). 
This is used to delete mapped portions that incorrectly mapped as rocky reef. The clean up mask is manually created in QGIS. 
The creation of the clean up mask process was done by review against the Sentinel 2 composite imagery (NESP 3.17 Low tide and All tide), 
along with occassional checks against Google Earth imagery.

This cleanup helps to correct classification errors that tend to occur in the following situations:
1. Mangroves
2. Some very brown sediment areas

This script also optionally clips the rocky reef data to the high resolution Coastline 50k shapefile. 
(data/in-3p/Coast50k_2024/Split/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_split.shp). This version
of the coastline dataset corresponds to the land area is split into a 2x2 degree grid to make the polygons
limited in size. 

In the previous stages of processing the rocky reefs were only clipped to a shrunken (small -buffer), 
simplifed version of the Coastline 50k to speed up processing. Clipping to the full resolution coastline 
is slow and so can be disabled during the development of the clean up mask. 
"""
#!/usr/bin/env python3

import argparse
import sys
import os
import geopandas as gpd
from shapely.ops import unary_union

# Constants for file paths
RAW_ROCKY_REEF_PATH = "working/07/raw-rocky-reef.shp"
CLEAN_UP_MASK_PATH = "data/in/cleanup/Rocky-reef-cleanup-mask.shp"
COASTLINE_PATH = "data/in-3p/Coast50k_2024/Split/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_split.shp"
OUTPUT_TEMPLATE = "data/out/AU_NESP-MaC-3-17_AIMS_Rocky-reefs_{version}.shp"
VERSION_FILE = "VERSION.txt"

def main():
    parser = argparse.ArgumentParser(
        description="Clean up rocky reef data by applying a manual mask and optionally clipping to the coastline."
    )
    parser.add_argument(
        "--clip", action="store_true",
        help="Enable clipping of the cleaned rocky reef data to the high-resolution coastline."
    )
    args = parser.parse_args()

    # Read version number from VERSION.txt
    print(f"Reading version from '{VERSION_FILE}'...")
    try:
        with open(VERSION_FILE, "r") as f:
            version = f.read().strip()
    except Exception as e:
        print(f"Error reading version from '{VERSION_FILE}': {e}")
        sys.exit(1)
    
    output_path = OUTPUT_TEMPLATE.format(version=version)
    print(f"Output will be saved to '{output_path}'")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Creating it...")
        os.makedirs(output_dir, exist_ok=True)
        
    # Load raw rocky reef data
    print(f"Loading raw rocky reef data from '{RAW_ROCKY_REEF_PATH}'...")
    try:
        raw_reef = gpd.read_file(RAW_ROCKY_REEF_PATH)
    except Exception as e:
        print(f"Error loading raw rocky reef data from '{RAW_ROCKY_REEF_PATH}': {e}. "
              "Please ensure that the '07-merge-scenes.py' script has been run.")
        sys.exit(1)

    # Load clean-up mask
    print(f"Loading clean-up mask from '{CLEAN_UP_MASK_PATH}'...")
    try:
        cleanup_mask = gpd.read_file(CLEAN_UP_MASK_PATH)
    except Exception as e:
        print(f"Error loading clean-up mask data from '{CLEAN_UP_MASK_PATH}': {e}. "
              "Please ensure that the '01c-download-input-data.py' script has been run.")
        sys.exit(1)

    # Ensure all data are in EPSG:4326
    if raw_reef.crs.to_string() != "EPSG:4326":
        print("Reprojecting raw rocky reef data to EPSG:4326...")
        raw_reef = raw_reef.to_crs(epsg=4326)
    if cleanup_mask.crs.to_string() != "EPSG:4326":
        print("Reprojecting clean-up mask data to EPSG:4326...")
        cleanup_mask = cleanup_mask.to_crs(epsg=4326)

    # Create union of clean-up mask geometries
    print("Creating union of clean-up mask geometries...")
    mask_union = cleanup_mask.unary_union

    # Apply the clean-up mask: remove masked areas from the raw rocky reef data
    print("Applying clean-up mask to raw rocky reef data...")
    raw_reef["geometry"] = raw_reef.geometry.apply(lambda geom: geom.difference(mask_union))
    cleaned_reef = raw_reef[~raw_reef.geometry.is_empty].copy()

    # Optionally clip to coastline
    if args.clip:
        print(f"Loading coastline data from '{COASTLINE_PATH}'...")
        try:
            coastline = gpd.read_file(COASTLINE_PATH)
        except Exception as e:
            print(f"Error loading coastline data from '{COASTLINE_PATH}': {e}. "
                  "Please ensure that the '01c-download-input-data.py' script has been run.")
            sys.exit(1)
        if coastline.crs.to_string() != "EPSG:4326":
            print("Reprojecting coastline data to EPSG:4326...")
            coastline = coastline.to_crs(epsg=4326)

        print("Creating union of coastline geometries...")
        coastline_union = coastline.unary_union

        print("Clipping cleaned rocky reef data to the coastline...")
        cleaned_reef["geometry"] = cleaned_reef.geometry.apply(lambda geom: geom.intersection(coastline_union))
        cleaned_reef = cleaned_reef[~cleaned_reef.geometry.is_empty].copy()
    else:
        print("Skipping coastline clipping as per user request.")

    # Save the final cleaned data as a shapefile
    print(f"Saving cleaned rocky reef data to '{output_path}'...")
    try:
        cleaned_reef.to_file(output_path)
    except Exception as e:
        print(f"Error saving output shapefile: {e}")
        sys.exit(1)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
