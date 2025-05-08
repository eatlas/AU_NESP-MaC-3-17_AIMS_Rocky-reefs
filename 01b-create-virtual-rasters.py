#!/usr/bin/env python3
"""
Script: 01b-create-virtual-rasters.py

Purpose:
    This script scans a given directory (referred to as `--base-dir`) up to 3 levels deep,
    looking for subfolders that contain at least 2 `.tif` files. For each qualifying subfolder,
    it automatically generates a single `.vrt` file in the parent directory of that subfolder.
    This script needs to be run prior to opening `data/Preview-maps.gqz` in QGIS as the preview
    maps use the virtual rasters created by this script.

Key Features:
    - Ignores any folders that contain fewer than 2 `.tif` files.
    - Recursively descends only 3 levels below the specified base directory.
    - Uses a temporary file list for `gdalbuildvrt` to avoid command-line length limits.
    - Produces `.vrt` files named after the subfolder (e.g., subfolder `detector-inshore` => `detector-inshore.vrt`).
    - With --combine-regions, creates national-scale VRTs that combine regional image collections.

Usage:
    python 01b-create-output-virtual-rasters.py --base-dir <PATH_TO_TOP_LEVEL_DIRECTORY> [--combine-regions]

To reproduce the dataset on HPC:
    python 01b-create-virtual-rasters.py --base-dir ~/AU_AIMS_S2-comp --combine-regions

To reproduce the dataset on local machine: 
    python 01b-create-virtual-rasters.py --base-dir D:\\AU_AIMS_S2-comp --combine-regions

Dependencies:
    - Python 3.x
    - GDAL's command-line utilities (particularly `gdalbuildvrt`) must be installed and
      available in your system PATH.

Notes:
    - The script checks if `gdalbuildvrt` is installed before processing.
    - On Windows, large numbers of `.tif` files are handled gracefully by using a temporary file
      list instead of passing paths directly to the command line.
    - The script can be adapted for deeper or shallower recursion by modifying the depth checks.
"""

import os
import sys
import argparse
import subprocess
import tempfile

def check_gdalbuildvrt():
    """Check if gdalbuildvrt is available in the system PATH."""
    try:
        subprocess.run(["gdalbuildvrt", "--version"],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        # gdalbuildvrt exists but returned a non-zero exit code
        return True

def create_vrt_from_tif_files(tif_files, vrt_path):
    """Create a VRT file from a list of TIF files."""
    if len(tif_files) < 2:
        return False
        
    # Create a temporary file list for gdalbuildvrt
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        for tif in tif_files:
            tmp.write(tif + "\n")
        tmp_name = tmp.name

    try:
        subprocess.run([
            "gdalbuildvrt",
            "-input_file_list", tmp_name,
            vrt_path
        ], check=True)
        print(f"  -> Created: {vrt_path}")
        result = True
    except subprocess.CalledProcessError as e:
        print(f"  !! Error creating VRT: {e}")
        result = False
    finally:
        # Remove the temporary file list
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    
    return result

def create_vrt_for_folder(folder_path):
    """
    Create a .vrt for all .tif files in `folder_path` if it contains 2 or more .tif files.
    The .vrt is saved in the parent directory with the same name as `folder_path`.
    Returns True if a .vrt was created, False otherwise.
    """
    # Collect .tif files
    tif_files = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".tif"):
            full_path = os.path.join(folder_path, file_name)
            tif_files.append(os.path.abspath(full_path))

    # Only proceed if we have 2 or more TIFs
    if len(tif_files) < 2:
        return False

    parent_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    vrt_path = os.path.join(parent_dir, folder_name + ".vrt")

    print(f"  Creating VRT for folder: {folder_name}")
    return create_vrt_from_tif_files(tif_files, vrt_path)

def create_combined_regional_vrt(parent_folder):
    """
    Create a combined VRT from all TIF files in all subdirectories.
    This is used to create national-scale VRTs that combine regional image collections.
    """
    all_tif_files = []
    region_count = 0
    
    # Collect all TIF files from all subdirectories
    for region_dir in os.listdir(parent_folder):
        region_path = os.path.join(parent_folder, region_dir)
        if os.path.isdir(region_path):
            region_count += 1
            for file_name in os.listdir(region_path):
                if file_name.lower().endswith(".tif"):
                    full_path = os.path.join(region_path, file_name)
                    all_tif_files.append(os.path.abspath(full_path))
    
    # Only proceed if we found files from at least 2 regions
    if region_count < 2 or len(all_tif_files) < 2:
        return False
        
    folder_name = os.path.basename(parent_folder)
    vrt_path = os.path.join(parent_folder, f"{folder_name}_national.vrt")
    
    print(f"  Creating national-scale VRT for: {folder_name}")
    return create_vrt_from_tif_files(all_tif_files, vrt_path)

def main(base_dir, combine_regions=False):
    if not check_gdalbuildvrt():
        print("Error: gdalbuildvrt is not installed or not available in the system PATH.")
        sys.exit(1)

    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        print(f"Error: The specified base directory does not exist or is not a directory: {base_dir}")
        sys.exit(1)

    print(f"Scanning base directory (up to 3 levels): {base_dir}")

    # Determine how deep the base directory is (for limiting recursion)
    base_depth = base_dir.count(os.sep)
    
    # Track image type directories for potential combined VRTs
    if combine_regions:
        print("National-scale VRT creation enabled: will combine regional image collections")

    for root, dirs, files in os.walk(base_dir):
        depth = root.count(os.sep) - base_depth

        # If we're deeper than 3 levels, skip descending further
        if depth >= 3:
            dirs[:] = []
        
        # Attempt to create a VRT if `root` has 2+ TIFs
        if root != base_dir:  # Skip the base_dir itself
            create_vrt_for_folder(root)
            
        # If combine_regions is enabled and we're at depth 1 (image type level), 
        # create a national-scale VRT
        if combine_regions and depth == 1:
            create_combined_regional_vrt(root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a .vrt for each subfolder (up to 3 levels down) that contains >=2 .tif files."
    )
    parser.add_argument("--base-dir", required=True,
                        help="Path to the top-level directory to scan.")
    parser.add_argument("--combine-regions", action="store_true",
                        help="Create national-scale VRTs that combine regional image collections.")
    args = parser.parse_args()

    main(args.base_dir, args.combine_regions)
