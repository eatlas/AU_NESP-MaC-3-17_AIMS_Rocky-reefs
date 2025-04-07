#!/usr/bin/env python3
"""
Script: 06d-create-output-virtual-rasters.py

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

Usage:
    python 06d-create-output-virtual-rasters.py --base-dir <PATH_TO_TOP_LEVEL_DIRECTORY>

To reproduce the dataset:
    python 06d-create-virtual-rasters.py --base-dir data/in-3p
    python 06d-create-virtual-rasters.py --base-dir working-data


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

    # Create a temporary file list for gdalbuildvrt
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        for tif in tif_files:
            tmp.write(tif + "\n")
        tmp_name = tmp.name

    print(f"  Creating VRT for folder: {folder_name}")
    try:
        subprocess.run([
            "gdalbuildvrt",
            "-input_file_list", tmp_name,
            vrt_path
        ], check=True)
        print(f"  -> Created: {vrt_path}")
    except subprocess.CalledProcessError as e:
        print(f"  !! Error creating VRT for {folder_name}: {e}")
    finally:
        # Remove the temporary file list
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    return True

def main(base_dir):
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

    for root, dirs, files in os.walk(base_dir):
        depth = root.count(os.sep) - base_depth

        # If we're deeper than 3 levels, skip descending further
        if depth >= 3:
            dirs[:] = []
        
        # Attempt to create a VRT if `root` has 2+ TIFs
        if root != base_dir:  # Skip the base_dir itself if needed
            create_vrt_for_folder(root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a .vrt for each subfolder (up to 3 levels down) that contains >=3 .tif files."
    )
    parser.add_argument("--base-dir", required=True,
                        help="Path to the top-level directory to scan.")
    args = parser.parse_args()

    main(args.base_dir)
