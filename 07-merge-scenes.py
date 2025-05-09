import os
import geopandas as gpd
import pandas as pd
import time
import argparse

"""
This script merges and dissolves all rocky reef scene polygons into a single shapefile.

Performing the dissolving in a binary hierarchical manner is much faster than as a single dissolve, 
or as a linear dissolve (dissolving each shapefile into the result one at a time). For the
NorthernAU region the binary dissolve was 6x faster than the linear dissolve. The single dissolve
didn't complete after a couple of hours.

The merge processing time is highly variable and some merges take several hours.

The version and sigma are used to determine the filenames of the input and output files.

The following are the command lines used to reproduce the full dataset:
python 07-merge-scenes.py

Merge the binary classifier results:
python 07-merge-scenes.py --binary-classifier
"""

def main():
    parser = argparse.ArgumentParser(
        description="Combine and dissolve rocky reef shapefiles into a single shapefile."
    )

    parser.add_argument(
        '--binary-classifier',
        action='store_true',
        help="Pickup the files from the binary classifier instead of the multi-class model"
    )
    args = parser.parse_args()

    # Set the input path based on the classifier type
    if args.binary_classifier:
        input_path = "working/06-binary"
        output_dir = "working/07-binary"
    else:
        input_path = "working/06-multi"
        output_dir = "working/07-multi"

    output_file = os.path.join(output_dir, f"raw-rocky-reef.shp")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load all shapefiles and merge non-empty ones
    file_paths = [
        os.path.join(input_path, file_name) 
        for file_name in os.listdir(input_path) 
        if file_name.endswith(".shp") and file_name.startswith("RockyReef_")
    ]

    start_load_time = time.time()

    # Filter out empty shapefiles. These happen for tiles where the area is just 
    # open water

    geo_dataframes = []
    for file_path in file_paths:
        gdf = gpd.read_file(file_path)
        if not gdf.empty:
            print(f"Loaded {os.path.basename(file_path)}, appending")
            geo_dataframes.append(gdf)
        else:
            print(f"Loaded {os.path.basename(file_path)}, empty")


    print(f"Loading time: {time.time() - start_load_time:.2f} s")

    start_dissolve_time = time.time()
    # Recursive binary dissolve function
    def binary_dissolve(gdf_list):
        if len(gdf_list) == 1:
            return gdf_list[0]
        
        print(f"Dissolving {len(gdf_list)/2} pairs")
        start_dissolve_level_time = time.time()
        next_level = []
        for i in range(0, len(gdf_list), 2):
            if i + 1 < len(gdf_list):
                print(f"  Dissolving pair: {i} and {i+1}")
                merged_gdf = pd.concat([gdf_list[i], gdf_list[i+1]], ignore_index=True)
                dissolved = merged_gdf.dissolve()
                next_level.append(dissolved)
            else:
                # Odd element, just carry forward
                next_level.append(gdf_list[i])
        print(f"Dissolve level time: {time.time() - start_dissolve_level_time:.2f} s")
        # Recursively process next level
        return binary_dissolve(next_level)

    # Perform binary dissolve
    if geo_dataframes:
        print("Starting binary dissolve...")
        dissolved_final = binary_dissolve(geo_dataframes)
        print("Binary dissolve complete.")
        
        print(f"Dissolve time: {time.time() - start_dissolve_time:.2f} s")
        
        # Convert dissolved result to single-part polygons
        print("Converting to single-part polygons...")
        singlepart_final = dissolved_final.explode(index_parts=False).reset_index(drop=True)

        # Save the final output
        print(f"Saving dissolved result to {output_file}...")
        singlepart_final.to_file(output_file)
        print("Processing complete.")
    else:
        print("No valid shapefiles found for processing.")

if __name__ == '__main__':
    main()