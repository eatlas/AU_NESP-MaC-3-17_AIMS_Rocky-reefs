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
# Directory for saving the generated rocky reef shapefiles for each tile.
MULTI_SHAPEFILE_DIR = 'working/06-multi_w{weight}'
BINARY_SHAPEFILE_DIR = 'working/06-binary_w{weight}'

MULTI_OUT_DIR = 'working/07-multi_w{weight}'
BINARY_OUT_DIR = 'working/07-binary_w{weight}'

def main():
    parser = argparse.ArgumentParser(
        description="Combine and dissolve rocky reef shapefiles into a single shapefile."
    )

    parser.add_argument(
        '--binary-classifier',
        action='store_true',
        help="Pickup the files from the binary classifier instead of the multi-class model"
    )

    parser.add_argument(
        '--weight',
        type=str,
        help="Weight of the rocky reef class in the model. Must be one of the values calculated in 04-train-random-forest.py",
        choices=["1.0", "1.5", "2.0", "2.5", "3.0"],
        default="2.0"
    )

    args = parser.parse_args()

    weight = args.weight

    # Set the input path based on the classifier type
    if args.binary_classifier:
        input_path = BINARY_SHAPEFILE_DIR.format(weight=weight)
        output_dir = BINARY_OUT_DIR.format(weight=weight)
    else:
        input_path = MULTI_SHAPEFILE_DIR.format(weight=weight)
        output_dir = MULTI_OUT_DIR.format(weight=weight)

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
                try:
                    # Validate and repair geometries before merging
                    for idx, gdf in enumerate([gdf_list[i], gdf_list[i+1]]):
                        # Reset index to avoid "level_0 already exists" errors
                        gdf.reset_index(drop=True, inplace=True)
                        
                        # Check for invalid geometries
                        invalid_mask = ~gdf.geometry.is_valid
                        if invalid_mask.any():
                            print(f"    Repairing {invalid_mask.sum()} invalid geometries in dataframe {i+idx}")
                            # Apply buffer(0) to repair
                            gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
                    
                    # Merge and dissolve with a safer method
                    merged_gdf = pd.concat([gdf_list[i], gdf_list[i+1]], ignore_index=True)
                    
                    # Try an alternative approach to dissolve that handles index issues
                    try:
                        # Method 1: Use union_all() directly 
                        geometry = merged_gdf.geometry.union_all()
                        dissolved = gpd.GeoDataFrame(geometry=[geometry], crs=merged_gdf.crs)
                    except Exception as e:
                        print(f"    Falling back to standard dissolve: {str(e)}")
                        # Method 2: Regular dissolve with explicit index handling
                        # Drop any existing level_0 column if it exists
                        if 'level_0' in merged_gdf.columns:
                            merged_gdf = merged_gdf.drop('level_0', axis=1)
                        # Use dissolve with explicit parameters
                        dissolved = merged_gdf.dissolve(by=None, aggfunc='first', as_index=False)
                    
                    # Ensure the result is valid
                    if not dissolved.geometry.is_valid.all():
                        print("    Post-dissolve repair needed")
                        dissolved['geometry'] = dissolved.geometry.buffer(0)
                        
                    next_level.append(dissolved)
                except Exception as e:
                    print(f"    Error dissolving pair {i} and {i+1}: {str(e)}")
                    # Fall back to simpler approach - just concat without dissolve and fix later
                    merged_gdf = pd.concat([gdf_list[i], gdf_list[i+1]], ignore_index=True)
                    next_level.append(merged_gdf)
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