"""
This script creates the land mask that can be downloaded diectly from in/landmask.
This land mask has a negative buffer of 0.0005° (~50 m) to allow most of the features
detected over land areas to be clipped, whilst still allowing the final clipping 
to be perform on higher resolution land mask (i.e. not the simplified version of the
Coastline 50k). After pulling back from the coastline we simplify to make the 
application of the land mask faster. 

I found that this processing was so slow (> 20 min) that I never waited for it to finish. I 
instead performed the same processing in QGIS and saved the result as a shapefile.
These operations take about 30 seconds in QGIS.
The script is provided here for completeness, but it is not used in the main workflow.
"""
import os
import argparse
import geopandas as gpd

def process_land_mask(original_landmask_path, output_landmask_path):
    # Check if the cached file already exists.
    if os.path.exists(output_landmask_path):
        print(f"Land mask already exists at: {output_landmask_path}")
        return

    print("Processing adjusted land mask...")
    # Read the original land mask.
    land_mask_gdf = gpd.read_file(original_landmask_path)
    # Merge all features into a single geometry.
    land_mask_geom = land_mask_gdf.unary_union
    # Apply a negative buffer of 0.0005°.
    land_mask_geom = land_mask_geom.buffer(-0.0005)
    # Simplify the geometry with a tolerance of 0.0001°.
    land_mask_geom = land_mask_geom.simplify(0.0001, preserve_topology=True)
    # Create a new GeoDataFrame with the adjusted geometry.
    adjusted_gdf = gpd.GeoDataFrame({'geometry': [land_mask_geom]}, crs=land_mask_gdf.crs)
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(output_landmask_path), exist_ok=True)
    # Save the adjusted land mask as a shapefile.
    adjusted_gdf.to_file(output_landmask_path)
    print(f"Adjusted land mask cached to: {output_landmask_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Process and cache an adjusted land mask. This step is slow and only needs to be done once."
    )
    parser.add_argument(
        '--original-landmask', 
        type=str, 
        help="Path to the original land mask shapefile.",
        default="data/in-3p/Coast50k_2024/Simp/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_simp.shp"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        help="Path to save the processed land mask shapefile.",
        default= "data/in/landmask/Coastline-50k_trimmed.shp"
    )
    args = parser.parse_args()
    
    process_land_mask(args.original_landmask, args.output)

if __name__ == '__main__':
    main()
