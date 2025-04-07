import os
import argparse
import geopandas as gpd

def process_land_mask(original_landmask_path, cached_landmask_path):
    # Check if the cached file already exists.
    if os.path.exists(cached_landmask_path):
        print(f"Cached land mask already exists at: {cached_landmask_path}")
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
    os.makedirs(os.path.dirname(cached_landmask_path), exist_ok=True)
    # Save the adjusted land mask as a shapefile.
    adjusted_gdf.to_file(cached_landmask_path)
    print(f"Adjusted land mask cached to: {cached_landmask_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Process and cache an adjusted land mask. This step is slow and only needs to be done once."
    )
    parser.add_argument(
        '--original-landmask', 
        type=str, 
        required=True,
        help="Path to the original land mask shapefile.",
        default="data/in-3p/Coast50k_2024/Simp/AU_NESP-MaC-3-17_AIMS_Aus-Coastline-50k_2024_V1-1_simp.shp"
    )
    parser.add_argument(
        '--cached-landmask', 
        type=str, 
        required=True,
        help="Path to save the processed (cached) land mask shapefile."
        default= "working/05/Coastline-50k_trimmed.shp"
    )
    args = parser.parse_args()
    
    process_land_mask(args.original_landmask, args.cached_landmask)

if __name__ == '__main__':
    main()
