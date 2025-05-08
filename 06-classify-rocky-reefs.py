"""
Rocky Reef Classification Script

Purpose:
    This script identifies rocky reef habitats in Australian coastal waters using Sentinel-2 satellite imagery.
    It processes satellite imagery tiles across multiple regions (NorthernAU, GBR) to generate rocky reef polygons
    in shapefile format.

Key Algorithms:
    1. Random Forest Classification:
       - Uses a pre-trained Random Forest model to predict rocky reef probabilities
       - Combines false-color (B5, B8, B12) and true-color (B2, B3, B4) Sentinel-2 imagery
       - Outputs probability maps scaled to 8-bit (1-255, with 0 as nodata)
       
    2. Post-processing Pipeline:
       - Clip probability values to [50,130] range and normalize
       - Apply median filter (7x7) to remove noise
       - Perform morphological closing to fill small gaps
       - Double resolution using bilinear interpolation
       - Threshold at 140 to create binary mask
       - Rasterize land mask and remove land pixels
       - Vectorize to polygons with topology-preserving simplification
       - Filter by area (minimum 900 m²)
       
    3. Region & Tile Processing:
       - Processes multiple geographic regions (NorthernAU, GBR)
       - Supports prioritization of specific tiles
       - Handles processing via local loops or SLURM array jobs

Usage:
    python 06-classify-rocky-reefs.py [options]
    
    Options:
        --priority-tiles  : Comma-separated list of tile IDs to process first
        --dataset-path    : Path to the satellite image dataset
        --geotiff-dir     : Temporary working folder for prediction images
        --landmask        : Path to the land mask shapefile
        --regions         : Comma-separated list of regions to process (NorthernAU,GBR)

Example:
    python 06-classify-rocky-reefs.py --priority-tiles "51KVB,51KWB,51LWC,51LXC,51LXD,51LYD,51LZE,52LCK,52LEK,52LGN,53LPH,53LPE,53LPC" --regions NorthernAU
    python 06-classify-rocky-reefs.py --priority-tiles "56KKA,55KFT,55KDV,55LLC,54LYM,54LXP" --regions GBR
"""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import joblib
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
import configparser

def extract_tile_id(filename):
    """
    Extract the tile ID from various filename patterns.
    
    Handles different formats:
    - All tide: AU_AIMS_MARB-S2-comp_p15_TrueColour_56KPU_v2_2015-2024.tif
    - Low tide: AU_AIMS_MARB-S2-comp_low-tide_p30_TrueColour_56KPU.tif
    - Low tide: AU_AIMS_MARB-S2-comp_low-tide_p30_NearInfraredFalseColour_56KPU.tif
    
    In all cases, returns the 5-character tile ID (e.g., "56KPU").
    Raises a ValueError if extraction fails.
    """
    # Remove file extension
    basename = os.path.splitext(filename)[0]
    
    # Check if this is an all-tide file (contains _v2_ pattern)
    if "_v2_" in basename:
        # Extract the segment before "_v2_"
        parts = basename.split("_v2_")[0].split("_")
        # The tile ID should be the last part before the v2 segment
        tile_id = parts[-1]
    else:
        # For low-tide files, the tile ID is the last segment
        parts = basename.split("_")
        tile_id = parts[-1]
    
    # Validation - tile IDs are typically 5 characters with digits and letters
    if len(tile_id) == 5 and tile_id.isalnum():
        return tile_id
    else:
        # If validation fails, raise an exception
        raise ValueError(f"Invalid tile ID format '{tile_id}' detected in filename '{filename}'. "
                         f"Expected a 5-character alphanumeric ID.")

def run_rf_prediction(tile_id, lt_false_path, lt_true_path, at_true_path, rocky_tif_path, rf, rocky_idx):
    """
    Runs the RF prediction for a single tile and saves the 'Rocky reef' probability as a single-band GeoTIFF.
    The prediction is performed block-by-block.
    The probability (0-1) is scaled to 1-255 (with 0 reserved for nodata).
    """
    print(f"  Running RF prediction for tile {tile_id}...")
    with rasterio.open(lt_false_path) as lt_false_src, rasterio.open(lt_true_path) as lt_true_src, rasterio.open(at_true_path) as at_true_src:
        if lt_false_src.width != lt_true_src.width or lt_false_src.height != lt_true_src.height:
            print(f"    Dimension mismatch for tile {tile_id}. Skipping prediction.")
            return False

        # Prepare output profile: single band, 8-bit, LZW compression.
        profile = lt_false_src.profile.copy()
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'LZW'
        })

        # Open output file for writing the rocky reef probability.
        with rasterio.open(rocky_tif_path, 'w', **profile) as dst:
            blocks = list(lt_false_src.block_windows(1))
            num_blocks = len(blocks)
            if num_blocks == 0:
                print(f"    No block windows found for tile {tile_id}.")
                return False

            # Simple GDAL-like progress indicator (one marker per percent).
            block_counter = 0
            last_percent = 0
            print("  Progress: 0", end='', flush=True)
            # Get nodata values for later.
            lt_false_nodata = lt_false_src.nodata
            lt_true_nodata = lt_true_src.nodata
            at_true_nodata = at_true_src.nodata

            for ji, window in blocks:
                block_counter += 1
                current_percent = int((block_counter / num_blocks) * 100)
                if current_percent > last_percent:
                    for p in range(last_percent + 1, current_percent + 1):
                        if p % 10 == 0:
                            print(f"{p}", end='', flush=True)
                        else:
                            print('.', end='', flush=True)
                    last_percent = current_percent

                # Read the block from both images.
                lt_false_window = lt_false_src.read(window=window)  # shape: (3, h, w) for B5, B8, B12
                lt_true_window = lt_true_src.read(window=window)    # shape: (3, h, w) for B2, B3, B4
                at_true_window = at_true_src.read(window=window)    # shape: (3, h, w) for B2, B3, B4

                # Combine into a 9-band array.
                combined = np.concatenate((lt_false_window, lt_true_window, at_true_window), axis=0)
                h, w = combined.shape[1], combined.shape[2]
                features = combined.reshape(9, -1).T  # shape: (h*w, 9)

                # Get class probabilities.
                probabilities = rf.predict_proba(features)  # shape: (num_pixels, n_classes)
                # Extract the probability for "Rocky reef" (using its index).
                rocky_probs = probabilities[:, rocky_idx]  # 1D array

                # Create a nodata mask (if a pixel is nodata in either source, mark it).
                if lt_false_nodata is not None:
                    lt_mask_false = np.all(lt_false_window == lt_false_nodata, axis=0)
                else:
                    lt_mask_false = np.zeros((h, w), dtype=bool)
                if lt_true_nodata is not None:
                    lt_mask_true = np.all(lt_true_window == lt_true_nodata, axis=0)
                else:
                    lt_mask_true = np.zeros((h, w), dtype=bool)
                if at_true_nodata is not None:
                    at_mask_true = np.all(at_true_window == at_true_nodata, axis=0)
                else:
                    at_mask_true = np.zeros((h, w), dtype=bool)
                nodata_mask = lt_mask_false | lt_mask_true | at_mask_true

                # Scale probability from 0-1 to 1-255.
                # For valid pixels: scaled = round(prob * 254) + 1; nodata pixels remain 0.
                scaled = (np.rint(rocky_probs * 254)).astype(np.uint8) + 1
                # Reshape to block.
                scaled = scaled.reshape(h, w)
                scaled[nodata_mask] = 0

                dst.write(scaled, window=window, indexes=1)
            print()  # Newline after progress output.
    return True

def postprocess_and_polygonize(rocky_tif_path, shapefile_path, land_gdf):
    """
    Postprocess the rocky reef probability GeoTIFF:
      - Clip non-zero pixel values to [50, 130] and linearly re-scale that range to [1,255].
      - Apply a 7-pixel median filter using OpenCV.
      - Apply a morphological closing: dilation (with a round kernel of 7x7) then erosion (same kernel) to fill holes.
      - Double the resolution using bilinear interpolation.
      - Apply a threshold of 140 to create a binary mask.
      - Rasterize the land mask and remove land pixels from the binary mask.
      - Convert the masked binary image to polygons (splitting multi-part geometries and simplifying them).
      - Remove polygons with an area less than 900 m².
      - Save the resulting shapefile at shapefile_path.
    """
    import cv2
    import rasterio
    from rasterio.features import shapes, rasterize
    from shapely.geometry import shape
    import geopandas as gpd
    import numpy as np
    from datetime import datetime
    import os

    # Read the original GeoTIFF.
    with rasterio.open(rocky_tif_path) as src:
        img = src.read(1)  # 8-bit image; 0 = nodata.
        transform = src.transform

    # Convert image to float32 for processing.
    img_float = img.astype(np.float32)
    mask = (img_float > 0)

    # Clip valid pixels to [50,130] and normalize to [1,255].
    img_clipped = img_float.copy()
    img_clipped[mask] = np.clip(img_clipped[mask], 50, 130)
    img_norm = img_clipped.copy()
    img_norm[mask] = ((img_norm[mask] - 50) / 80) * 254 + 1
    img_norm[~mask] = 0
    img_norm = img_norm.astype(np.uint8)

    # Apply a 7x7 median filter. Use a largish kernel to only keep rocky reefs that are at least 50 m across.
    img_median = cv2.medianBlur(img_norm, 7)
    
    # Morphological closing: dilate then erode with elliptical kernels.
    # Use this to fill small holes in the reef polygons.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_closed = cv2.dilate(img_median, kernel, iterations=1)
    img_closed = cv2.erode(img_closed, kernel, iterations=1)

    # Double the resolution using bilinear interpolation.
    # This is done to improve the accuracy of polygon conversion step.
    height, width = img_closed.shape
    img_resized = cv2.resize(img_closed, (width*2, height*2), interpolation=cv2.INTER_LINEAR)

    # Apply threshold of 140 to create a binary mask.
    # A value of 140 was chose to give a good balance between false positives and false negatives.
    ret, img_thresh = cv2.threshold(img_resized, 140, 255, cv2.THRESH_BINARY)

    # Update the affine transform since resolution was doubled.
    new_transform = rasterio.Affine(transform.a / 2, transform.b, transform.c,
                                    transform.d, transform.e / 2, transform.f)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Rasterizing land mask and applying raster space clipping")

    # Rasterize the land geometries to the same grid as img_thresh.
    # It is faster to perform a rough clipping in raster space than in vector space.
    # Since out the land mask we use is shrunk at little any edge effects in
    # pixel space will the cleaned up in the final land clipping.
    land_mask = rasterize(
        [(geom, 255) for geom in land_gdf.geometry],
        out_shape=img_thresh.shape,
        transform=new_transform,
        fill=0,
        all_touched=True,
        dtype='uint8'
    )

    # Remove rocky reef pixels that fall on land.
    img_thresh[land_mask == 255] = 0

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Converting raster mask to polygons")

    # Extract polygons from the modified binary image (only for value 255).
    # Apply simplification that is close to 1 pixel in size, this helps remove
    # any remanent stair case steps in the polygons left over from the raster
    # to polygon conversion.
    polys = []
    for geom, val in shapes(img_thresh, transform=new_transform, mask=(img_thresh==255)):
        if val == 255:
            geom_shape = shape(geom)
            # Split multipolygons into individual polygons.
            if geom_shape.geom_type == 'MultiPolygon':
                for poly in geom_shape:
                    polys.append(poly.simplify(0.00007, preserve_topology=True))
            elif geom_shape.geom_type == 'Polygon':
                polys.append(geom_shape.simplify(0.00007, preserve_topology=True))

    if not polys:
        print("  No rocky reef polygons extracted after raster-based clipping.")
        return False

    # Create GeoDataFrame from the extracted polygons (initially in EPSG:4326).
    reef_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
    reef_gdf['tile_id'] = os.path.basename(shapefile_path)

    # Reproject to a metric CRS for accurate area calculation.
    # Note: Adjust the EPSG code as needed for your study area.
    reef_gdf_metric = reef_gdf.to_crs(epsg=3857)

    # Compute area (in square metres) and filter out features smaller than 900 m².
    # This is to limit help remove small artefacts.
    reef_gdf_metric = reef_gdf_metric[reef_gdf_metric.area >= 900]

    if reef_gdf_metric.empty:
        print("  No rocky reef polygons remain after area filtering.")
        return False

    # Reproject back to EPSG:4326.
    reef_gdf = reef_gdf_metric.to_crs("EPSG:4326")

    # Save the resulting polygons as a shapefile.
    reef_gdf.to_file(shapefile_path)
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Shapefile saved to {shapefile_path}")
    return True

def process_tile_tile(tile_file, lt_false_dir, lt_true_files, at_true_files, geotiff_dir, shapefile_dir, rf, rocky_idx, land_gdf, region):
    """
    Process a single tile: if the output shapefile exists, skip.
    Otherwise, run the RF prediction (if the prediction GeoTIFF doesn't exist) and then
    postprocess it to create a shapefile (clipped with the provided land mask).
    """
    tile_id = extract_tile_id(tile_file)
    shapefile_path = os.path.join(shapefile_dir, f"RockyReef_{region}_{tile_id}.shp")
    if os.path.exists(shapefile_path):
        print(f"Tile {tile_id} ({region}): shapefile already exists. Skipping tile.")
        return

    # Determine input paths.
    lt_false_path = os.path.join(lt_false_dir, tile_file)
    
    # Check if we have matching true-color and all-tide images
    if tile_id not in lt_true_files:
        print(f"Tile {tile_id} ({region}): No matching low tide true-colour image found. Skipping tile.")
        return
    lt_true_path = lt_true_files[tile_id]
    
    if tile_id not in at_true_files:
        print(f"Tile {tile_id} ({region}): No matching all tide true-colour image found. Skipping tile.")
        return
    at_true_path = at_true_files[tile_id]
    
    rocky_tif_path = os.path.join(geotiff_dir, f"Classified-{region}_{tile_id}.tif")

    # Run RF prediction if the prediction GeoTIFF doesn't exist.
    if not os.path.exists(rocky_tif_path):
        success = run_rf_prediction(tile_id, lt_false_path, lt_true_path, at_true_path, rocky_tif_path, rf, rocky_idx)
        if not success:
            print(f"Tile {tile_id} ({region}): RF prediction failed. Skipping tile.")
            return
    else:
        print(f"Tile {tile_id} ({region}): Prediction GeoTIFF already exists, skipping RF prediction.")

    # Postprocess the prediction GeoTIFF and polygonize, clipping with the land mask.
    success = postprocess_and_polygonize(rocky_tif_path, shapefile_path, land_gdf)
    if not success:
        print(f"Tile {tile_id} ({region}): Post-processing failed or no polygons extracted.")
    else:
        print(f"Tile {tile_id} ({region}): Processing complete.")

def process_region(region, dataset_path, geotiff_dir, shapefile_dir, rf, rocky_idx, land_gdf, priority_tiles):
    """
    Process all tiles in a specific region (NorthernAU or GBR).
    """
    lt_false_dir = os.path.join(dataset_path, 'low_tide_infrared', region)
    lt_true_dir = os.path.join(dataset_path, 'low_tide_true_colour', region)
    at_true_dir = os.path.join(dataset_path, '15th_percentile', region)  # All tide true color directory
    
    if not os.path.isdir(lt_false_dir) or not os.path.isdir(lt_true_dir) or not os.path.isdir(at_true_dir):
        print(f"Error: One or more directories not found for region {region}")
        return 0
        
    # Build dictionaries for true-colour files in this region
    lt_true_files = {}
    for file in os.listdir(lt_true_dir):
        if file.lower().endswith('.tif'):
            tile_id = extract_tile_id(file)
            lt_true_files[tile_id] = os.path.join(lt_true_dir, file)
            
    # Build dictionary for all-tide true-color files
    at_true_files = {}
    for file in os.listdir(at_true_dir):
        if file.lower().endswith('.tif'):
            tile_id = extract_tile_id(file)
            at_true_files[tile_id] = os.path.join(at_true_dir, file)
    # List all false-colour files
    false_files = [f for f in os.listdir(lt_false_dir) if f.lower().endswith('.tif')]

    # Reorder false_files so that any tile in the priority list is processed first
    priority_false_files = [f for f in false_files if extract_tile_id(f) in priority_tiles]
    non_priority_false_files = [f for f in false_files if extract_tile_id(f) not in priority_tiles]
    ordered_false_files = priority_false_files + non_priority_false_files

    total_files = len(ordered_false_files)
    print(f"Found {total_files} false-colour files to process in region {region}.")
    print(f"Found {len(lt_true_files)} low-tide true-colour files.")
    print(f"Found {len(at_true_files)} all-tide true-colour files.")
    
    # Check for SLURM_ARRAY_TASK_ID to determine if we should process a single tile
    slurm_task = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_task is not None:
        try:
            idx = int(slurm_task)
        except ValueError:
            print("Error: SLURM_ARRAY_TASK_ID is not a valid integer.")
            return 0
            
        if idx < 0 or idx >= total_files:
            print(f"Error: SLURM_ARRAY_TASK_ID {idx} is out of bounds for region {region} (0-{total_files-1}).")
            return 0
            
        tile_file = ordered_false_files[idx]
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{current_time}] Processing tile {extract_tile_id(tile_file)} in {region} (SLURM_ARRAY_TASK_ID={idx})...")
        process_tile_tile(tile_file, lt_false_dir, lt_true_files, at_true_files, geotiff_dir, shapefile_dir, rf, rocky_idx, land_gdf, region)
        return 1
    else:
        processed_count = 0
        for i, tile_file in enumerate(ordered_false_files, 1):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{current_time}] Processing tile {extract_tile_id(tile_file)} in {region} ({i}/{total_files})...")
            process_tile_tile(tile_file, lt_false_dir, lt_true_files, at_true_files, geotiff_dir, shapefile_dir, rf, rocky_idx, land_gdf, region)
            processed_count += 1
        return processed_count

def main():

    # Read configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    version = config.get('general', 'version')

    parser = argparse.ArgumentParser(
        description="Progressively process each tile: apply RF prediction, post-process, and output a shapefile per tile."
    )
    parser.add_argument(
        '--priority-tiles', 
        type=str, 
        help="Comma-separated list of tile IDs to process first", 
        default=""
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        help="Path to the satellite image dataset",
        default=r'D:\AU_AIMS_S2-comp'
    )
    parser.add_argument(
        '--geotiff-dir',
        type=str,
        help="Temporary working folder for prediction images",
        default=r'C:\Temp\gis\AU_NESP-MaC-3-17_AIMS_Rocky-reefs'
    )
    parser.add_argument(
        '--landmask',
        type=str,
        help="Path to the cached (processed) land mask shapefile",
        default=f'data/{version}/in/landmask/Coastline-50k_trimmed.shp'
    )
    parser.add_argument(
        '--regions',
        type=str,
        help="Comma-separated list of regions to process (NorthernAU,GBR)",
        default="NorthernAU,GBR"
    )
    args = parser.parse_args()

    # Process priority tiles if provided.
    priority_tiles = []
    if args.priority_tiles:
        priority_tiles = [pt.strip() for pt in args.priority_tiles.split(',') if pt.strip()]

    # Define directories using the provided command line arguments.
    dataset_path = args.dataset_path
    geotiff_dir = args.geotiff_dir
    os.makedirs(geotiff_dir, exist_ok=True)
    
    shapefile_dir = r'working/06'
    os.makedirs(shapefile_dir, exist_ok=True)

    # Load the cached land mask.
    landmask_path = args.landmask
    if os.path.exists(landmask_path):
        print("Loading cached adjusted land mask...")
        cached_gdf = gpd.read_file(landmask_path)
        # Assume the cached file contains a single geometry.
        land_mask_geom = cached_gdf.geometry.unary_union
    else:
        print(f"Error: Land mask not found at {landmask_path}.")
        print("Please run the land mask processing script first or download the land mask.")
        sys.exit(1)

    print("Converting cached land mask to GeoDataFrame...")
    land_gdf = gpd.GeoDataFrame(geometry=[land_mask_geom], crs="EPSG:4326")

    print("Loading trained model")
    model_path = r'working/training-data/random_forest_model.pkl'
    encoder_path = r'working/training-data/label_encoder.pkl'
    rf = joblib.load(model_path)
    le = joblib.load(encoder_path)

    # Determine the index for "Rocky reef" in the model's classes.
    try:
        rocky_idx = list(le.classes_).index("Rocky reef")
    except ValueError:
        print("Error: 'Rocky reef' is not present in the model's classes.")
        sys.exit(1)

    # Get regions to process
    regions = [region.strip() for region in args.regions.split(',') if region.strip()]
    if not regions:
        regions = ["NorthernAU", "GBR"]
    
    total_processed = 0
    for region in regions:
        print(f"\n--- Processing region: {region} ---")
        processed = process_region(region, dataset_path, geotiff_dir, shapefile_dir, rf, rocky_idx, land_gdf, priority_tiles)
        total_processed += processed
        
    print(f"\nProcessing complete. {total_processed} tiles processed across {len(regions)} regions.")

if __name__ == '__main__':
    main()


