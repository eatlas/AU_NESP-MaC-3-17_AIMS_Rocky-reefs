"""
python 06-classify-rocky-reefs.py --priority-tiles "51KVB,51KWB,51LWC,51LXC,51LXD,51LYD,51LZE,52LCK,52LEK,52LGN,53LPH,53LPE,53LPC"
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

def extract_tile_id(filename):
    """Extract the tile ID from a filename (assumes it's the part after the last underscore)."""
    return filename.rsplit('_', 1)[1].replace('.tif','')

def run_rf_prediction(tile_id, false_path, true_path, rocky_tif_path, rf, rocky_idx):
    """
    Runs the RF prediction for a single tile and saves the 'Rocky reef' probability as a single‐band GeoTIFF.
    The prediction is performed block‐by‐block.
    The probability (0-1) is scaled to 1–255 (with 0 reserved for nodata).
    """
    print(f"  Running RF prediction for tile {tile_id}...")
    with rasterio.open(false_path) as false_src, rasterio.open(true_path) as true_src:
        if false_src.width != true_src.width or false_src.height != true_src.height:
            print(f"    Dimension mismatch for tile {tile_id}. Skipping prediction.")
            return False

        # Prepare output profile: single band, 8-bit, LZW compression.
        profile = false_src.profile.copy()
        profile.update({
            'dtype': rasterio.uint8,
            'count': 1,
            'compress': 'LZW'
        })

        # Open output file for writing the rocky reef probability.
        with rasterio.open(rocky_tif_path, 'w', **profile) as dst:
            blocks = list(false_src.block_windows(1))
            num_blocks = len(blocks)
            if num_blocks == 0:
                print(f"    No block windows found for tile {tile_id}.")
                return False

            # Simple GDAL-like progress indicator (one marker per percent).
            block_counter = 0
            last_percent = 0
            print("  Progress: 0", end='', flush=True)
            # Get nodata values for later.
            false_nodata = false_src.nodata
            true_nodata = true_src.nodata

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
                false_window = false_src.read(window=window)  # shape: (3, h, w) for B5, B8, B12
                true_window = true_src.read(window=window)    # shape: (3, h, w) for B2, B3, B4

                # Combine into a 6-band array.
                combined = np.concatenate((false_window, true_window), axis=0)
                h, w = combined.shape[1], combined.shape[2]
                features = combined.reshape(6, -1).T  # shape: (h*w, 6)

                # Get class probabilities.
                probabilities = rf.predict_proba(features)  # shape: (num_pixels, n_classes)
                # Extract the probability for "Rocky reef" (using its index).
                rocky_probs = probabilities[:, rocky_idx]  # 1D array

                # Create a nodata mask (if a pixel is nodata in either source, mark it).
                if false_nodata is not None:
                    mask_false = np.all(false_window == false_nodata, axis=0)
                else:
                    mask_false = np.zeros((h, w), dtype=bool)
                if true_nodata is not None:
                    mask_true = np.all(true_window == true_nodata, axis=0)
                else:
                    mask_true = np.zeros((h, w), dtype=bool)
                nodata_mask = mask_false | mask_true

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
      - Apply a 3-pixel median filter using OpenCV.
      - Apply a morphological closing: dilation (with a round kernel of 3x3) then erosion (same kernel) to fill holes.
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

    # Apply a 3x3 median filter.
    img_median = cv2.medianBlur(img_norm, 5)
    
    # Morphological closing: dilate then erode with elliptical kernels.
    img_closed = cv2.dilate(img_median, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    img_closed = cv2.erode(img_closed, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Double the resolution using bilinear interpolation.
    height, width = img_closed.shape
    img_resized = cv2.resize(img_closed, (width*2, height*2), interpolation=cv2.INTER_LINEAR)

    # Apply threshold of 140 to create a binary mask.
    ret, img_thresh = cv2.threshold(img_resized, 140, 255, cv2.THRESH_BINARY)

    # Update the affine transform since resolution was doubled.
    new_transform = rasterio.Affine(transform.a / 2, transform.b, transform.c,
                                    transform.d, transform.e / 2, transform.f)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Rasterizing land mask and applying raster space clipping")

    # Rasterize the land geometries to the same grid as img_thresh.
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
    reef_gdf_metric = reef_gdf_metric[reef_gdf_metric.area >= 900]

    if reef_gdf_metric.empty:
        print("  No rocky reef polygons remain after area filtering.")
        return False

    # Optionally, reproject back to EPSG:4326 if needed.
    reef_gdf = reef_gdf_metric.to_crs("EPSG:4326")

    # Save the resulting polygons as a shapefile.
    reef_gdf.to_file(shapefile_path)
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Shapefile saved to {shapefile_path}")
    return True


    
def postprocess_and_polygonize_dep(rocky_tif_path, shapefile_path, land_gdf):
    """
    Postprocess the rocky reef probability GeoTIFF:
      - Clip non-zero pixel values to [50, 130] and linearly re-scale that range to [1,255].
      - Apply a 3-pixel median filter using OpenCV.
      - Apply a morphological closing: dilation (with a round kernel of 3x3) then erosion (same kernel) to fill holes.
      - Double the resolution using bilinear interpolation.
      - Apply a threshold of 140 to create a binary mask.
      - Save the threshold image for debugging purposes.
      - Convert the binary mask to polygons (splitting multi-part geometries and simplifying them).
      - Clip the resulting rocky reef polygons by subtracting the pre-processed land mask.
    The resulting shapefile is saved at shapefile_path.
    """
    import cv2  # Ensure OpenCV is imported here if not globally.
    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    import geopandas as gpd

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

    # Save the threshold image for debugging purposes.
    #debug_path = rocky_tif_path.replace(".tif", "_before-median.png")
    #cv2.imwrite(debug_path, img_norm)
    #print(f"  Debug threshold image saved to {debug_path}")
    
    # Apply a 3x3 median filter.
    img_median = cv2.medianBlur(img_norm, 5)
    
    # Save the threshold image for debugging purposes.
    #debug_path = rocky_tif_path.replace(".tif", "_after-median.png")
    #cv2.imwrite(debug_path, img_median)
    #print(f"  Debug threshold image saved to {debug_path}")

    # Create a round (elliptical) kernel
    # Apply morphological closing: dilate one pixel then erode one pixel.
    img_closed = cv2.dilate(img_median, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    # Expand by 1 pixel
    img_closed = cv2.erode(img_closed, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Double the resolution using bilinear interpolation.
    height, width = img_closed.shape
    img_resized = cv2.resize(img_closed, (width*2, height*2), interpolation=cv2.INTER_LINEAR)

    # Apply threshold of 140 to create a binary mask.
    ret, img_thresh = cv2.threshold(img_resized, 140, 255, cv2.THRESH_BINARY)

    # Save the threshold image for debugging purposes.
    #debug_path = rocky_tif_path.replace(".tif", "_debug.png")
    #cv2.imwrite(debug_path, img_thresh)
    #print(f"  Debug threshold image saved to {debug_path}")

    # Update the affine transform since resolution was doubled.
    new_transform = rasterio.Affine(transform.a / 2, transform.b, transform.c,
                                    transform.d, transform.e / 2, transform.f)
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Converting to polygons and simplifying")
    # Extract polygons from the binary image (only for value 255).
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
                
    current_time = datetime.now().strftime("%H:%M:%S")
    if not polys:
        print("  No rocky reef polygons extracted before land clipping.")
        return False
    
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Clipping polygons")
    
    # Create GeoDataFrame for all reef polygons at once
    reef_gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")
    
    print(f"  Performing difference")
    # Use spatial indexing by calling geopandas.overlay for efficient clipping
    reef_clipped_gdf = gpd.overlay(reef_gdf, land_gdf, how='difference')

    # Check if any polygons remain after clipping
    if reef_clipped_gdf.empty:
        print("No rocky reef polygons remain after land clipping.")
        return False

    # Save the resulting polygons as shapefile efficiently
    reef_clipped_gdf['tile_id'] = os.path.basename(shapefile_path)
    reef_clipped_gdf.to_file(shapefile_path)
    print(f"  Shapefile saved to {shapefile_path}")
    return True




def process_tile_tile(tile_file, false_dir, true_files, geotiff_dir, shapefile_dir, rf, rocky_idx, land_gdf):
    """
    Process a single tile: if the output shapefile exists, skip.
    Otherwise, run the RF prediction (if the prediction GeoTIFF doesn't exist) and then
    postprocess it to create a shapefile (clipped with the provided land mask).
    """
    tile_id = extract_tile_id(tile_file)
    shapefile_path = os.path.join(shapefile_dir, f"RockyReef_{tile_id}.shp")
    if os.path.exists(shapefile_path):
        print(f"Tile {tile_id}: shapefile already exists. Skipping tile.")
        return

    # Determine input paths.
    false_path = os.path.join(false_dir, tile_file)
    if tile_id not in true_files:
        print(f"Tile {tile_id}: No matching true-colour image found. Skipping tile.")
        return
    true_path = true_files[tile_id]
    rocky_tif_path = os.path.join(geotiff_dir, f"Classified-{tile_id}.tif")

    # Run RF prediction if the prediction GeoTIFF doesn't exist.
    if not os.path.exists(rocky_tif_path):
        success = run_rf_prediction(tile_id, false_path, true_path, rocky_tif_path, rf, rocky_idx)
        if not success:
            print(f"Tile {tile_id}: RF prediction failed. Skipping tile.")
            return
    else:
        print(f"Tile {tile_id}: Prediction GeoTIFF already exists, skipping RF prediction.")

    # Postprocess the prediction GeoTIFF and polygonize, clipping with the land mask.
    success = postprocess_and_polygonize(rocky_tif_path, shapefile_path, land_gdf)
    if not success:
        print(f"Tile {tile_id}: Post-processing failed or no polygons extracted.")
    else:
        print(f"Tile {tile_id}: Processing complete.")


def main():

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
        default=r'D:\AU_AIMS_MARB-S2-comp_p15\AU_NESP-MaC-3-17_AIMS_Shallow-mask\data\in-3p\AU_AIMS_S2-comp'
    )
    parser.add_argument(
        '--geotiff-dir',
        type=str,
        help="Temporary working folder for prediction images",
        default=r'C:\Temp\gis\AU_NESP-MaC-3-17_AIMS_Rocky-reefs'
    )
    parser.add_argument(
        '--cached-landmask',
        type=str,
        help="Path to the cached (processed) land mask shapefile",
        default=r'working/05/Coastline-50k_trimmed.shp'
    )
    args = parser.parse_args()

    # Process priority tiles if provided.
    priority_tiles = []
    if args.priority_tiles:
        priority_tiles = [pt.strip() for pt in args.priority_tiles.split(',') if pt.strip()]

    # Define directories using the provided command line arguments.
    dataset_path = args.dataset_path
    false_dir = os.path.join(dataset_path, 'low_tide_infrared', 'NorthernAU')
    true_dir = os.path.join(dataset_path, 'low_tide_true_colour', 'NorthernAU')

    geotiff_dir = args.geotiff_dir
    os.makedirs(geotiff_dir, exist_ok=True)
    
    shapefile_dir = r'working/06'
    os.makedirs(shapefile_dir, exist_ok=True)

    # Load the cached land mask.
    cached_landmask_path = args.cached_landmask
    if os.path.exists(cached_landmask_path):
        print("Loading cached adjusted land mask...")
        cached_gdf = gpd.read_file(cached_landmask_path)
        # Assume the cached file contains a single geometry.
        land_mask_geom = cached_gdf.geometry.unary_union
    else:
        print(f"Error: Cached land mask not found at {cached_landmask_path}.")
        print("Please run the land mask processing script first.")
        sys.exit(1)

    print("Converting cached land mask to GeoDataFrame...")
    land_gdf = gpd.GeoDataFrame(geometry=[land_mask_geom], crs="EPSG:4326")

    # Build a dictionary for true-colour files.
    true_files = {}
    for file in os.listdir(true_dir):
        if file.lower().endswith('.tif'):
            tile_id = extract_tile_id(file)
            true_files[tile_id] = os.path.join(true_dir, file)

    # List all false-colour files.
    false_files = [f for f in os.listdir(false_dir) if f.lower().endswith('.tif')]

    # Reorder false_files so that any tile in the priority list is processed first.
    priority_false_files = [f for f in false_files if extract_tile_id(f) in priority_tiles]
    non_priority_false_files = [f for f in false_files if extract_tile_id(f) not in priority_tiles]
    ordered_false_files = priority_false_files + non_priority_false_files

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

    total_files = len(ordered_false_files)
    print(f"Found {total_files} false-colour files to process.")
    processed_count = 0

    for i, tile_file in enumerate(ordered_false_files, 1):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{current_time}] Processing tile {extract_tile_id(tile_file)} ({i}/{total_files})...")
        process_tile_tile(tile_file, false_dir, true_files, geotiff_dir, shapefile_dir, rf, rocky_idx, land_gdf)
        processed_count += 1

    print(f"\nProcessing complete. {processed_count} tiles processed.")

if __name__ == '__main__':
    main()



