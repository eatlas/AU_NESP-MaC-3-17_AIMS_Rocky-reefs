"""
Image Feature Processing Script

Purpose:
    This script processes low tide true color satellite imagery with different methods
    and saves the results to a specified output directory.

Key Operations:
    1. Reads low tide true color Sentinel-2 imagery
    2. Applies one of the following processes:
       - Gaussian blur (7x7)
       - Local standard deviation calculation
       - Median-filter texture difference
    3. Saves the processed images to derived/[output-folder]/[region]
    4. Supports parallel processing
    
Usage:
    python 01d-image-features.py [options]
    
    Options:
        --imagery-path    : Path to the satellite image dataset
        --regions         : Comma-separated list of regions to process (NorthernAU,GBR)
        --parallel        : Number of parallel processes to use (default: 1)
        --process-type    : Type of processing to apply (blur, stddev, texture_diff)

Example:
    python 01d-image-features.py --imagery-path ~/Documents/2025/GIS/AU_AIMS_S2-comp --regions NorthernAU --process-type texture_diff --parallel 4
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import cv2
import rasterio
import concurrent.futures
import re
import math

# Folder name constants
LOW_TIDE_TRUE_COLOUR_FOLDER_NAME = 'low_tide_true_colour'
# Dictionary mapping process types to their output folder names
PROCESS_TYPE_FOLDERS = {
    'blur': 'low-tide-true-colour-blur7',
    'stddev': 'low-tide-true-colour-stddev',
    'texture_diff': 'low-tide-true-colour-texture-diff'
}

def nowstr():
    """
    Returns the current time as a formatted string for printing progress.
    """
    return datetime.now().strftime("%H:%M:%S")

def create_clipping_mask_cv2(sat_image, trim_pixels):
    """
    This function creates a clipping mask from the black borders of the Sentinel 2
    images. This corresponds to NoData values of 0 around edges of the image corresponding
    to the slanted rectangle of the Sentinel 2 tile inside the image. 

    The original border is expanded using erosion by trim_pixels.
    
    Args:
        sat_image (numpy.ndarray): Original Sentinel 2 Geotiff image loaded in
        trim_pixels (int): Number of pixels to trim off
    Returns:
        clipping mask at native resolution (numpy.ndarray).
    """
    if trim_pixels < 0:
        mask = (sat_image[0] != 0).astype(np.uint8)
        return mask
    # Process only the first band as the border is the same in all three 
    first_band = sat_image[0]

    # Define border width
    border_width = 1
    
    # Expand the image by padding with a 1 border. This guarantees that
    # there is a boarder of 1 pixel around the image to erode.
    padded_sat_image = np.pad(first_band, pad_width=border_width, mode='constant', 
                              constant_values=0)
    
    mask = (padded_sat_image != 0).astype(np.uint8)  # NoData value is 0
    
    # save memory
    del padded_sat_image, first_band

    # Perform final erosion based on the trim_pixels
    final_structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
        (trim_pixels+border_width, trim_pixels+border_width))
    mask = cv2.erode(mask, final_structure, iterations=1)
    
    # Trim back to original size (remove padding)
    mask = mask[border_width:-border_width, border_width:-border_width]

    return mask


def apply_clipping_mask_u8b3(image, clipping_mask):
    """
    Apply a clipping mask to a uint 8, 3 band image. The mask is a 2D array, where 0
    indicates the pixels to be masked out. The function sets the masked pixels in the
    image to 0 (black). 0 Is assumed to be the NoData value for the image.

    Args:
        image (numpy.ndarray): The image to be masked, should have shape (channels, height, width).
        clipping_mask (numpy.ndarray): The clipping mask, should be 2D (height, width).
    Returns:
        numpy.ndarray: The masked image, with the same shape as the input image.
    """
    # Assertion to ensure image and clipping_mask have compatible shapes
    assert image.shape[1:] == clipping_mask.shape, \
        f"image and clipping_mask must have compatible shapes. Got image[1:] = {image.shape[1:]} and clipping_mask = {clipping_mask.shape}."
    
    assert image.dtype == np.uint8, \
        f"Expecting image to be uint8 but found {image.dtype}"
    
    assert clipping_mask.dtype == np.uint8, \
        f"Expecting clipping_mask to be uint8 but found {clipping_mask.dtype}"
    
    assert clipping_mask.ndim == 2, \
        f"Expecting clipping mask shape height, width, got {clipping_mask.shape}"
    
    assert image.ndim == 3, \
        f"Expecting image shape channels, height, width, got {image.shape}"
    
    # Apply the mask to each channel
    # For images with shape (channels, height, width)
    for i in range(image.shape[0]):
        # Where clipping_mask is 0, set the pixel to 0
        image[i][clipping_mask == 0] = 0
    
    return image


def extract_tile_id(filename):
    """
    Extract the tile ID from various filename patterns using regex.
    
    Handles different formats:
    - All tide: AU_AIMS_MARB-S2-comp_p15_TrueColour_56KPU_v2_2015-2024.tif
    - Low tide: AU_AIMS_MARB-S2-comp_low-tide_p30_TrueColour_56KPU.tif
    
    Looks for a pattern of underscore followed by two digits and three uppercase letters
    (e.g., "_56KPU") and returns the 5-character tile ID (e.g., "56KPU").
    """
    basename = os.path.splitext(filename)[0]
    match = re.search(r'_(\d{2}[A-Z]{3})', basename)
    
    if match:
        tile_id = match.group(1)  # Extract the captured group (without underscore)
        return tile_id
    else:
        raise ValueError(f"Invalid tile ID format in filename '{filename}'. "
                         f"Expected a pattern of underscore followed by two digits and three uppercase letters.")

def collect_tiles_for_regions(regions, dataset_path):
    """
    Collects and returns information about tiles to be processed across multiple regions.
    """
    # Initialize empty dictionaries to store files from all regions
    lt_true_files = {}
    tile_regions = {}
    
    def collect_geotiff_files(directory, region):
        """Helper function to collect GeoTIFF files from a directory"""
        geotiff_files = {}
        if not os.path.isdir(directory):
            print(f"Warning: Directory {directory} does not exist")
            return geotiff_files
            
        for file in os.listdir(directory):
            if file.lower().endswith('.tif'):
                try:
                    tile_id = extract_tile_id(file)
                    geotiff_files[tile_id] = os.path.join(directory, file)
                    # Only record the region if we haven't seen this tile before
                    if tile_id not in tile_regions:
                        tile_regions[tile_id] = region
                except ValueError as e:
                    print(f"Warning: {e}")
        return geotiff_files
    
    # Process each region and collect files
    for region in regions:
        lt_true_dir = os.path.join(dataset_path, LOW_TIDE_TRUE_COLOUR_FOLDER_NAME, region)
        
        # Collect files for this region
        region_lt_true = collect_geotiff_files(lt_true_dir, region)
        
        # Merge with the main dictionaries
        lt_true_files.update(region_lt_true)
    
    # Get all tile IDs
    all_tile_ids = list(lt_true_files.keys())
    
    # Create ordered arrays
    ordered_lt_true_files = [lt_true_files[tile_id] for tile_id in all_tile_ids]
    ordered_tile_regions = [tile_regions.get(tile_id, "unknown") for tile_id in all_tile_ids]
    
    return (all_tile_ids, ordered_lt_true_files, ordered_tile_regions)


def process_image_with_texture_diff(input_path, output_path, tile_id):
    """
    Process an image using a median filter difference approach:
    1. Apply median filter to original image
    2. Calculate difference between original and median-filtered image
    3. Apply Gaussian blur to smooth the differences
    4. Apply another median filter to further smooth the differences
    5. Save the result with LZW compression
    """
    MEDIAN_KERNEL_SIZE = 13
    GAUSSIAN_KERNEL_SIZE = 3  # New parameter for Gaussian blur
    GAUSSIAN_SIGMA = 0        # 0 means auto-calculate based on kernel size
    SMOOTHING_KERNEL_SIZE = 13
    try:
        with rasterio.open(input_path) as src:
            # Read the image data
            img = src.read()
            profile = src.profile.copy()
            # Set LZW compression for output file
            profile.update(compress='LZW')

            # Process each band to find texture variations
            texture_img = np.zeros_like(img, dtype=np.uint8)
            for i in range(img.shape[0]):
                # Convert to float for calculations
                img_float = img[i].astype(np.float32)
                
                # Step 1: Apply median filter
                median_filtered = cv2.medianBlur(img_float.astype(np.uint8), MEDIAN_KERNEL_SIZE).astype(np.float32)
                
                # Step 2: Calculate absolute difference between original and median-filtered
                diff = np.abs(img_float - median_filtered)
                
                # Step 3: Apply Gaussian blur to smooth the differences
                diff_uint8 = np.clip(diff, 0, 255).astype(np.uint8)
                gaussian_smoothed = cv2.GaussianBlur(diff_uint8, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), GAUSSIAN_SIGMA)
                
                # Step 4: Apply another median filter to further smooth the differences
                smoothed_diff = cv2.medianBlur(gaussian_smoothed, SMOOTHING_KERNEL_SIZE).astype(np.float32)
                
                # Scale to fit in uint8 range (adjust multiplier based on your data)
                # Add 1 to avoid zero values as these would be NoData
                scaled_diff = np.clip(smoothed_diff * 6.0 + 1, 1, 255).astype(np.uint8)
                texture_img[i] = scaled_diff

            # Create and apply mask
            mask = create_clipping_mask_cv2(img, MEDIAN_KERNEL_SIZE)
            texture_img = apply_clipping_mask_u8b3(texture_img, mask)
            
            # Write the texture difference image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(texture_img)
            
            print(f"[{nowstr()}] Tile {tile_id}: Texture difference calculated and saved with LZW compression")
            return True
    except Exception as e:
        print(f"[{nowstr()}] Error processing tile {tile_id}: {str(e)}")
        return False

def process_image_with_stddev(input_path, output_path, tile_id):
    """
    Process an image by calculating local standard deviation within a kernel and save with LZW compression.
    """
    KERNEL_SIZE = 7
    try:
        with rasterio.open(input_path) as src:
            # Read the image data
            img = src.read()
            profile = src.profile.copy()
            # Set LZW compression for output file
            profile.update(compress='LZW')

            # Calculate standard deviation for each band
            stddev_img = np.zeros_like(img, dtype=np.uint8)
            for i in range(img.shape[0]):
                # Convert to float for calculations
                img_float = img[i].astype(np.float32)
                
                # Calculate mean using a box filter
                mean = cv2.blur(img_float, (KERNEL_SIZE, KERNEL_SIZE))
                
                # Calculate mean of squared values
                mean_sq = cv2.blur(img_float**2, (KERNEL_SIZE, KERNEL_SIZE))
                
                # Calculate standard deviation
                variance = mean_sq - mean**2
                # Avoid negative values due to numerical precision
                variance = np.maximum(variance, 0)
                stddev = np.sqrt(variance)
                
                # Scale to fit in uint8 range (multiply by 2 since most stddev values will be < 127.5)
                # Add 1 to avoid zero values as these are NoData
                stddev_scaled = np.clip(stddev * 4.0 + 1, 1, 255).astype(np.uint8)
                stddev_img[i] = stddev_scaled

            # Create and apply mask
            mask = create_clipping_mask_cv2(img, KERNEL_SIZE)
            stddev_img = apply_clipping_mask_u8b3(stddev_img, mask)
            
            # Write the standard deviation image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(stddev_img)
            
            print(f"[{nowstr()}] Tile {tile_id}: Local standard deviation calculated and saved with LZW compression")
            return True
    except Exception as e:
        print(f"[{nowstr()}] Error processing tile {tile_id}: {str(e)}")
        return False

def process_image_with_blur(input_path, output_path, tile_id):
    """
    Process an image with a Gaussian blur and save the result with LZW compression.
    """
    BLUR = 7
    try:
        with rasterio.open(input_path) as src:
            # Read the image data
            img = src.read()
            profile = src.profile.copy()
            # Set LZW compression for output file
            profile.update(compress='LZW')

            # Apply Gaussian blur to each band
            blurred_img = np.zeros_like(img)
            for i in range(img.shape[0]):
                # Apply Gaussian blur with kernel size 7x7
                blurred_img[i] = cv2.GaussianBlur(img[i], (7, 7), 0)
            mask = create_clipping_mask_cv2(img, BLUR)
            #mask = create_clipping_mask_cv2(img, 1)
            # Apply the mask to the blurred image
            blurred_img = apply_clipping_mask_u8b3(blurred_img, mask)
            # Write the blurred image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(blurred_img)
            
            print(f"[{nowstr()}] Tile {tile_id}: Gaussian blur applied and saved with LZW compression")
            return True
    except Exception as e:
        print(f"[{nowstr()}] Error processing tile {tile_id}: {str(e)}")
        return False

def process_worker(args):
    """
    Worker function for parallel processing.
    """
    i, tile_id, input_file, region, dataset_path, total_files, process_type, output_folder = args
    
    output_dir = os.path.join(dataset_path, output_folder, region)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the original filename and construct the output path
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, filename)
    
    print(f"[{nowstr()}] Processing tile {tile_id} ({i+1}/{total_files}) - region: {region}, method: {process_type}")
    
    if process_type == 'blur':
        success = process_image_with_blur(input_file, output_file, tile_id)
    elif process_type == 'stddev':
        success = process_image_with_stddev(input_file, output_file, tile_id)
    elif process_type == 'texture_diff':
        success = process_image_with_texture_diff(input_file, output_file, tile_id)
    else:
        print(f"[{nowstr()}] Error: Unknown process type '{process_type}'")
        success = False
    
    return 1 if success else 0

def main():
    parser = argparse.ArgumentParser(
        description="Process satellite imagery with different methods."
    )
    parser.add_argument(
        '--imagery-path',
        type=str,
        help="Path to the satellite image dataset",
        default=r'D:\AU_AIMS_S2-comp'
    )
    parser.add_argument(
        '--regions',
        type=str,
        help="Comma-separated list of regions to process (NorthernAU,GBR)",
        default="NorthernAU,GBR"
    )
    parser.add_argument(
        '--parallel',
        type=int,
        help="Number of parallel processes to use (default: 1 for sequential processing)",
        default=1
    )
    parser.add_argument(
        '--process-type',
        type=str,
        choices=['blur', 'stddev', 'texture_diff'],
        help="Type of processing to apply (blur, stddev, texture_diff)",
        default='stddev'
    )
    
    args = parser.parse_args()
    
    # Get regions to process
    regions = [region.strip() for region in args.regions.split(',') if region.strip()]
    if not regions:
        regions = ["NorthernAU", "GBR"]
    
    dataset_path = args.imagery_path
    process_type = args.process_type
    
    # Set output folder based on process type
    output_folder = PROCESS_TYPE_FOLDERS.get(process_type)
    if not output_folder:
        print(f"Error: Unknown process type '{process_type}'")
        return 1
    
    print(f"Processing type: {process_type}, output folder: {output_folder}")
    
    # Collect tiles for processing
    tile_ids, lt_true_files, tile_regions = collect_tiles_for_regions(regions, dataset_path)
    
    total_files = len(tile_ids)
    if total_files == 0:
        print(f"No processable tiles found.")
        return 0
    
    print(f"Found {total_files} tiles across {len(regions)} regions.")
    
    # Process the tiles
    num_workers = args.parallel
    
    if num_workers > 1:
        print(f"Processing {total_files} tiles in parallel using {num_workers} workers")
        
        # Prepare task arguments for each tile
        worker_args = []
        for i in range(total_files):
            worker_args.append((
                i,                  # Index
                tile_ids[i],        # Tile ID
                lt_true_files[i],   # Path to input file
                tile_regions[i],    # Region name
                dataset_path,       # Dataset path
                total_files,        # Total files count
                process_type,       # Process type
                output_folder       # Output folder name
            ))
        
        # Process tiles in parallel
        processed_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_worker, worker_args))
            processed_count = sum(results)  # Count successful completions
        
        print(f"\nProcessing complete. Successfully processed {processed_count} of {total_files} tiles.")
    
    else:
        # Sequential processing
        processed_count = 0
        for i, tile_id in enumerate(tile_ids):
            args_tuple = (i, tile_id, lt_true_files[i], tile_regions[i], dataset_path, total_files, 
                         process_type, output_folder)
            if process_worker(args_tuple):
                processed_count += 1
        
        print(f"\nProcessing complete. Successfully processed {processed_count} of {total_files} tiles.")

if __name__ == '__main__':
    main()