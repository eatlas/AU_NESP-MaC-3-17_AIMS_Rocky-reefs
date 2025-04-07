import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
import os
import numpy as np # Import numpy for handling nodata potentially

print("Script started...")

# --- Configuration ---

# --- Input file paths ---
# *** UPDATED: Define separate base path for imagery on external drive ***
# Use a raw string (r'...') for Windows paths to avoid issues with backslashes
imagery_base_path = Path(r'D:\AU_AIMS_MARB-S2-comp_p15\AU_NESP-MaC-3-17_AIMS_Shallow-mask\data\in-3p\AU_AIMS_S2-comp')

# *** UPDATED: Training data path relative to the script location ***
training_shp_path = Path('data/in/Training-data.shp')

# VRT paths using the imagery base path
false_color_vrt_path = imagery_base_path / 'low_tide_infrared' / 'NorthernAU.vrt'
true_color_vrt_path = imagery_base_path / 'low_tide_true_colour' / 'NorthernAU.vrt'

# --- Output file path (relative to the script location) ---
output_dir = Path('working/training-data')
output_csv_path = output_dir / 'training-data-with-pixels.csv'

# Expected CRS for the VRT rasters (as previously stated by user)
# Will be validated against the actual VRT CRS read by rasterio
expected_raster_crs_str = "EPSG:4326" # Use this for comparison

# --- Preparations ---

# Create output directory if it doesn't exist
print(f"Ensuring output directory exists: {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load Training Data ---

print(f"Loading training data from Shapefile: {training_shp_path}")
# Check if the relative path exists from the current working directory
if not training_shp_path.exists():
     print(f"Error: Training Shapefile not found at relative path '{training_shp_path}'.")
     print(f"Please ensure the script is run from a directory containing the '{training_shp_path.parent.parent}' folder,")
     print(f"or provide an absolute path.")
     exit()

try:
    gdf = gpd.read_file(training_shp_path)
    print(f"Loaded {len(gdf)} training points.")
    print(f"Shapefile CRS detected: {gdf.crs}")

    # Basic validation
    required_cols = ['FeatType', 'geometry'] # Check for geometry implicitly via GeoDataFrame
    if 'FeatType' not in gdf.columns:
        raise ValueError("Shapefile must contain 'FeatType' attribute column.")
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry.isnull().any():
         raise ValueError("Shapefile did not load correctly as a GeoDataFrame with valid geometries.")
    if not gdf.geom_type.isin(['Point']).all():
         print("Warning: Shapefile contains non-point geometries. Only points will be processed correctly.")

except Exception as e:
    print(f"Error reading training data Shapefile: {e}")
    exit()

# --- Extract Lat/Lon for Output CSV ---
print(f"Extracting Lat/Lon columns from geometry (using original CRS: {gdf.crs})...")
try:
    gdf['Long'] = gdf.geometry.x
    gdf['Lat'] = gdf.geometry.y
    print(" 'Lat' and 'Long' columns added.")
except Exception as e:
    print(f"Could not extract Lat/Lon from geometry: {e}")
    if 'Lat' not in gdf.columns: gdf['Lat'] = np.nan
    if 'Long' not in gdf.columns: gdf['Long'] = np.nan

# --- Define Function to Extract Pixels (modified for clarity) ---

def extract_pixel_values(vrt_path, coords_for_sampling, band_names):
    """
    Opens a VRT and extracts pixel values for given coordinates.
    Assumes coords_for_sampling are ALREADY in the VRT's CRS.

    Args:
        vrt_path (Path): Path to the Virtual Raster Table (.vrt).
        coords_for_sampling (list): A list of (x, y) tuples in the VRT's CRS.
        band_names (list): List of strings for naming output columns.

    Returns:
        tuple: (pandas.DataFrame containing extracted values, rasterio.crs object)
               Returns (None, None) on error.
    """
    extracted_data = {name: [] for name in band_names}
    print(f"Processing VRT: {vrt_path.name}")
    raster_crs = None
    # *** Check if VRT exists before trying to open ***
    if not vrt_path.exists():
        print(f"  Error: VRT file not found at {vrt_path}")
        return None, None

    try:
        with rasterio.open(vrt_path) as src:
            raster_crs = src.crs # Store the actual raster CRS
            print(f"  Opened VRT. CRS: {raster_crs}, Band Count: {src.count}")

            if src.count != len(band_names):
                print(f"  Error: VRT has {src.count} bands, but {len(band_names)} band names were expected.")
                return None, None # Indicate failure

            print(f"  Sampling {len(coords_for_sampling)} points...")
            pixel_generator = src.sample(coords_for_sampling)

            # Iterate through the generator and store results
            nodata_count = 0
            value_count = 0
            src_nodata = src.nodatavals # Get nodata values for each band

            for i, pixel_values in enumerate(pixel_generator):
                # Check if the entire sample matches nodata for respective bands
                is_nodata = False
                if src_nodata:
                    if len(pixel_values) == len(src_nodata):
                        # Check if all returned values match the nodata value for their respective band
                        is_nodata = all( (pv == nd or (np.isnan(pv) and np.isnan(nd)) ) # Handle NaN nodata values
                                         for pv, nd in zip(pixel_values, src_nodata) if nd is not None)

                if is_nodata:
                     nodata_count += 1
                     for band_name in band_names:
                          extracted_data[band_name].append(np.nan)
                elif pixel_values.size == len(band_names):
                    value_count += 1
                    for band_idx, band_name in enumerate(band_names):
                        extracted_data[band_name].append(pixel_values[band_idx])
                else:
                     print(f"  Warning: Unexpected number of values ({pixel_values.size}) returned for point index {i}. Expected {len(band_names)}. Filling with NaN.")
                     nodata_count +=1
                     for band_name in band_names:
                         extracted_data[band_name].append(np.nan)

            print(f"  Finished sampling {vrt_path.name}. Found {value_count} valid points, {nodata_count} nodata/error points.")

        # Convert the extracted data dictionary to a DataFrame
        extracted_df = pd.DataFrame(extracted_data)
        return extracted_df, raster_crs

    except rasterio.RasterioIOError as e:
        print(f"  Error opening or reading VRT {vrt_path}: {e}")
        return None, None
    except Exception as e:
        print(f"  An unexpected error occurred while processing {vrt_path}: {e}")
        return None, None


# --- Prepare Coordinates and Handle CRS ---

# We need coordinates in the CRS of the *rasters* for sampling.
# Let's open one VRT first to get the definitive raster CRS.
print("Checking raster CRS...")
raster_crs = None
# *** Check VRT existence before trying to open ***
if not false_color_vrt_path.exists():
    print(f"Error: Cannot determine raster CRS because VRT file not found at {false_color_vrt_path}")
    exit()
try:
    with rasterio.open(false_color_vrt_path) as src:
        raster_crs = src.crs
        print(f"Detected Raster CRS: {raster_crs}")
except Exception as e:
    print(f"Could not open {false_color_vrt_path} to determine raster CRS: {e}")
    exit()

# Compare GeoDataFrame CRS with Raster CRS
coords_for_sampling = []
if gdf.crs == raster_crs:
    print("Point CRS matches Raster CRS. Using original coordinates for sampling.")
    coords_for_sampling = [(pt.x, pt.y) for pt in gdf.geometry]
else:
    print(f"Point CRS ({gdf.crs}) differs from Raster CRS ({raster_crs}). Transforming points...")
    try:
        gdf_transformed = gdf.to_crs(raster_crs)
        coords_for_sampling = [(pt.x, pt.y) for pt in gdf_transformed.geometry]
        print("Transformation successful.")
    except Exception as e:
        print(f"Error transforming points from {gdf.crs} to {raster_crs}: {e}")
        exit()

if not coords_for_sampling:
     print("Error: Coordinate list for sampling is empty.")
     exit()

# --- Extract from False Color VRT ---
fc_band_names = ['S2_B5', 'S2_B8', 'S2_B12']
fc_extracted_df, fc_crs = extract_pixel_values(false_color_vrt_path, coords_for_sampling, fc_band_names)
if fc_extracted_df is not None and fc_crs and fc_crs != raster_crs:
     print(f"Warning: False Color VRT CRS ({fc_crs}) differs from initially detected raster CRS ({raster_crs}). This might indicate inconsistent VRTs.")

# --- Extract from True Color VRT ---
tc_band_names = ['S2_B2', 'S2_B3', 'S2_B4']
tc_extracted_df, tc_crs = extract_pixel_values(true_color_vrt_path, coords_for_sampling, tc_band_names)
if tc_extracted_df is not None and tc_crs and tc_crs != raster_crs:
     print(f"Warning: True Color VRT CRS ({tc_crs}) differs from initially detected raster CRS ({raster_crs}). This might indicate inconsistent VRTs.")


# --- Combine Data ---
if fc_extracted_df is not None and tc_extracted_df is not None:
    print("Combining original data attributes with extracted pixel values...")

    # Select desired columns from the original GeoDataFrame (excluding geometry)
    output_columns = ['FeatType']
    if 'Lat' in gdf.columns: output_columns.append('Lat')
    if 'Long' in gdf.columns: output_columns.append('Long')

    # Create the base dataframe for the output CSV
    final_df_base = gdf[output_columns].copy()

    # Ensure the extracted dataframes have the same index as the original gdf
    fc_extracted_df.index = final_df_base.index
    tc_extracted_df.index = final_df_base.index

    # Concatenate base dataframe with the new pixel value dataframes
    final_df = pd.concat([final_df_base, fc_extracted_df, tc_extracted_df], axis=1)

    # Reorder columns for clarity (optional)
    cols_order = output_columns + fc_band_names + tc_band_names
    # Ensure all columns actually exist before trying to reorder
    cols_order = [col for col in cols_order if col in final_df.columns]
    final_df = final_df[cols_order]


    # --- Save Results ---
    print(f"Saving combined data to: {output_csv_path}")
    try:
        # Drop rows where *all* extracted pixel values are NaN (optional)
        all_band_cols = fc_band_names + tc_band_names
        # Filter band cols that actually exist in the dataframe
        all_band_cols = [col for col in all_band_cols if col in final_df.columns]
        if all_band_cols: # Only drop if there are band columns to check
            rows_before = len(final_df)
            final_df.dropna(subset=all_band_cols, how='all', inplace=True)
            rows_after = len(final_df)
            if rows_before > rows_after:
                print(f"Dropped {rows_before - rows_after} rows where all sampled band values were NoData/NaN.")
        else:
             print("Skipping dropna step as no valid band columns were extracted.")

        final_df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved {len(final_df)} records to {output_csv_path}.")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
else:
    print("Skipping final combination and saving due to errors during pixel extraction.")

print("Script finished.")