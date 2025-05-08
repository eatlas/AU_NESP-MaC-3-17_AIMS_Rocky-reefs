"""
This script download imagery from the 15th percentile (All tide) and low tide image datasets (Hammerton & Lawrey, 2024a; 2024b). 

These image datasets are large (203 GB) making downloading the datasets challenging. This
script progressively downloads each of the images. If the script is interrupted then it will
pick up from where it left off.

This script allows subsets of the datasets to be downloaded. This can be a custom region of Sentinel 2
tiles or named regions, such as NorthernAU that corresponds to the study area for the NESP MaC 3.17 project,
and GBR for the Great Barrier Reef and Torres Strait.

Download all image styles across all regions: Note these can be run in parallel or in serial. The low_tide_infrared
is not used in the Shallow mask mapping.

To reproduce this dataset:
Note: The low_tide_infrared imagery is not yet used in version 1-1 of this dataset.
python 01-download-sentinel2.py --dataset 15th_percentile --region NorthernAU --output ~/AU_AIMS_S2-comp
python 01-download-sentinel2.py --dataset low_tide_true_colour --region NorthernAU --output ~/AU_AIMS_S2-comp
python 01-download-sentinel2.py --dataset low_tide_infrared --region NorthernAU --output ~/AU_AIMS_S2-comp
python 01-download-sentinel2.py --dataset 15th_percentile --region GBR --output ~/AU_AIMS_S2-comp
python 01-download-sentinel2.py --dataset low_tide_true_colour --region GBR --output ~/AU_AIMS_S2-comp
python 01-download-sentinel2.py --dataset low_tide_infrared --region GBR --output ~/AU_AIMS_S2-comp

Single Tile Demo Run: (Takes approximately 1 min per tile)
python 01-download-sentinel2.py --dataset 15th_percentile --region GBR --tiles 55KDA
python 01-download-sentinel2.py --dataset low_tide_true_colour --region GBR --tiles 55KDA
The file ID can be determine from the following map:
https://maps.eatlas.org.au/index.html?intro=false&z=7&ll=148.00000,-18.00000&l0=ea_ref%3AWorld_ESA_Sentinel-2-tiling-grid_Poly,ea_ea-be%3AWorld_Bright-Earth-e-Atlas-basemap
We need to specify the region in this call so that subsequent script know where to find the imagery.
Note: The region needs to match tiles otherwise an error will be generated.


References:

Hammerton, M., & Lawrey, E. (2024a). North Australia Sentinel 2 Satellite Composite Imagery - 15th percentile true colour (NESP MaC 3.17, AIMS) (2nd Ed.) [Data set]. eAtlas. https://doi.org/10.26274/HD2Z-KM55

Hammerton, M., & Lawrey, E. (2024b). Tropical Australia Sentinel 2 Satellite Composite Imagery - Low Tide - 30th percentile true colour and near infrared false colour (NESP MaC 3.17, AIMS) (1st Ed.) [Data set]. eAtlas. https://doi.org/10.26274/2bfv-e921

Errors seen in the wild:
    <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1129)> - Retries don't help
    [SSL] unknown error (_ssl.c:3161)
    OSError: [WinError 1450] Insufficient system resources exist to complete the requested service: '.\\low_tide_true_colour\\GBR'
"""

import urllib.request
import os
import sys
import time
import argparse
import csv
from typing import List, Dict
from http.client import IncompleteRead

class DataDownloader:
    def __init__(self, download_path="", max_retries=5):
        self.start_time = 0
        self.last_report_time = 0
        self.download_path = download_path
        self.max_retries = max_retries
        self.missing_tiles = []

    def _reporthook(self, count, block_size, total_size):
        current_time = time.time()
        if count == 0:
            self.start_time = current_time
            self.last_report_time = current_time
            return
        time_since_last_report = current_time - self.last_report_time
        if time_since_last_report > 1:
            self.last_report_time = current_time
            duration = current_time - self.start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))

            if total_size != -1:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"{percent}%, {progress_size / (1024 * 1024):.2f} MB, {speed} KB/s, {duration:.2f} secs    \r")
            else:
                sys.stdout.write(f"{progress_size / (1024 * 1024):.2f} MB, {speed} KB/s, {duration:.2f} secs    \r")
            sys.stdout.flush()

    def download(self, url: str, path: str, tile_id: str):
        if os.path.exists(path):
            print(f"Skipping download of {os.path.abspath(path)}; it already exists")
        else:
            print(f"Downloading from {url}")
            print(f"Saving to {os.path.abspath(path)}")
            dest_dir = os.path.dirname(path)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            tmp_path = path + '.tmp'
            attempt = 0
            while attempt < self.max_retries:
                try:
                    urllib.request.urlretrieve(url, tmp_path, self._reporthook)
                    os.rename(tmp_path, path)
                    print("\nDownload complete")
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print(f"\nError 404: Not Found for {url}. Skipping...")
                        self.missing_tiles.append(tile_id)
                        break
                    else:
                        print(f"\nHTTP Error downloading {url}: {e}")
                except (urllib.error.URLError, IncompleteRead, ConnectionResetError) as e:
                    attempt += 1
                    print(f"\nError downloading {url} (attempt {attempt} of {self.max_retries}): {e}")
                    if attempt == self.max_retries:
                        print(f"Failed to download {url} after {self.max_retries} attempts.")
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff
                except IOError as e:
                    print(f"\nError writing file {path}: {e}")
                    break
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

def get_tile_sets() -> Dict[str, Dict[str, List[str]]]:
    tile_ids_all = [
        '46LGM', '46LGN', '46LHM', '46LHN', '47LKG', '47LKH', '47LLF', '47LLG', '48LWP',
        '48LXP', '49JFK', '49JFL', '49JFM', '49JFN', '49JGH', '49JGJ', '49JGK', '49JGL',
        '49JGM', '49JGN', '49JHM', '49KFP', '49KFQ', '49KGP', '49KGQ', '49KGR', '49KGS',
        '49KHR', '49KHS', '49KHT', '50JKN', '50JKP', '50JKQ', '50JKR', '50JKS', '50KKB',
        '50KKC', '50KKD', '50KLB', '50KLC', '50KLD', '50KMB', '50KMC', '50KMD', '50KME',
        '50KNC', '50KND', '50KNE', '50KNF', '50KPC', '50KPD', '50KPE', '50KPF', '50KPG',
        '50KQC', '50KQD', '50KQE', '50KQF', '50KQG', '50KRC', '50KRD', '50KRE', '50KRF',
        '50KRG', '50LRH', '51KTA', '51KTB', '51KTU', '51KTV', '51KUA', '51KUB', '51KUU',
        '51KUV', '51KVA', '51KVB', '51KVV', '51KWA', '51KWB', '51KXB', '51LTC', '51LTD',
        '51LTE', '51LUC', '51LUD', '51LUE', '51LUF', '51LUG', '51LVC', '51LVD', '51LVE',
        '51LVF', '51LVG', '51LVH', '51LWC', '51LWD', '51LWE', '51LWF', '51LWG', '51LWH',
        '51LXC', '51LXD', '51LXE', '51LXF', '51LXG', '51LXH', '51LXJ', '51LYC', '51LYD',
        '51LYE', '51LYF', '51LYG', '51LYH', '51LYJ', '51LZD', '51LZE', '51LZF', '51LZG',
        '51LZH', '51LZJ', '52LBK', '52LBL', '52LBM', '52LBN', '52LBP', '52LBQ', '52LCH',
        '52LCJ', '52LCK', '52LCL', '52LCM', '52LCN', '52LCP', '52LCQ', '52LDH', '52LDJ',
        '52LDK', '52LDL', '52LDM', '52LDN', '52LDP', '52LDQ', '52LEJ', '52LEK', '52LEL',
        '52LEM', '52LEN', '52LEP', '52LEQ', '52LFL', '52LFM', '52LFN', '52LFP', '52LFQ',
        '52LGM', '52LGN', '52LGP', '52LGQ', '52LHM', '52LHN', '52LHP', '52LHQ', '53KRB',
        '53LKG', '53LKH', '53LKJ', '53LKK', '53LLG', '53LLH', '53LLJ', '53LLK', '53LLL',
        '53LMG', '53LMH', '53LMJ', '53LMK', '53LML', '53LND', '53LNE', '53LNF', '53LNG',
        '53LNH', '53LNJ', '53LNK', '53LPC', '53LPD', '53LPE', '53LPF', '53LPG', '53LPH',
        '53LPJ', '53LPK', '53LQC', '53LQD', '53LQE', '53LQF', '53LQG', '53LQH', '53LQJ',
        '53LQK', '53LRC', '53LRD', '53LRE', '53LRF', '53LRG', '53LRH', '53LRJ', '53LRK',
        '54KTG', '54KUF', '54KUG', '54KVF', '54KVG', '54KWG', '54LTH', '54LTJ', '54LTK',
        '54LTL', '54LTM', '54LTN', '54LTP', '54LUH', '54LUJ', '54LUK', '54LUL', '54LUM',
        '54LUN', '54LUP', '54LVH', '54LVJ', '54LVK', '54LVL', '54LVM', '54LVN', '54LVP',
        '54LWH', '54LWJ', '54LWK', '54LWL', '54LWM', '54LWN', '54LWP', '54LXM', '54LXN',
        '54LXP', '57HVE', '57HWE', '57JVF', '57JVG', '57JVH', '57JVJ', '57JWF', '57JWG',
        '57JWH', '58JGN', '58JGP', '59JKH', '59JKJ'
        ]
                
    tile_ids_gbr = [
        '54LWQ', '54LXQ', '54LYK', '54LYL', '54LYM', '54LYN', '54LYP', '54LYQ', '54LZK',
        '54LZL', '54LZM', '54LZN', '54LZP', '54LZQ', '55KCA', '55KCB', '55KCV', '55KDA',
        '55KDB', '55KDU', '55KDV', '55KEA', '55KEU', '55KEV', '55KFA', '55KFS', '55KFT',
        '55KFU', '55KFV', '55KGR', '55KGS', '55KGT', '55KGU', '55KGV', '55KHR', '55KHS',
        '55KHT', '55KHU', '55LBD', '55LBE', '55LBF', '55LBG', '55LBH', '55LBJ', '55LBK',
        '55LCC', '55LCD', '55LCE', '55LCF', '55LDC', '55LDD', '56JLT', '56JMT', '56JNT',
        '56JPT', '56KKA', '56KKB', '56KKC', '56KKD', '56KKU', '56KKV', '56KLA', '56KLB',
        '56KLC', '56KLD', '56KLU', '56KLV', '56KMA', '56KMB', '56KMU', '56KMV', '56KNA',
        '56KNB', '56KNU', '56KNV', '56KPU'
        ]
    tile_ids_gbr_coastal = [
        '54LWQ', '54LXQ', '54LYK', '54LYL', '54LYM', '54LYN', '54LYP', '54LYQ', '54LZK', '54LZL',
        '54LZM', '54LZN', '54LZP', '54LZQ', '55KCA', '55KCB', '55KCV', '55KDA', '55KDB', '55KDU',
        '55KDV', '55KEA', '55KEU', '55KEV', '55KFA', '55KFS', '55KFT', '55KFU', '55KFV', '55KGR',
        '55KGS', '55KGT', '55KGU', '55KGV', '55KHR', '55KHS', '55KHT', '55KHU', '55LBD', '55LBE',
        '55LBF', '55LBG', '55LBH', '55LBJ', '55LBK', '55LCC', '55LCD', '55LCE', '55LCF', '55LDC',
        '55LDD', '56JLT', '56JMT', '56JNT', '56JPT', '56KKA', '56KKB', '56KKC', '56KKD', '56KKU',
        '56KKV', '56KLA', '56KLB', '56KLC', '56KLD', '56KLU', '56KLV', '56KMA', '56KMB', '56KMC',
        '56KMU', '56KMV', '56KNA', '56KNB', '56KNU', '56KNV', '56KPU'
        ]
    tile_ids_all_coastline = [
        '47LKG', '48LWP', '49JFM', '49JFN', '49JGJ', '49JGK', '49JGL', '49JGM', '49JGN', '49JHM',
        '49KGP', '49KGQ', '49KGR', '49KHR', '50JKN', '50JKP', '50JKQ', '50JKR', '50KKB', '50KLB',
        '50KLC', '50KMB', '50KMC', '50KNC', '50KPC', '50KPD', '50KPF', '50KQC', '50KQD', '50KQF',
        '50KQG', '50KRC', '50KRD', '51KTU', '51KUA', '51KUB', '51KUU', '51KUV', '51KVA', '51KVB',
        '51KVV', '51KWA', '51KWB', '51KXB', '51LUE', '51LVC', '51LVG', '51LWC', '51LWD', '51LWE',
        '51LWG', '51LXC', '51LXD', '51LXE', '51LYC', '51LYD', '51LYE', '51LZD', '51LZE', '51LZF',
        '52LBK', '52LBL', '52LCH', '52LCJ', '52LCK', '52LDH', '52LDJ', '52LDK', '52LEJ', '52LEK',
        '52LEL', '52LFL', '52LFM', '52LFN', '52LGM', '52LGN', '52LHM', '52LHN', '53KRB', '53LKG',
        '53LKH', '53LKJ', '53LLG', '53LLH', '53LMG', '53LMH', '53LND', '53LNE', '53LNF', '53LNG',
        '53LNH', '53LPC', '53LPD', '53LPE', '53LPF', '53LPG', '53LPH', '53LQC', '53LQE', '53LQG',
        '53LRC', '54KTG', '54KUF', '54KUG', '54KVF', '54KVG', '54KWG', '54LTH', '54LUH', '54LWH',
        '54LWJ', '54LWK', '54LWL', '54LWM', '54LWN', '54LWP', '54LXM', '54LXN', '54LXP', '57JWF',
        '57JWH', '58JGN', '58JGP'
        ]
    return {
        "15th_percentile": {
            "NorthernAU": tile_ids_all,
            "GBR": tile_ids_gbr
        },
        "low_tide_true_colour": {
            "NorthernAU": tile_ids_all_coastline,
            "GBR": tile_ids_gbr_coastal
        },
        "low_tide_infrared": {
            "NorthernAU": tile_ids_all_coastline,
            "GBR": tile_ids_gbr_coastal
        }
    }

def get_tiles_from_csv(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return [row['TileID'] for row in reader if 'TileID' in row]

def main():
    parser = argparse.ArgumentParser(description="Download AIMS S2 data tiles.")
    parser.add_argument("--output", help="Specify the output directory for downloads", default="data/in-3p/AU_AIMS_S2-comp")
    parser.add_argument("--region", help="Specify the tile region to download (NorthernAU, GBR, custom)", default="all")
    parser.add_argument("--custom", help="Path to CSV file with custom TileIDs (used with --region custom)")
    parser.add_argument("--dataset", help="Specify the dataset to download (15th_percentile, low_tide_true_colour, low_tide_infrared)", default="15th_percentile")
    parser.add_argument("--tiles", type=str, nargs='+', default=[], 
                        help="Space-separated list of TileIDs to download. For example: --tiles 46LGM 46LGN")
    args = parser.parse_args()

    # Ensure the output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    downloader = DataDownloader(args.output)
    tile_sets = get_tile_sets()

    if args.dataset not in tile_sets:
        print(f"Error: Invalid dataset specified. Choose from {', '.join(tile_sets.keys())}")
        sys.exit(1)

    if args.tiles:
        # Validate that --region is provided and valid
        if args.region not in tile_sets[args.dataset]:
            print(f"Error: When using --tiles, you must specify a valid region using --region. Choose from {', '.join(tile_sets[args.dataset].keys())}")
            sys.exit(1)

        # Filter the specified tiles to ensure they exist in the given region
        region_tiles = set(tile_sets[args.dataset][args.region])
        user_tiles = args.tiles
        invalid_tiles = [tile for tile in user_tiles if tile not in region_tiles]

        if invalid_tiles:
            print(f"Error: The following TileIDs are not valid for region '{args.region}': {', '.join(invalid_tiles)}")
            sys.exit(1)

        tiles_to_download = {args.region: user_tiles}
    elif args.region in tile_sets[args.dataset]:
        tiles_to_download = {args.region: tile_sets[args.dataset][args.region]}
    elif args.region == "custom":
        if not args.custom:
            print("Error: --custom argument is required when using --region custom")
            sys.exit(1)
        custom_tiles = get_tiles_from_csv(args.custom)
        tiles_to_download = {"custom": custom_tiles}
    else:
        print(f"Error: Invalid region specified. Choose from {', '.join(tile_sets[args.dataset].keys())}, custom")
        sys.exit(1)

    base_urls = {
        "15th_percentile": "https://nextcloud.eatlas.org.au/s/cHbWktYnk2DSJ4Y/download?path=%2FgeoTiffs&files=AU_AIMS_MARB-S2-comp_p15_TrueColour_{TileID}_v2_2015-2024.tif",
        "low_tide_true_colour": "https://nextcloud.eatlas.org.au/s/cYzPXsbJkfFRmre/download?path=%2FgeoTiffs&files=AU_AIMS_MARB-S2-comp_low-tide_p30_TrueColour_{TileID}.tif",
        "low_tide_infrared": "https://nextcloud.eatlas.org.au/s/YHxo69E2e8tP66H/download?path=%2FgeoTiffs&files=AU_AIMS_MARB-S2-comp_low-tide_p30_NearInfraredFalseColour_{TileID}.tif"
    }

    total_tiles = sum(len(tiles) for tiles in tiles_to_download.values())
    print(f"Starting download of {total_tiles} tiles from dataset '{args.dataset}'")

    downloaded_count = 0

    for region_name, tiles in tiles_to_download.items():
        print(f"\nProcessing tile region: {region_name}")
        set_output_dir = os.path.join(args.output, args.dataset, region_name)
        if not os.path.exists(set_output_dir):
            os.makedirs(set_output_dir)
        
        for tile in tiles:
            downloaded_count += 1
            url = base_urls[args.dataset].format(TileID=tile)
            file_name = url.split('=')[-1]
            file_path = os.path.join(set_output_dir, file_name)
            print(f"\nDownloading {downloaded_count} of {total_tiles} images")
            print(f"Processing tile: {tile}")
            downloader.download(url, file_path, tile)

    if downloader.missing_tiles:
        print("\nThe following TileIDs were not downloaded due to 404 errors:")
        for missing_tile in downloader.missing_tiles:
            print(missing_tile)
            
    print("\nAll downloads completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Exiting...")
        sys.exit(1)
