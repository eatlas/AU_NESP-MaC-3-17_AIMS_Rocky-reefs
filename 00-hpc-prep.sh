#!/bin/bash 					
# This script prepares the Python environment and downloads the data for the project.
# It then runs all the preparation steps for the dataset. Each of these steps is run
# sequentially.

# Get this script by first running the following command in your terminal:
# git clone https://github.com/eatlas/AU_NESP-MaC-3-17_AIMS_Rocky-reefs
# cd AU_NESP-MaC-3-17_AIMS_Rocky-reefs ## move to a directory where the code is.
# Then use:
# chmod +x 00-hpc-prep.sh ## to make the script executable

module purge ## Purge all loaded modules
module load conda/anaconda3 #Loading Anaconda module. Change this top suit your environment. Execute module avail to get the list of available modules in your environment. 

echo "create environment-3-13"
## The following 4 lines will test whether the virtual environment exists and if not, create it. If yes, the script will skip to the line after fi.
if conda info --envs | grep -q reef_maps2; then
		echo "reef_maps2 already exists";
		else conda env create -f environment-3-13.yaml;
fi
echo "activate rocky-reefs_3-13"
## Activate virtual environment for us to run the scripts and access the libraries that we need
conda activate rocky-reefs_3-13

# Check if the data has already been downloaded
# If not, download it and create a flag file
FLAG_FILE="~/AU_AIMS_S2-comp/download_complete.flag"

if [ ! -f "$FLAG_FILE" ]; then
    echo "Downloading the data..."
    
    python 01a-download-sentinel2.py --dataset low_tide_true_colour --region NorthernAU --output ~/AU_AIMS_S2-comp
    python 01a-download-sentinel2.py --dataset low_tide_infrared --region NorthernAU --output ~/AU_AIMS_S2-comp
    python 01a-download-sentinel2.py --dataset low_tide_true_colour --region GBR --output ~/AU_AIMS_S2-comp
    python 01a-download-sentinel2.py --dataset low_tide_infrared --region GBR --output ~/AU_AIMS_S2-comp
    python 01b-create-virtual-rasters.py --base-dir ~/AU_AIMS_S2-comp --combine-regions
    # Create the flag file to indicate the download is complete
    touch "$FLAG_FILE"
else
    echo "Data already downloaded. Skipping download step."
fi

python 01c-download-input-data.py
python 02-extract-training-data.py --imagery-path ~/AU_AIMS_S2-comp
python 04-train-random-forest.py
