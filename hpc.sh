#!/bin/bash 					
#SBATCH --ntasks=50 			## The number of paralel tasks are limited to 50. Please change this number to match your environment. A higher number here will normally mean longer wait in the Slurm queue.
#SBATCH --cpus-per-task=1		## The scipt will use 1 CPU per task. 
#SBATCH --mem-per-cpu=5G		## Pleease adjust this to match your environment. The higher the number the longer the wait in the Slurm queue.
#SBATCH --job-name=AU-NESP		## Name that shows in the Slurm queue
#SBATCH --time=0				## Zero means no (wall) time limitation. Wall time is maximum time to run the script for. 
#SBATCH --partition=cpuq		## Please change cpuq to your partition name (or delete the whole line to automatically assign partition)

module purge ## Purge all loaded modules
module load slurm ## load Slurm module. On some HPCs this is automatically loaded after Purge.
module load conda/anaconda3 #Loading Anaconda module. Change this top suit your environment. Execute module avail to get the list of available modules in your environment. 
## The following 4 lines test whether the directory exists and if not, the script will clone the code from GitHub, otherwise, it will skip to after fi
if [ ! -d "AU_NESP-MaC-3-17_AIMS_Rocky-reefs" ] ; then 
		echo "cloning"
		git clone https://github.com/eatlas/AU_NESP-MaC-3-17_AIMS_Rocky-reefs
fi
cd AU_NESP-MaC-3-17_AIMS_Rocky-reefs ## move to a directory where the code is.
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
    python 01b-create-virtual-rasters.py --base-dir ~/AU_AIMS_S2-comp
    # Create the flag file to indicate the download is complete
    touch "$FLAG_FILE"
else
    echo "Data already downloaded. Skipping download step."
fi

python 01c-download-input-data.py
python 02-extract-training-data.py --imagery-path ~/AU_AIMS_S2-comp
python 04-train-random-forest.py
