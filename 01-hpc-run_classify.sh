#!/bin/bash
#SBATCH --job-name=rockyreef_classification   # Job name
#SBATCH --output=logs/%A_%a.out                # Standard output log (%A=jobID, %a=array index)
#SBATCH --error=logs/%A_%a.err                 # Standard error log
#SBATCH --time=00:20:00                        # Wall time limit (20 minutes per task)
#SBATCH --array=0-122                          # Job array index range (adjust if you have a different number of tiles)
#SBATCH --cpus-per-task=1                      # Number of CPUs per task (adjust if needed)
#SBATCH --mem=4G                               # Memory per task (adjust based on your needs)

module purge ## Purge all loaded modules
module load conda/anaconda3

echo "activate rocky-reefs_3-13"
## Activate virtual environment for us to run the scripts and access the libraries that we need
conda activate rocky-reefs_3-13

# Run the Python script.
# The script will read the SLURM_ARRAY_TASK_ID from the environment to process a single tile.
python 06-classify-rocky-reefs.py --geotiff-dir working/rocky-reefs-geotiffs --dataset-path ~/AU_AIMS_S2-comp
