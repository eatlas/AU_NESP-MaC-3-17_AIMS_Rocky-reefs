This dataset generates an estimate of the intertidal rocky reefs along the tropical Australian coastline, based
on Sentinel 2 composite imagery using both true colour imagery and infrared false colour imagery. This dataset
is intended to provide raw polygons for manual review to be incorporated into the North and West Australia Features
dataset (AU_NESP-MaC-3-17_AIMS_NW-Aus-Features). It is not intended to be its own dataset because it does not
include additional quality control.

This dataset was produced by training a random forest classifier on Sentinel 2 composite imagery with 6 bands
as inputs to the classifier. The classifier was provided with point training data create via expert review of 
the satellite imagery. Training data points were placed with the goal of capturing the diversity of each classification.
The classification included:
- Open water
- Coral reef
- Sediment
- Rocky reef

The dataset focuses on extracting the rocky reefs, and primarily focused only on the intertidal rocky rocks. 
Subtidal rocky reefs overlap in the colour space with coral reefs and so can only be reliably distinguished
by texture and context. For this we rely on manual mapping. The shallow intertidal rocky reefs have a relatively
strong colour signature, particularly in the far red and infrared bands. This makes using pixel based classification
effective and reasonably reliable.

To ensure the output polygons were as clean as possible we used the following post prediction processing:
1. We predicted probability instead of classification to obtain a continuously variable estimate of the 
rockiness of the terrain. This helps with the post smoothing process.
2. We apply a level adjustment to trim off values below and above a given threshold. This ensures that 
their noise doesn't contribute to subsequent processing.
3. We apply a median filter to integrate the local confidence of the rocky nature. This also removes a lot
of spurious small features and tracks along edges of features where the colours pass through the values
that match the rock prediction. 
4. We the apply a dilation and erosion to fill in holes in the reefs where there are small patches of
sediment that can cause the rocky reef to appear fragmented.
5. We mask out land areas so that predictions on land are not converted to polygons. We do this masking
in pixel space by rasterising the land mask, because it is much faster. To ensure that the resulting 
polygons will overlap with any final land mask we do this operation on a land mask that has had a negative
buffer applied so that we retain approximately a 50 m overlap with the land. This will remove most of the
land area artefacts, whilst ensure we can trim the final dataset to our high resolution coastline dataset.
6. We upscale the prediction to double the resolution (~5 m) prior to converting to a polygon. This helps
reduce the step size of the pixels, making it easier to remove the stair case in the polygon boundaries
without loosing too much detail.
7. We convert to a polygon, then apply a simplication to eliminate the stair case from the pixel boundaries.


### Requirements
- **Python**: Version 3.9 or higher. The scripts have been tested on Python 3.9 and 3.11.

1. Install required Python packages using the following command:
```bash
conda env create -f environment-3-13.yaml
```
2. Activate the environment
```bash
conda activate rocky-reefs_3-13
```
  
### Required Packages
The following are the top level libraries needed. These will in turn pull in many dependencies.
  - python=3.11.11
  - affine=2.3.0
  - geopandas=0.14.2
  - matplotlib-base=3.9.2
  - numpy-base=1.26.4
  - pandas=2.1.1
  - rasterio=1.3.10
  - scipy=1.14.1
  - shapely=2.0.6
  - tqdm=4.67.1
  - scikit-image=0.24.0
  - fiona=1.10.1
  - opencv-python-headless=4.10.0 (installed via pip as conda was causing DLL issues)

## Creating the land mask
While the land mask can be produced using the `05-prepare-landmask.py` script Python is
very slow at this process and so we perform the processing in QGIS instead as it is
at least 20 x faster.
The script serves as a reference for the steps needed to be processed.