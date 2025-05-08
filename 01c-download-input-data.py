from data_downloader import DataDownloader
import os
import configparser


# Read configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
version = config.get('general', 'version')

# Create an instance of the DataDownloader class
downloader = DataDownloader(download_path=f"data/{version}/in-3p")

print("Downloading source data files. This will take a while ...")
    
# --------------------------------------------------------
# Australian Coastline 50K 2024 (NESP MaC 3.17, AIMS)
# https://eatlas.org.au/geonetwork/srv/eng/catalog.search#/metadata/c5438e91-20bf-4253-a006-9e9600981c5f
# Hammerton, M., & Lawrey, E. (2024). Australian Coastline 50K 2024 (NESP MaC 3.17, AIMS) (2nd Ed.) [Data set]. eAtlas. https://doi.org/10.26274/qfy8-hj59
direct_download_url = 'https://nextcloud.eatlas.org.au/s/DcGmpS3F5KZjgAG/download?path=%2FV1-1%2F&files=Split'
downloader.download_and_unzip(direct_download_url, 'Coast50k_2024', subfolder_name='Split', flatten_directory=True)

# Use this version for overview maps
direct_download_url = 'https://nextcloud.eatlas.org.au/s/DcGmpS3F5KZjgAG/download?path=%2FV1-1%2F&files=Simp'
downloader.download_and_unzip(direct_download_url, 'Coast50k_2024', subfolder_name='Simp', flatten_directory=True)


# --------------------------------------------------------
# Input data for the Rocky reefs dataset. This includes training data for the model
# and land mask.
downloader.download_path = f'data/{version}'
direct_download_url = f'https://nextcloud.eatlas.org.au/s/QD84aRGoKYs3KtP/download?path=%2F{version}%2F&files=in'
downloader.download_and_unzip(direct_download_url, 'in', flatten_directory=True)



