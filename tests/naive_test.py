print("Performing a naive test to see if the code can run")

#########################
# imports
#########################

import requests, zipfile
import os, sys
import multiDGD
import numpy as np
import anndata as ad
import scanpy as sc

#########################
# Download data
#########################

file_name = "human_bonemarrow.h5ad.zip"
file_url = "https://api.figshare.com/v2/articles/23796198/files/41740251"

file_response = requests.get(file_url).json()
file_download_url = file_response["download_url"]
response = requests.get(file_download_url, stream=True)
with open(file_name, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Unzip
with zipfile.ZipFile(file_name, "r") as zip_ref:
    zip_ref.extractall(".")
print("   Downloaded data")

#########################
# Load data
#########################

data = ad.read_h5ad("./human_bonemarrow.h5ad")

data = data[::10, :]  # this is just to make it smaller for the demo
# set it up with the model (similar to MultiVI)
data = multiDGD.functions._data.setup_data(
    data,
    modality_key="feature_types",  # adata.var column indicating which feature belongs to which modality
    observable_key="cell_type",  # cell annotation key to initialize GMM components
    covariate_keys=["Site"],  # confounders
)
# save this data to keep the train-val-test split for later
data.write_h5ad("./example_data.h5ad")
data = sc.read_h5ad("./example_data.h5ad")
print("   Loaded and subsampled data")

#########################
# init model
#########################

if not os.path.exists("./models/"):
    os.mkdir("./models/")
model = multiDGD.DGD(data=data, save_dir="./models/", model_name="dgd_bonemarrow_default")
model.view_data_setup()
print("   Initialized model")

#########################
# train model
#########################

model.train(n_epochs=10)
model.plot_history(export=True)
model.save()
print("   Trained model")

print("Test successful")
