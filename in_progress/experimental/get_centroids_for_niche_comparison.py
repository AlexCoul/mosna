#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import scanpy as sc
import stagate
from stagate.Train_STAGATE import train_STAGATE
from stagate.utils import Cal_Spatial_Net
import stagate.stagate_func as SGT
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()  # Correctly disable eager execution in TensorFlow 1.x

# Define the directory containing spot files
preprocessed_dir = 'preprocessed/'
output_dir = '/home/bram/CRCT/test_stagate/output/'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set total number of spots to be processed
all_spots = list(range(71))
print(all_spots)

# Function to calculate centroids
def extract_centroids(adata):
    # Group by cluster label and calculate mean for each cluster
    cluster_means = adata.to_df().groupby(adata.obs['louvain']).mean()
    return cluster_means

# Iterate over all spot files in the directory
for spot in all_spots:
    try:
        df_markers = pd.read_csv(os.path.join(preprocessed_dir, f'spot_{spot}_markers.csv'))
        df_coords = pd.read_csv(os.path.join(preprocessed_dir, f'spot_{spot}_coords.csv'))
        df_cell_types = pd.read_csv(os.path.join(preprocessed_dir, f'spot_{spot}_cell_types.csv'))
    except FileNotFoundError as e:
        print(f"Error reading files for spot {spot}: {e}")
        continue
    
    # Create an Anndata object from the data
    adata_object = sc.AnnData(df_markers)
    adata_object.obsm['spatial'] = df_coords.values
    cell_types = df_cell_types['ClusterName']
    adata_object.obs["cell_type"] = pd.Categorical(cell_types)
    
    # Perform feature generation and clustering
    adata_object = SGT.make_features_STARGATE(adata_object)
    adata_with_louvain_clustering = SGT.clustering_louvain(adata_object)

    # Calculate and save centroids for the clusters
    centroids = extract_centroids(adata_with_louvain_clustering)
    centroids.to_csv(os.path.join(output_dir, f'centroid_spot_{spot}.csv'))

    # Niche visualization and save plot
    SGT.niches_visualization(adata_with_louvain_clustering)
    plt.savefig(os.path.join(output_dir, f'niches_visualization_spot_{spot}.png'))
    plt.close()

    print(f"Completed processing for spot {spot}")

