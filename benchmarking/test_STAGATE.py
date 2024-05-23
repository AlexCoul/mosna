#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import scanpy as sc
import stagate
import matplotlib.pyplot as plt
from stagate.Train_STAGATE import train_STAGATE
from stagate.utils import Cal_Spatial_Net
import stagate.stagate_func as SGT

import tensorflow.compat.v1 as tf

#tf.disable_eager_execution()

# Define the directory containing spot files
preprocessed_dir = 'preprocessed/'

# Set total number of spots to be processed
all_spots = list(range(71))
print(all_spots)

# Iterate over all spot files in the directory
for spot in all_spots:
  
    try:
        df_markers = pd.read_csv(os.path.join(preprocessed_dir, f'spot_{spot}_markers.csv'))
        df_coords = pd.read_csv(os.path.join(preprocessed_dir, f'spot_{spot}_coords.csv'))
        df_cell_types = pd.read_csv(os.path.join(preprocessed_dir, f'spot_{spot}_cell_types.csv'))
    except FileNotFoundError as e:
        print(f"Error reading files for spot {spot}: {e}")
        continue

    # Read markers, coordinates, and cell types data for the current spot
    #df_markers = pd.read_csv(preprocessed_dir + f'spot_{spot_number}_markers.csv')
    #df_coords = pd.read_csv(preprocessed_dir + f'spot_{spot_number}_coords.csv')
    #df_cell_types = pd.read_csv(preprocessed_dir + f'spot_{spot_number}_cell_types.csv')
    
    # Create an Anndata object from the data
    adata_object = sc.AnnData(df_markers)
    adata_object.obsm['spatial'] = df_coords.values
    cell_types = df_cell_types['ClusterName']
    adata_object.obs["cell_type"] = pd.Categorical(cell_types)
    
    print(f"An Anndata object is created for spot {spot}")

    # Perform feature generation and add features to the Anndata object
    adata_with_features = SGT.make_features_STARGATE(adata_object)
    print(f"Features are made and added to Anndata object for spot {spot}")

    # Perform Louvain clustering on the Anndata object
    adata_with_louvain_clustering = SGT.clustering_louvain(adata_object)
    print(f"Louvain clustering has been performed on Anndata object for spot {spot}")

    # Perform niche visualization
    print(f"We show the Anndata object prior to niche visualization for spot {spot}")
    print(adata_object)
    print(adata_object.X)
    SGT.niches_visualization(adata_with_louvain_clustering)
    print(f"Niche detection has been performed without errors for spot {spot}")
    
    # Save the plot for the current spot
    plt.savefig(f'/home/bram/CRCT/test_stagate/output/niches_visualization_spot_{spot}.png')
    plt.close()
