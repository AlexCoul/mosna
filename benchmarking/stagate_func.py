import pandas as pd
import numpy as np
import scanpy as sc
import stagate
import matplotlib.pyplot as plt
from stagate.Train_STAGATE import train_STAGATE
from stagate.utils import Cal_Spatial_Net

def create_test_adata()-> object:
    """ 
     Create a simple testing adata object
     
    """
    
    df_markers = pd.read_csv('stagate/data/test_markers.csv')
    df_coords = pd.read_csv('stagate/data/test_coords.csv')
    ### make addata o
    adata = sc.AnnData(df_markers)
    adata.obsm['spatial'] = df_coords.values
    return adata
    
def make_features_STARGATE(
    adata: object
    )-> object:

    """ Compute feature vectors of each node in a network given the STARGATE method.
    Accept adata object with X and Y coordinates and columns with gene/protein expression
    

    Parameters
    ----------
    Adata : Annotated object with values, obsm:'spatial' and optionally uns['Spatial_Net']. 
    
    Returns
    -------
    anndata
        Anndata object 
        
    """
    # It doesn't work withour this function. I just put the n of genes higher than the b of markers 
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # checks if there is the spatial graph in the adata, if not - constructs it itself:
    if 'Spatial_Net' not in adata.uns.keys():
        Cal_Spatial_Net(adata, rad_cutoff=50)
    adata = train_STAGATE(adata, alpha=0)
    return adata

def clustering_louvain(
    adata: object,
    resolution: float=0.8
    )-> object:
    """ Compute louvain clustering using the results of stagate
    Accept adata object with ['niches_vectors']
    Parameters
    ----------
    Adata : Annotated object with values, obsm:'niches_vectors'
    Resolution: float for resoltion of the method
    
    Returns
    -------
     anndata
        Anndata object 
        
    """
    sc.pp.neighbors(adata, use_rep='niches_vectors')
    sc.tl.umap(adata)
    sc.tl.louvain(adata, resolution=resolution)
    n_unique = adata.obs['louvain'].nunique()
    print(f'Number of unique clusters: {n_unique}')
    return adata

def niches_visualization(
    adata:object) -> None:
    plt.rcParams["figure.figsize"] = (6, 6)
    sc.pl.spatial(adata, color=['louvain', 'cell_type'],  spot_size=20)