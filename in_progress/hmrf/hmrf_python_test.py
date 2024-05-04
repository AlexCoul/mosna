#!/usr/bin/env python3

"""
This script is designed for spatial data analysis using the smfishHmrf framework, particularly tailored for examining cortical tissue sections. It allows for the 
integration of spatial coordinates with gene expression data to identify distinct regions within the tissue based on gene expression profiles and spatial distribution. 

The script operates either on specified files containing gene names, spatial coordinates, and gene expression data. It supports filtering based on specific field values,
normalization of expression data, and spatial graph construction.

Flags and Parameters:
- cortex_FDs: Sets the field value(s) for filtering spatial data.
- base_dir: The base directory from where the script is executed; usually the project's root.
- input_data_dir: The directory containing input files such as genes list and spatial coordinates.
- output_dir: The directory where output files, including plots, will be saved.

Main Functionalities:
- Reads and processes gene names and spatial coordinates.
- Filters and normalizes gene expression data.
- Constructs spatial neighborhood graphs based on Euclidean distance.
- Identifies independent regions within the tissue sample.
- Visualizes and saves the resulting independent regions as a scatter plot.

Outputs:
- A PNG image file named 'independent_regions.png' depicting the independent regions within the cortical tissue based on spatial and expression data.
- Terminal output detailing the script's progress, data dimensions, and the adjacency list testing results.

Note: This script utilizes the smfishHmrf library for reading data and constructing spatial models. Ensure all dependencies are installed and the smfishHmrf framework
is correctly set up before executing the script, see link: https://bitbucket.org/qzhudfci/smfishhmrf-py/src/master/smfishHmrf/
"""


import sys
import math
import os
import numpy as np
import scipy
import scipy.stats
from scipy.stats import zscore
from scipy.spatial.distance import euclidean,squareform,pdist
sys.setrecursionlimit(10000)
import smfishHmrf.reader as reader
from smfishHmrf.HMRFInstance import HMRFInstance
from smfishHmrf.DatasetMatrix import DatasetMatrix, DatasetMatrixSingleField


import matplotlib.pyplot as plt





if __name__=="__main__":

    cortex_FDs = [0]  # Set field value, based on which to filter data

    # Set directory paths and print to check if correct
    base_dir = os.path.abspath(".")
    print("Our base directory is set to {}".format(base_dir))

    input_data_dir = os.path.join(base_dir, "data")
    print("Our input data directory is set to {}".format(input_data_dir))

    output_dir = os.path.join(base_dir, "output")
    print("Our output directory is set to {}".format(output_dir))

    # Get file containing gene names
    file_genes = os.path.join(input_data_dir, "genes")
    genes = reader.read_genes(file_genes)

    print(genes)

    # Get spatial coordinates
    file_coords = os.path.join(input_data_dir, "fcortex.coordinates.txt")
    spatial_coords, field = reader.read_coord(file_coords)

    # Check if coordinates are stored correctly
    print(spatial_coords[:5])

    file_expression = os.path.join(input_data_dir, "fcortex.expression.txt")

    #expr = np.empty((len(genes), spatial_coords.shape[0]), dtype="float32")
    #    for ind,g in enumerate(genes):
    #        expr[ind,:] = reader.read_expression("%s/pca_corrected/f%s.gene.%s.txt" % (input_data_dir, directory, roi, g))

    expr = np.empty((len(genes), spatial_coords.shape[0]), dtype="float32")
    for ind, g in enumerate(genes):
        expr[ind, :] = reader.read_expression(file_expression)

    # Check if gene expression values stored correctly
    print(expr[:10])

    # Filter data based on field (FD) value
    good_i = np.array([i for i in range(spatial_coords.shape[0]) if field[i] in set(cortex_FDs)])
    expr = expr[:,good_i]
    spatial_coords = spatial_coords[good_i]
    field = field[good_i]

    print(field[:7])

    ngene = len(genes)
    ncell = spatial_coords.shape[0]

    print("The number of genes is {}".format(ngene))

    expr = zscore(expr, axis=1)	 #z-score per row (gene)
    expr = zscore(expr, axis=0)  #z-score per column (cell)


    this_dset = DatasetMatrixSingleField(expr, genes, None, spatial_coords)
    this_dset.test_adjacency_list([0.3, 0.5, 1], metric="euclidean")        #      are other metrics possible here? Running the script with delaunay for example, doesn't work
    this_dset.calc_neighbor_graph(0.3, metric="euclidean")          # Calculate network
    this_dset.calc_independent_region()         # Determine the different regions in the network

    #new_genes = reader.read_genes("../HMRF.genes")
    #new_dset = this_dset.subset_genes(new_genes)


    plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=this_dset.blocks, cmap='jet')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Independent Regions')
    plt.colorbar()  # Show color scale

    plt.savefig(os.path.join(output_dir, "independent_regions.png"))
    print("Plot saved to:", os.path.join(output_dir, "independent_regions.png"))

    print("Showing plot...")
    plt.show()


