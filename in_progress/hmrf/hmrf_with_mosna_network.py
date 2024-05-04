#!/usr/bin/env python3

"""
This script is designed to perform the Hidden Markov Random Field (HMRF) method, given a network. The goal is to enable HMRF implementation
on a cellular network reconstruction, as produced by mosna. It is compatible with a variety of different spatial omics methods, although it
will require different preprocessing for different methods.


The data imported in the main function, from files like "fcortex.coordinates.txt" are public smFISH data. They can be found here:
https://bitbucket.org/qzhudfci/smfishhmrf-py/src/master/data/


The script operates either on specified files containing gene names, spatial coordinates, and gene expression data, or on the test data created
in the function load_test_data_mosna_style(). It supports filtering based on specific field values, normalization of expression data, and spatial
graph construction.

Flags or Parameters:
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

Note: This script utilizes the smfishHmrf library for reading data and constructing spatial models. Ensure all dependencies are installed and the 
smfishHmrf framework is correctly set up before executing the script, see link: https://bitbucket.org/qzhudfci/smfishhmrf-py/src/master/smfishHmrf/
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
import pandas as pd





def investigate_data_structure(data, name):
    """
    This function is for code development only. It's purpose is to show what the data objects required by hmrf-py look like
    """
    print(f"Analyzing {name}:")

    # Shape and Size
    try:
        print(f"Shape of {name}:", data.shape)
    except AttributeError:
        print(f"{name} does not have a shape attribute, likely not an array or DataFrame.")

    # Data Types within Array or Series
    try:
        print(f"Data type of elements in {name}:", data.dtype)
    except AttributeError:
        # This block will execute if 'data' is a list or similar
        print(f"{name} does not have a dtype attribute, checking individual elements.")
        if isinstance(data, list):
            types = set(type(element) for element in data)
            print(f"Types of elements in {name} list:", types)

    # Inspecting the First Few Elements
    try:
        print(f"First 5 elements of {name}:", data[:5])
    except TypeError:
        print(f"Cannot slice {name}, it may not support indexing.")

    # Summary Statistics for numpy arrays or pandas structures
    if isinstance(data, np.ndarray):
        print(f"Mean values (if applicable) in {name}:", np.mean(data, axis=0)[:5])
        print(f"Standard deviation (if applicable) in {name}:", np.std(data, axis=0)[:5])
        print(f"Min values in {name}:", np.min(data, axis=0)[:5])
        print(f"Max values in {name}:", np.max(data, axis=0)[:5])
        print(f"Missing values in {name}:", np.isnan(data).sum())
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        print(f"Summary statistics for {name}:", data.describe())
        print(f"Missing values in {name}:", data.isnull().sum().sum())

    # Unique Values (for lists or pandas structures)
    if isinstance(data, list):
        print(f"Number of unique elements in {name} list:", len(set(data)))
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        print(f"Unique values per column in {name}:", data.nunique())

    # Extra for non-numeric data like 'genes'
    if isinstance(data, (list, pd.Series)) and all(isinstance(item, str) for item in data):
        print(f"Sample elements from {name}:", data[:5])

    print(f"End of analysis for {name}\n")

    return 0


def display_data_struct(expr, genes, spatial_coords):
    """
    This function is for code development only. It's purpose is to show what the data objects required by hmrf-py look like
    """
    expr_dtype = type(expr)

    print("The expr parameter is of type {}".format(expr_dtype))
    print(expr_dtype[:5])

    print("------------")
    investigate_data_structure(expr, "expr")
    print("------------")
    investigate_data_structure(genes, "genes")
    print("------------")
    investigate_data_structure(spatial_coords, "spatial_coords")

    return 0


def get_paths_seqfish_data():
    # Set directory paths and print to check if correct
    base_dir = os.path.abspath("..")
    print("Our base directory is set to {}".format(base_dir))

    input_data_dir = os.path.join(base_dir, "data")        #"data/giotto_seqfish_hmrf_data")
    print("Our input data directory is set to {}".format(input_data_dir))

    output_dir = os.path.join(base_dir, "output/giotto_seqfish_hmrf_output")
    print("Our output directory is set to {}".format(output_dir))

    return base_dir, input_data_dir, output_dir

def get_genes(input_data_dir):
    # Get file containing gene names
    file_genes = os.path.join(input_data_dir, "genes")
    genes = reader.read_genes(file_genes)

    print(genes)

    return (genes)


def load_test_data_mosna_style():
    # Increase the number of test nodes
    total_nodes = 2000
    # Simulating more complex network edges, resembling clusters or communities
    mosna_edges = [(i, (i+1) % total_nodes) for i in range(total_nodes)]  # Chain-like connections
    mosna_edges += [(i, (i+10) % total_nodes) for i in range(0, total_nodes, 10)]  # Additional cross links to simulate community

    np.random.seed(42)  # Seed for reproducibility
    # More genes for variability, simulating different expression patterns
    expr = np.random.rand(50, total_nodes)
    # Introducing clusters in expression data by adding patterns
    for cluster_start in range(0, total_nodes, 20):
        for gene in range(10):
            expr[gene, cluster_start:cluster_start+10] += 3  # Increase expression in 'clusters'
    expr = zscore(expr, axis=1)

    # Increase gene list corresponding to the new expression data size
    genes = [f'Gene{i+1}' for i in range(50)]

    # No specific cell identifiers, but generate corresponding to new total_nodes
    cells = [f'Cell{i+1}' for i in range(total_nodes)]

    # Generate more natural spatial coordinates: simulate a tissue section or cell culture
    spatial_coords = np.random.normal(loc=0.0, scale=1.0, size=(total_nodes, 2))  # 2D Gaussian distribution for positioning

    # Identify bottom-left quarter cells
    median_x, median_y = np.median(spatial_coords, axis=0)
    bottom_left_cells = [i for i, (x, y) in enumerate(spatial_coords) if x < median_x and y < median_y]

    # Increase expression for a subsection of the genes in the bottom-left quarter cells
    num_genes_to_modify = len(genes) // 3  # 1/3 of the genes
    for gene in range(num_genes_to_modify):
        expr[gene, bottom_left_cells] += 10  # Significantly increase expression in 'clusters'

    expr = zscore(expr, axis=1)  # Re-normalize after modification

    # Return all generated data structures
    return expr, genes, cells, spatial_coords, mosna_edges, total_nodes





def create_adjacency_list_from_mosna(mosna_edges, total_nodes):
    """
    Converts MOSNA network reconstruction output into an adjacency list format compatible with calc_independent_region().
    
    Args:
    mosna_edges (list of tuples): Each tuple represents an edge between two nodes, identified by their indices.
    total_nodes (int): The total number of nodes in the network.
    
    Returns:
    dict: A dictionary where keys are node indices (0-indexed), and values are sets of indices representing adjacent nodes.
    """

    # Initialize an empty dictionary to hold the adjacency list
    adjacency_list = {i: set() for i in range(total_nodes)}
    
    # Iterate through the MOSNA output edges
    for node1, node2 in mosna_edges:
        # Add each node to the other's adjacency set
        adjacency_list[node1].add(node2)
        adjacency_list[node2].add(node1)
    
    # Convert sets to sorted lists for consistency
    adjacency_list = {node: sorted(adj_list) for node, adj_list in adjacency_list.items()}
    
    return adjacency_list


def execute_hmrf(expr, genes, cells, spatial_coords, output_dir, mosna_edges, total_nodes):
    # Initialize data object
    this_dset = DatasetMatrixSingleField(expr, genes, cells, spatial_coords)
    
    # Use MOSNA approach to set adjacency information
    adj_list = create_adjacency_list_from_mosna(mosna_edges, total_nodes)
    this_dset.set_adjacency_from_external_source(adj_list)
    
    # Test, showing adjacency list
    print("This is what our adjacency list object looks like: {}".format(adj_list))

    # Now, it’s safe to call this method
    this_dset.calc_independent_region()
    # Initialize data object
    this_dset = DatasetMatrixSingleField(expr, genes, cells, spatial_coords)


    # get_adjacency_list() is part of constructing the network with giotto. If we use mosna for the network reconstruction, this may not be necessary.
    # However, we do need to provide calc_independent_region() - our objective - with an adjacency list, that contains in the 1st column all nodes, and
    # in the next columns all nodes with which this node shares an edge (according to the network reconstruction). As far as I know mosna doesn't create
    # this. Therefore, we should make a new function to do so in mosna, and make sure the output matches what calc_independent_region() requires
    """ # --------------- smfish-py meets Giotto approach ---------------
    this_dset.get_adjacency_list()

    this_dset.test_adjacency_list([0.3, 0.5, 1], metric="euclidean")
    #get_adjacency_list(dist=s_dist, cutoff=percent_value)
    """

    # --------------- mosna approach ---------------
    #adj_list = create_adjacency_list_from_mosna(mosna_edges, total_nodes)




    # Determine the different regions in the network, based on gene expression or marker values
    #this_dset.calc_independent_region()

    plt.scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=this_dset.blocks, cmap='jet', s=1)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Independent Regions')
    plt.colorbar()  # Show color scale

    plt.savefig(os.path.join(output_dir, "independent_regions.png"))
    print("Plot saved to:", os.path.join(output_dir, "giotto_independent_regions.png"))

    print("Showing plot...")
    print(f"Total plotted points: {len(spatial_coords)}")  # Should print 2000 for test data
    plt.show()





if __name__=="__main__":

    cortex_FDs = [0]  # Set field value, based on which to filter data

    base_dir, input_data_dir, output_dir = get_paths_seqfish_data()
    genes = get_genes(input_data_dir)

    # Get spatial coordinates
    file_coords = os.path.join(input_data_dir, "fcortex.coordinates.txt")
    spatial_coords, field = reader.read_coord(file_coords)

    # Check if coordinates are stored correctly
    print(spatial_coords[:5])

    file_expression = os.path.join(input_data_dir, "fcortex.expression.txt")

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


    # --- Check if data types are correct ---

    display_data_struct(expr, genes, spatial_coords)

    # ----------------------------------------


    expr, genes, cells, spatial_coords, mosna_edges, total_nodes = load_test_data_mosna_style()



    # Execute Hidden Markov Random Field (HMRF) method, see repo https://bitbucket.org/qzhudfci/smfishhmrf-py/src/master/smfishHmrf/
    cells = None    # List of cell identifiers (not required)
    execute_hmrf(expr, genes, cells, spatial_coords, output_dir, mosna_edges, total_nodes)
