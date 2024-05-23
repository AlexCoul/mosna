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



# load_genes
def load_data(input_data_dir, genes = "genes", coordiantes_file = "fcortex.coordinates.txt", expression_file = "fcortex.expression.txt"):
    file_genes = os.path.join(input_data_dir, genes)
    Genes = reader.read_genes(file_genes)

    file_coords = os.path.join(input_data_dir, coordiantes_file)
    Coords, field = reader.read_coord(file_coords)

    file_expression = os.path.join(input_data_dir, expression_file)
    # Initialize an empty numpy array 
    Expr = np.empty((len(Genes), Coords.shape[0]), dtype="float32")
    for ind, g in enumerate(genes):# Loop over each gene in the list of genes
        # For each gene, read its expression data from the file and assign it to the corresponding row in the 'expr' array
        Expr[ind, :] = reader.read_expression(file_expression)
    return expr

def filter_data_by_field(spatial_coords, field, cortex_FDs):
    good_i = np.array([i for i in range(spatial_coords.shape[0]) if field[i] in cortex_FDs])
    return spatial_coords[good_i], field[good_i]