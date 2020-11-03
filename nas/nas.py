import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy
import networkx as nx
import igraph as ig
from sklearn.utils import shuffle


############ Make test networks ############

def make_triangonal_net():
    dict_nodes = {'x': [1,3,2],
                  'y': [2,2,1],
                  'a': [1,0,0],
                  'b': [0,1,0],
                  'c': [0,0,1]}
    nodes = pd.DataFrame.from_dict(dict_nodes)
    
    data_edges = [[0,1],
                  [1,2],
                  [2,0]]
    edges = pd.DataFrame(data_edges, columns=['source','target'])
    
    return nodes, edges

def make_trigonal_net():
    dict_nodes = {'x': [1,3,2,0,4,2],
                  'y': [2,2,1,3,3,0],
                  'a': [1,0,0,1,0,0],
                  'b': [0,1,0,0,1,0],
                  'c': [0,0,1,0,0,1]}
    nodes = pd.DataFrame.from_dict(dict_nodes)
    
    data_edges = [[0,1],
                  [1,2],
                  [2,0],
                  [0,3],
                  [1,4],
                  [2,5]]
    edges = pd.DataFrame(data_edges, columns=['source','target'])
    
    return nodes, edges

def make_P_net():
    dict_nodes = {'x': [0,0,0,0,1,1],
                  'y': [0,1,2,3,3,2],
                  'a': [1,0,0,0,0,0],
                  'b': [0,0,0,0,1,0],
                  'c': [0,1,1,1,0,1]}
    nodes = pd.DataFrame.from_dict(dict_nodes)
    
    data_edges = [[0,1],
                  [1,2],
                  [2,3],
                  [3,4],
                  [4,5],
                  [5,2]]
    edges = pd.DataFrame(data_edges, columns=['source','target'])
    
    return nodes, edges

def make_high_assort_net():
    dict_nodes = {'x': np.arange(12).astype(int),
                  'y': np.zeros(12).astype(int),
                  'a': [1] * 4 + [0] * 8,
                  'b': [0] * 4 + [1] * 4 + [0] * 4,
                  'c': [0] * 8 + [1] * 4}
    nodes = pd.DataFrame.from_dict(dict_nodes)
    
    edges_block = np.vstack((np.arange(3), np.arange(3) +1)).T
    data_edges = np.vstack((edges_block, edges_block + 4, edges_block + 8))
    edges = pd.DataFrame(data_edges, columns=['source','target'])
    
    return nodes, edges

def make_high_disassort_net():
    dict_nodes = {'x': [1,2,3,4,4,4,3,2,1,0,0,0],
                  'y': [0,0,0,1,2,3,4,4,4,3,2,1],
                  'a': [1,0,0] * 4,
                  'b': [0,1,0] * 4,
                  'c': [0,0,1] * 4}
    nodes = pd.DataFrame.from_dict(dict_nodes)
    
    data_edges = np.vstack((np.arange(12), np.roll(np.arange(12), -1))).T
    edges = pd.DataFrame(data_edges, columns=['source','target'])
    
    return nodes, edges

def make_random_graph_2libs(nb_nodes=100, p_connect=0.1, attributes=['a', 'b', 'c'], multi_mod=False):

    # initialize the network
    G = nx.fast_gnp_random_graph(nb_nodes, p_connect, directed=False)
    pos = nx.kamada_kawai_layout(G)
    nodes = pd.DataFrame.from_dict(pos, orient='index', columns=['x','y'])
    edges = pd.DataFrame(list(G.edges), columns=['source', 'target'])

    # set attributes
    if multi_mod:
        nodes_class = np.random.randint(0, 2, size=(nb_nodes, len(attributes))).astype(bool)
        nodes = nodes.join(pd.DataFrame(nodes_class, index=nodes.index, columns=attributes))
    else:
        nodes_class = np.random.choice(attributes, nb_nodes)
        nodes = nodes.join(pd.DataFrame(nodes_class, index=nodes.index, columns=['nodes_class']))
        nodes = nodes.join(pd.get_dummies(nodes['nodes_class']))

    if multi_mod:
        for col in attributes:
        #     nx.set_node_attributes(G, df_nodes[col].to_dict(), col.replace('+','AND')) # only for glm extension file
            nx.set_node_attributes(G, nodes[col].to_dict(), col)
    else:
        nx.set_node_attributes(G, nodes['nodes_class'].to_dict(), 'nodes_class')
    
    return nodes, edges, G

############ Assortativity ############

def count_edges_undirected(nodes, edges, attributes):
    """Compute the count of edges whose end nodes correspond to given attributes.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes
    edges : dataframe
        Edges between nodes given by their index
    attributes: list
        The attributes of nodes whose edges are selected
        
    Returns
    -------
    count : int
       Count of edges
    """
    
    pairs = np.logical_or(np.logical_and(nodes.loc[edges['source'], attributes[0]].values, nodes.loc[edges['target'], attributes[1]].values),
                          np.logical_and(nodes.loc[edges['target'], attributes[0]].values, nodes.loc[edges['source'], attributes[1]].values))
    count = pairs.sum()
    
    return count

def count_edges_directed(nodes, edges, attributes):
    """Compute the count of edges whose end nodes correspond to given attributes.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes
    edges : dataframe
        Edges between nodes given by their index
    attributes: list
        The attributes of nodes whose edges are selected
        
    Returns
    -------
    count : int
       Count of edges
    """
    
    pairs = np.logical_and(nodes.loc[edges['source'], attributes[0]].values, nodes.loc[edges['target'], attributes[1]].values)
    count = pairs.sum()
    
    return count

def mixing_matrix(nodes, edges, attributes, normalized=True, double_diag=True):
    """Compute the mixing matrix of a network described by its `nodes` and `edges`.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes
    edges : dataframe
        Edges between nodes given by their index
    attributes: list
        Categorical attributes considered in the mixing matrix
    normalized : bool (default=True)
        Return counts if False or probabilities if True.
    double_diag : bool (default=True)
        If True elements of the diagonal are doubled like in NetworkX or iGraph 
       
    Returns
    -------
    mixmat : array
       Mixing matrix
    """
    
    mixmat = np.zeros((len(attributes), len(attributes)))

    for i in range(len(attributes)):
        for j in range(i+1):
            mixmat[i, j] = count_edges_undirected(nodes, edges, attributes=[attributes[i],attributes[j]])
            mixmat[j, i] = mixmat[i, j]
        
    if double_diag:
        for i in range(len(attributes)):
            mixmat[i, i] += mixmat[i, i]
            
    if normalized:
        mixmat = mixmat / mixmat.sum()
    
    return mixmat

# NetworkX code:
def attribute_ac(M):
    """Compute assortativity for attribute matrix M.

    Parameters
    ----------
    M : numpy array or matrix
        Attribute mixing matrix.

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(e)-sum(e^2))/(1-sum(e^2)),
    where e is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    try:
        import numpy
    except ImportError:
        raise ImportError(
            "attribute_assortativity requires NumPy: http://scipy.org/ ")
    if M.sum() != 1.0:
        M = M / float(M.sum())
    M = numpy.asmatrix(M)
    s = (M * M).sum()
    t = M.trace()
    r = (t - s) / (1 - s)
    return float(r)

def mixmat_to_df(mixmat, attributes):
    return pd.DataFrame(mixmat, columns=attributes, index=attributes)

def mixmat_to_columns(mixmat):
    """
    Flattens a mixing matrix taking only elements of the lower triangle and diagonal.
    """
    N = mixmat.shape[0]
    val = []
    for i in range(N):
        for j in range(i+1):
            val.append(mixmat[i,j])
    return val

def attributes_pairs(attributes, prefix='', medfix=' - ', suffix=''):
    """
    Make a list of unique pairs of attributes.
    """
    N = len(attributes)
    col = []
    for i in range(N):
        for j in range(i+1):
            col.append(prefix + attributes[i] + medfix + attributes[j] + suffix)
    return col

def randomized_mixmat(nodes, edges, attributes, n_shuffle=20):
    
    mixmat_rand = np.zeros((n_shuffle, len(attributes), len(attributes)))
    for i in range(n_shuffle):
        nodes_rand = deepcopy(nodes)
        nodes_rand[attributes] = shuffle(nodes_rand[attributes].values)
        mixmat_rand[i] = mixing_matrix(nodes_rand, edges, attributes)
    return mixmat_rand

def zscore(mat, mat_rand, axis=0, return_stats=False):
    rand_mean = mat_rand.mean(axis=axis)
    rand_std = mat_rand.std(axis=axis)
    zscore = (mat - rand_mean) / rand_std
    if return_stats:
        return rand_mean, raand_std, zscore
    else:
        return zscore

############ Neighbors Aggegation Statistics ############