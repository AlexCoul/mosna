import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from copy import deepcopy
from sklearn.utils import shuffle
from tqdm import tqdm


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
    import networkx as nx
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
    Convenient to make the names of elements of the mixing matrix 
    that is flattened.
    """
    N = len(attributes)
    col = []
    for i in range(N):
        for j in range(i+1):
            col.append(prefix + attributes[i] + medfix + attributes[j] + suffix)
    return col

def core_rand_mixmat(nodes, edges, attributes):
    nodes_rand = deepcopy(nodes)
    nodes_rand[attributes] = shuffle(nodes_rand[attributes].values)
    mixmat_rand = mixing_matrix(nodes_rand, edges, attributes)
    return mixmat_rand

def randomized_mixmat(nodes, edges, attributes, n_shuffle=20, parallel='max'):
    """Randomize several times a network by shuffling the nodes' attributes.
    Then compute the mixing matrix and the corresponding assortativity coefficient.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes
    edges : dataframe
        Edges between nodes given by their index
    attributes: list
        Categorical attributes considered in the mixing matrix
    n_shuffle : int (default=20)
        Number of attributes permutations.
    parallel : bool, int or str (default="max")
        How parallelization is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
       
    Returns
    -------
    mixmat_rand : array (n_shuffle x n_attributes x n_attributes)
       Mixing matrices of each randomized version of the network
    assort_rand : array  of size n_shuffle
       Assortativity coefficients of each randomized version of the network
    """
    
    mixmat_rand = np.zeros((n_shuffle, len(attributes), len(attributes)))
    assort_rand = np.zeros(n_shuffle)
    
    if parallel is False:
        for i in tqdm(range(n_shuffle), desc="randomization"):
            mixmat_rand[i] = core_rand_mixmat(nodes, edges, attributes)
            assort_rand[i] = attribute_ac(mixmat_rand[i])
    else:
        from multiprocessing import cpu_count
        from dask.distributed import Client, LocalCluster
        from dask import delayed
        
        # select the right number of cores
        nb_cores = cpu_count()
        if isinstance(parallel, int):
            use_cores = min(parallel, nb_cores)
        elif parallel == 'max-1':
            use_cores = nb_cores - 1
        elif parallel == 'max':
            use_cores = nb_cores
        # set up cluster and workers
        cluster = LocalCluster(n_workers=use_cores, 
                               threads_per_worker=1,
                               memory_limit='50GB')
        client = Client(cluster)
        
        # store the matrices-to-be
        mixmat_delayed = []
        for i in range(n_shuffle):
            mmd = delayed(core_rand_mixmat)(nodes, edges, attributes)
            mixmat_delayed.append(mmd)
        # evaluate the parallel computation and return is as a 3d array
        mixmat_rand = delayed(np.array)(mixmat_delayed).compute()
        # only the assortativity coeff is not parallelized
        for i in range(n_shuffle):
            assort_rand[i] = attribute_ac(mixmat_rand[i])
        # close workers and cluster
        client.close()
        cluster.close()
            
    return mixmat_rand, assort_rand

def zscore(mat, mat_rand, axis=0, return_stats=False):
    rand_mean = mat_rand.mean(axis=axis)
    rand_std = mat_rand.std(axis=axis)
    zscore = (mat - rand_mean) / rand_std
    if return_stats:
        return rand_mean, rand_std, zscore
    else:
        return zscore

############ Neighbors Aggegation Statistics ############