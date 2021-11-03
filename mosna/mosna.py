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
    """
    Make a triangonal network.
    """
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
    """
    Make a trigonal network.
    """
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
    """
    Make a P-shaped network.
    """
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
    """
    Make a highly assortative network.
    """
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
    """
    Make a highly dissassortative network.
    """
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
    """
    Make a dataframe of a mixing matrix.
    """
    return pd.DataFrame(mixmat, columns=attributes, index=attributes)

def mixmat_to_columns(mixmat):
    """
    Flattens a mixing matrix taking only elements of the lower triangle and diagonal.
    To revert this use `series_to_mixmat`.
    """
    N = mixmat.shape[0]
    val = []
    for i in range(N):
        for j in range(i+1):
            val.append(mixmat[i,j])
    return val

def series_to_mixmat(series, medfix=' - ', discard=' Z'):
    """
    Convert a 1D pandas series into a 2D dataframe.
    To revert this use `mixmat_to_columns`.
    """
    N = series.size
    combi = [[x.split(medfix)[0].replace(discard, ''), x.split(medfix)[1].replace(discard, '')] for x in series.index]
    # get unique elements of the list of mists
    from itertools import chain 
    uniq = [*{*chain.from_iterable(combi)}]
    mat = pd.DataFrame(data=None, index=uniq, columns=uniq)
    for i in series.index:
        x = i.split(medfix)[0].replace(discard, '')
        y = i.split(medfix)[1].replace(discard, '')
        val = series[i]
        mat.loc[x, y] = val
        mat.loc[y, x] = val
    return mat

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
    """
    Compute the mixing matrix of a network after nodes' attributes
    are randomized once.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
       
    Returns
    -------
    mixmat_rand : array
       Mmixing matrix of the randomized network.
    """
    nodes_rand = deepcopy(nodes)
    nodes_rand[attributes] = shuffle(nodes_rand[attributes].values)
    mixmat_rand = mixing_matrix(nodes_rand, edges, attributes)
    return mixmat_rand

def randomized_mixmat(nodes, edges, attributes, n_shuffle=50, parallel='max', memory_limit='50GB'):
    """Randomize several times a network by shuffling the nodes' attributes.
    Then compute the mixing matrix and the corresponding assortativity coefficient.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
    n_shuffle : int (default=50)
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
                               memory_limit=memory_limit)
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
    
def select_pairs_from_coords(coords_ids, pairs, how='inner', return_selector=False):
    """
    Select edges related to specific nodes.
    
    Parameters
    ----------
    coords_ids : array
        Indices or ids of nodes.
    pairs : array
        Edges defined as pairs of nodes ids.
    how : str (default='inner')
        If 'inner', only edges that have both source and target 
        nodes in coords_ids are select. If 'outer', edges that 
        have at least a node in coords_ids are selected.
    return_selector : bool (default=False)
        If True, only the boolean mask is returned.
    
    Returns
    -------
    pairs_selected : array
        Edges having nodes in coords_ids.
    select : array
        Boolean array to select latter on edges.
    
    Example
    -------
    >>> coords_ids = np.array([5, 6, 7])
    >>> pairs = np.array([[1, 2],
                          [3, 4],
                          [5, 6],
                          [7, 8]])
    >>> select_pairs_from_coords(coords_ids, pairs, how='inner')
    array([[5, 6]])
    >>> select_pairs_from_coords(coords_ids, pairs, how='outer')
    array([[5, 6],
           [7, 8]])
    """
    
    select_source = np.in1d(pairs[:, 0], coords_ids)
    select_target = np.in1d(pairs[:, 1], coords_ids)
    if how == 'inner':
        select = np.logical_and(select_source, select_target)
    elif how == 'outer':
        select = np.logical_or(select_source, select_target)
    if return_selector:
        return select
    pairs_selected = pairs[select, :]
    return pairs_selected

def sample_assort_mixmat(nodes, edges, attributes, sample_id=None ,n_shuffle=50, 
                         parallel='max', memory_limit='50GB'):
    """
    Computed z-scored assortativity and mixing matrix elements for 
    a network of a single sample.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
    sample_id : str
        Name of the analyzed sample.
    n_shuffle : int (default=50)
        Number of attributes permutations.
    parallel : bool, int or str (default="max")
        How parallelization is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
    memory_limit : str (default='50GB')
        Dask memory limit for parallelization.
        
    Returns
    -------
    sample_stats : dataframe
        Network's statistics including total number of nodes, attributes proportions,
        assortativity and mixing matrix elements, both raw and z-scored.
    """
    
    col_sample = (['id', '# total'] +
                 ['% ' + x for x in attributes] +
                 ['assort', 'assort MEAN', 'assort STD', 'assort Z'] +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix='') +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix=' MEAN') +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix=' STD') +
                 attributes_pairs(attributes, prefix='', medfix=' - ', suffix=' Z'))
    
    if sample_id is None:
        sample_id = 'None'
    # Network statistics
    mixmat = mixing_matrix(nodes, edges, attributes)
    assort = attribute_ac(mixmat)

    # ------ Randomization ------
    print(f"randomization")
    np.random.seed(0)
    mixmat_rand, assort_rand = randomized_mixmat(nodes, edges, attributes, n_shuffle=n_shuffle, parallel=False)
    mixmat_mean, mixmat_std, mixmat_zscore = zscore(mixmat, mixmat_rand, return_stats=True)
    assort_mean, assort_std, assort_zscore = zscore(assort, assort_rand, return_stats=True)

    # Reformat sample's network's statistics
    nb_nodes = len(nodes)
    sample_data = ([sample_id, nb_nodes] +
                   [nodes[col].sum()/nb_nodes for col in attributes] +
                   [assort, assort_mean, assort_std, assort_zscore] +
                   mixmat_to_columns(mixmat) +
                   mixmat_to_columns(mixmat_mean) +
                   mixmat_to_columns(mixmat_std) +
                   mixmat_to_columns(mixmat_zscore))
    sample_stats = pd.DataFrame(data=sample_data, index=col_sample).T
    return sample_stats

def _select_nodes_edges_from_group(nodes, edges, group, groups):
    """
    Select nodes and edges related to a given group of nodes.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    group: int or str
        Group of interest. 
    groups: pd.Series
        Group identifier of each node. 
    
    Returns
    ------
    nodes_sel : dataframe
        Nodes belonging to the group.
    edges_sel : dataframe
        Edges belonging to the group.
    """
    select = groups == group
    nodes_sel = nodes.loc[select, :]
    nodes_ids = np.where(select)[0]
    edges_selector = select_pairs_from_coords(nodes_ids, edges.values, return_selector=True)
    edges_sel = edges.loc[edges_selector, :]
    return nodes_sel, edges_sel
    
def batch_assort_mixmat(nodes, edges, attributes, groups, n_shuffle=50,
                        parallel='max', memory_limit='50GB',
                        save_intermediate_results=False, dir_save_interm='~'):
    """
    Computed z-scored assortativity and mixing matrix elements for all
    samples in a batch, cohort or other kind of groups.
    
    Parameters
    ----------
    nodes : dataframe
        Attributes of all nodes.
    edges : dataframe
        Edges between nodes given by their index.
    attributes: list
        Categorical attributes considered in the mixing matrix.
    groups: pd.Series
        Group identifier of each node. 
        It can be a patient or sample id, chromosome number, etc...
    n_shuffle : int (default=50)
        Number of attributes permutations.
    parallel : bool, int or str (default="max")
        How parallelization is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
    memory_limit : str (default='50GB')
        Dask memory limit for parallelization.
    save_intermediate_results : bool (default=False)
        If True network statistics are saved for each group.
    dir_save_interm : str (default='~')
        Directory where intermediate group network statistics are saved.
        
    Returns
    -------
    networks_stats : dataframe
        Networks's statistics for all groups, including total number of nodes, 
        attributes proportions, assortativity and mixing matrix elements, 
        both raw and z-scored.
    
    Examples
    --------
    >>> nodes_high, edges_high = make_high_assort_net()
    >>> nodes_low, edges_low = make_high_disassort_net()
    >>> nodes = nodes_high.append(nodes_low, ignore_index=True)
    >>> edges_low_shift = edges_low + nodes_high.shape[0]
    >>> edges = edges_high.append(edges_low_shift)
    >>> groups = pd.Series(['high'] * len(nodes_high) + ['low'] * len(nodes_low))
    >>> net_stats = batch_assort_mixmat(nodes, edges, 
                                        attributes=['a', 'b', 'c'], 
                                        groups=groups, 
                                        parallel=False)
"""
    
    if not isinstance(groups, pd.Series):
        groups = pd.Series(groups).copy()
    
    groups_data = []
 
    if parallel is False:
        for group in tqdm(groups.unique(), desc='group'):
            # select nodes and edges of a specific group
            nodes_sel, edges_sel = _select_nodes_edges_from_group(nodes, edges, group, groups)
            # compute network statistics
            group_data = sample_assort_mixmat(nodes_sel, edges_sel, attributes, sample_id=group, 
                                              n_shuffle=n_shuffle, parallel=parallel, memory_limit=memory_limit)
            if save_intermediate_results:
                group_data.to_csv(os.path.join(dir_save_interm, 'network_statistics_group_{}.csv'.format(group)), 
                                  encoding='utf-8', 
                                  index=False)
            groups_data.append(group_data)
        networks_stats = pd.concat(groups_data, axis=0)
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
                               memory_limit=memory_limit)
        client = Client(cluster)
        
        for group in groups.unique():
            # select nodes and edges of a specific group
            nodes_edges_sel = delayed(_select_nodes_edges_from_group)(nodes, edges, group, groups)
            # individual samples z-score stats are not parallelized over shuffling rounds
            # because parallelization is already done over samples
            group_data = delayed(sample_assort_mixmat)(nodes_edges_sel[0], nodes_edges_sel[1], attributes, sample_id=group, 
                                                       n_shuffle=n_shuffle, parallel=False) 
            groups_data.append(group_data)
        # evaluate the parallel computation
        networks_stats = delayed(pd.concat)(groups_data, axis=0, ignore_index=True).compute()
    return networks_stats
    
############ Neighbors Aggegation Statistics ############

def neighbors(pairs, n):
    """
    Return the list of neighbors of a node in a network defined 
    by edges between pairs of nodes. 
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
        
    Returns
    -------
    neigh : array_like
        The indices of neighboring nodes.
    """
    
    left_neigh = pairs[pairs[:,1] == n, 0]
    right_neigh = pairs[pairs[:,0] == n, 1]
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    return neigh

def neighbors_k_order(pairs, n, order):
    """
    Return the list of up the kth neighbors of a node 
    in a network defined by edges between pairs of nodes
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
    order : int
        Max order of neighbors.
        
    Returns
    -------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order
    
    
    Examples
    --------
    >>> pairs = np.array([[0, 10],
                        [0, 20],
                        [0, 30],
                        [10, 110],
                        [10, 210],
                        [10, 310],
                        [20, 120],
                        [20, 220],
                        [20, 320],
                        [30, 130],
                        [30, 230],
                        [30, 330],
                        [10, 20],
                        [20, 30],
                        [30, 10],
                        [310, 120],
                        [320, 130],
                        [330, 110]])
    >>> neighbors_k_order(pairs, 0, 2)
    [[array([0]), 0],
     [array([10, 20, 30]), 1],
     [array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    """
    
    # all_neigh stores all the unique neighbors and their oder
    all_neigh = [[np.array([n]), 0]]
    unique_neigh = np.array([n])
    
    for k in range(order):
        # detected neighbor nodes at the previous order
        last_neigh = all_neigh[k][0]
        k_neigh = []
        for node in last_neigh:
            # aggregate arrays of neighbors for each previous order neighbor
            neigh = np.unique(neighbors(pairs, node))
            k_neigh.append(neigh)
        # aggregate all unique kth order neighbors
        if len(k_neigh) > 0:
            k_unique_neigh = np.unique(np.concatenate(k_neigh, axis=0))
            # select the kth order neighbors that have never been detected in previous orders
            keep_neigh = np.in1d(k_unique_neigh, unique_neigh, invert=True)
            k_unique_neigh = k_unique_neigh[keep_neigh]
            # register the kth order unique neighbors along with their order
            all_neigh.append([k_unique_neigh, k+1])
            # update array of unique detected neighbors
            unique_neigh = np.concatenate([unique_neigh, k_unique_neigh], axis=0)
        else:
            break
        
    return all_neigh

def flatten_neighbors(all_neigh):
    """
    Convert the list of neighbors 1D arrays with their order into
    a single 1D array of neighbors.

    Parameters
    ----------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order.

    Returns
    -------
    flat_neigh : array_like
        The indices of neighboring nodes.
        
    Examples
    --------
    >>> all_neigh = [[np.array([0]), 0],
                     [np.array([10, 20, 30]), 1],
                     [np.array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    >>> flatten_neighbors(all_neigh)
    array([  0,  10,  20,  30, 110, 120, 130, 210, 220, 230, 310, 320, 330])
        
    Notes
    -----
    For future features it should return a 2D array of
    nodes and their respective order.
    """
    
    list_neigh = []
    for neigh, order in all_neigh:
        list_neigh.append(neigh)
    flat_neigh = np.concatenate(list_neigh, axis=0)

    return flat_neigh

def aggregate_k_neighbors(X, pairs, order=1, var_names=None, stat_funcs='default', stat_names='default', var_sep=' '):
    """
    Compute the statistics on aggregated variables across
    the k order neighbors of each node in a network.

    Parameters
    ----------
    X : array_like
        The data on which to compute statistics (mean, std, ...).
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    order : int
        Max order of neighbors.
    var_names : list
        Names of variables of X.
    stat_funcs : str or list of functions
        Statistics functions to use on aggregated data. If 'default' np.mean and np.std are use.
        All functions are used with the `axis=0` argument.
    stat_names : str or list of str
        Names of the statistical functions used on aggregated data.
        If 'default' 'mean' and 'std' are used.
    var_sep : str
        Separation between variables names and statistical functions names
        Default is ' '.

    Returns
    -------
    aggreg : dataframe
        Neighbors Aggregation Statistics of X.
        
    Examples
    --------
    >>> x = np.arange(5)
    >>> X = x[np.newaxis,:] + x[:,np.newaxis] * 10
    >>> pairs = np.array([[0, 1],
                          [2, 3],
                          [3, 4]])
    >>> aggreg = aggregate_k_neighbors(X, pairs, stat_funcs=[np.mean, np.max], stat_names=['mean', 'max'], var_sep=' - ')
    >>> aggreg.values
    array([[ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
           [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
           [25., 26., 27., 28., 29., 30., 31., 32., 33., 34.],
           [30., 31., 32., 33., 34., 40., 41., 42., 43., 44.],
           [35., 36., 37., 38., 39., 40., 41., 42., 43., 44.]])
    """
    
    nb_obs = X.shape[0]
    nb_var = X.shape[1]
    if stat_funcs == 'default':
        stat_funcs = [np.mean, np.std]
        if stat_names == 'default':
            stat_names = ['mean', 'std']
    nb_funcs = len(stat_funcs)
    aggreg = np.zeros((nb_obs, nb_var*nb_funcs))

    for i in range(nb_obs):
        all_neigh = neighbors_k_order(pairs, n=i, order=order)
        neigh = flatten_neighbors(all_neigh)
        for j, (stat_func, stat_name) in enumerate(zip(stat_funcs, stat_names)):
            aggreg[i, j*nb_var : (j+1)*nb_var] = stat_func(X[neigh,:], axis=0)
    
    if var_names is None:
        var_names = [str(i) for i in range(nb_var)]
    columns = []
    for stat_name in stat_names:
        stat_str = var_sep + stat_name
        columns = columns + [var + stat_str for var in var_names]
    aggreg = pd.DataFrame(data=aggreg, columns=columns)
    
    return aggreg

def make_cluster_cmap(labels, grey_pos='start'):
    """
    Creates an appropriate colormap for a vector of cluster labels.
    
    Parameters
    ----------
    labels : array_like
        The labels of multiple clustered points
    grey_pos: str
        Where to put the grey color for the noise
    
    Returns
    -------
    cmap : matplotlib colormap object
        A correct colormap
    
    Examples
    --------
    >>> my_cmap = make_cluster_cmap(labels=np.array([-1,3,5,2,4,1,3,-1,4,2,5]))
    """
    
    from matplotlib.colors import ListedColormap
    
    if labels.max() < 9:
        cmap = list(plt.get_cmap('tab10').colors)
        if grey_pos == 'end':
            cmap.append(cmap.pop(-3))
        elif grey_pos == 'start':
            cmap = [cmap.pop(-3)] + cmap
        elif grey_pos == 'del':
            del cmap[-3]
    else:
        cmap = list(plt.get_cmap('tab20').colors)
        if grey_pos == 'end':
            cmap.append(cmap.pop(-6))
            cmap.append(cmap.pop(-6))
        elif grey_pos == 'start':
            cmap = [cmap.pop(-5)] + cmap
            cmap = [cmap.pop(-5)] + cmap
        elif grey_pos == 'del':
            del cmap[-5]
            del cmap[-5]
    cmap = ListedColormap(cmap)
    
    return cmap