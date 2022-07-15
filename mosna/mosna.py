import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import stats
from scipy.stats import ttest_ind    # Welch's t-test
from scipy.stats import mannwhitneyu # Mann-Whitney rank test
from scipy.stats import ks_2samp     # Kolmogorov-Smirnov statistic
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
import warnings
from sklearn.impute import KNNImputer


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
    
    if M.sum() != 1.0:
        M = M / float(M.sum())
    M = np.asmatrix(M)
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


###### Survival and response statistics ######

def clean_data(data, method='mixed', thresh=1, cat_cols=None, modify_infs=True, verbose=1):
    """
    Delete or impute missing or non finite data.
    During imputation, they are replaced by continuous values, not by binary values.
    We correct them into int values

    Parameters
    ----------
    data : datafrrame
        Dataframe with nan or inf elements.
    method : str
        Imputation method or 'drop' to discard lines. 'mixed' allows
        to drop lines that have more than a given threshold of non finite values,
        then use the knn imputation method to replace the remaining non finite values.
    thresh : int or float
        Absolute or relative number of finite variables for a line to be conserved.
        If 1, all variables (100%) have to be finite.
    cat_cols : None or list
        If not None, list of categorical columns were imputed float values 
        are transformed to integers
    modify_infs : bool
        If True, inf values are also replaced by imputation, or discarded.
    verbose : int
        If 0 the function stays quiet.
    
    Return
    ------
    data : dataframe
        Cleaned-up data.
    select : If method == 'drop', returns also a boolean vector
        to apply on potential other objects like survival data.
    """
    to_nan = ~np.isfinite(data.values)
    nb_nan = to_nan.sum()
    if nb_nan != 0:
        if verbose > 0: 
            print(f"There are {nb_nan} non finite values")
        # set also inf values to nan
        if modify_infs:
            data[to_nan] = np.nan
        # convert proportion threshold into absolute number of variables threshold
        if method in ['drop', 'mixed'] and (0 < thresh <= 1):
            thresh = thresh * data.shape[1]
        if method in ['drop', 'mixed']:
            # we use a custom code instead of pandas.dropna to return the boolean selector
            count_nans = to_nan.sum(axis=1)
            select = count_nans <= thresh
            data = data.loc[select, :]
        # impute non finite values (nan, +/-inf)
        if method in ['knn', 'mixed']:
            if verbose > 0:
                print('Imputing data')
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            data.loc[:,:] = imputer.fit_transform(data.values)
            # set to intergers the imputed int-coded categorical variables
            # note: that's fine for 0/1 variables, avoid imputing on me categories
            if cat_cols is not None:
                data.loc[:, cat_cols] = data.loc[:, cat_cols].round().astype(int)
    if select is not None:
        return data, select
    else:
        return data


def make_clean_dummies(data, thresh=1, drop_first_binnary=True, verbose=1):
    """
    Delete missing or non finite categorical data and make dummy variables from them.
    Contrary to pandas' `get_dummy`, here nan values are not replaced by 0.

    Parameters
    ----------
    data : datafrrame
        Dataframe of categorical data.
    thresh : int or float
        Absolute or relative number of finite variables for a line to be conserved.
        If 1, all variables (100%) have to be finite.
    drop_first_binnary : bool
        If True, the first dummy variable of a binary variable is dropped.
    verbose : int
        If 0 the function stays quiet.
    
    Return
    ------
    df_dum : dataframe
        Cleaned dummy variables.
    """

    # convert proportion threshold into absolute number of variables threshold
    if (0 < thresh <= 1):
        thresh = thresh * data.shape[1]
    # delete colums that have too many nan
    df_cat = data.dropna(axis=1, thresh=thresh)
    col_nan = df_cat.isna().sum()

    # one hot encoding of categories:
    # we make the nan dummy variable otherwise nan are converted and information is lost
    # then we manually change corresponding nan values and drop this column
    df_dum = pd.get_dummies(df_cat, drop_first=False, dummy_na=True)
    for col, nb_nan in col_nan.iteritems():
        col_nan = col + '_nan'
        if nb_nan > 0:
            columns = [x for x in df_dum.columns if x.startswith(col + '_')]
            df_dum.loc[df_dum[col_nan] == 1, columns] = np.nan
        df_dum.drop(columns=[col_nan], inplace=True)

    # Drop first class of binary variables for regression
    if drop_first_binnary:
        for col, nb_nan in col_nan.iteritems():
            columns = [x for x in df_dum.columns if x.startswith(col + '_')]
            if len(columns) == 2:
                if verbose > 0: 
                    print("dropping first class:", columns[0])
                df_dum.drop(columns=columns[0], inplace=True)
    return df_dum


def binarize_data(data, zero, one):
    """
    Tranform specific values of an array, dataframe or index into 0s and 1s.
    """
    binarized = deepcopy(data)
    binarized[binarized == zero] = 0
    binarized[binarized == one] = 1
    return binarized


def convert_quanti_to_categ(data, method='median'):
    """
    Transform continuous data into categorical data.
    """
    categ = {}
    if method == 'median':
        for col in data.columns:
            new_var = f'> med( {col} )'
            new_val = data[col] > np.median(data[col])
            categ[new_var] = new_val
    categ = pd.DataFrame(categ)
    return categ


def extract_X_y(data, y_name, y_values=None, col_names=None, binarize=True):
    """
    Extract data corresponding to specific values of a target variable.
    Useful to fit or train a statistical (learning) model. 

    Parameters
    ----------
    data : dataframe
        Data containing the X variables and target y variable
    y_name : str
        Name of the column or index of the target variable
    y_values : list or None
        List of accepted conditions to extract observations
    col_names : list or None
        List of variable to extract.
    binarize : bool
        If true and `y_values` has 2 elements, the vector `y` is
        binarized, with the 1st and 2nd elements of `y_vallues`
        tranformed into 0 and 1 respectivelly.
    
    Returns
    -------
    X : dataframe
        Data corresponding to specific target y values.
    y : array
        The y values related to X.
    """

    if col_names is None:
        col_names = data.columns
    # if the y variable is in a pandas multiindex:
    if y_name not in col_names and y_name in data.index.names:
        X = data.reset_index()
    else:
        X = deepcopy(data)
    if y_values is None:
        y_values = X[y_name].unique()

    # select desired groups
    select = np.any([X[y_name] == i for i in y_values], axis=0)
    y = X.loc[select, y_name]
    X = X.loc[select, col_names]
    if len(y_values) == 2 and binarize:
        y = binarize_data(y, y_values[0], y_values[1])
    return X, y


def make_composed_variables(data, use_col=None, method='ratio', order=2):
    """
    Create derived or composed variables from simpler ones.

    Example
    -------
    >>> df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [2, 4, 6],
            'c': [6, 12, 18],
        })
    >>> mosna.make_composed_variables(df)
     (a / b)   (a / c)   (b / c)  ((a / b) / (a / c))  ((a / b) / (b / c))  \
    0      0.5  0.166667  0.333333                  3.0                  1.5   
    1      0.5  0.166667  0.333333                  3.0                  1.5   
    2      0.5  0.166667  0.333333                  3.0                  1.5   

    ((a / c) / (b / c))  
    0                  0.5  
    1                  0.5  
    2                  0.5 
    """
    
    if use_col is None:
        use_col = data.columns
    if method == 'ratio':
        combis = list(combinations(use_col, 2))
        new_vars = {}
        for var_1, var_2 in combis:
            new_var_name = f"({var_1} / {var_2})"
            new_vars[new_var_name] = data[var_1] / data[var_2]
        new_data = pd.DataFrame(data=new_vars)
    
    # make higher order composed variables recursively
    if order > 1:
        next_data = make_composed_variables(new_data, method=method, order=order-1)
        new_data = pd.concat([new_data, next_data], axis=1)

    return new_data


def find_DE_markers(data, group_ref, group_tgt, group_var, markers=None, composed_vars=False, 
                    composed_order=2, test='Kolmogorov-Smirnov', fdr_method='indep', alpha=0.05):
    

    if composed_vars:
        data = pd.concat([data, make_composed_variables(data, order=composed_order)], axis=1)
    if markers is None:
        markers = data.columns
    if isinstance(group_var, str):
        if group_var in data.columns:
            group_var = data[group_var].values
        elif group_var in data.index.names:
            group_var = data.index.to_frame()[group_var]
        else:
            raise ValueError('The name of the group variable is not in columns nor in the index.')

    select_tgt = group_var == group_tgt
    if group_ref == 'other':
        select_ref = group_var != group_tgt
    else:
        select_ref = group_var == group_ref
    select_tgt = select_tgt.values
    select_ref = select_ref.values

    pvals = []
    for marker in markers:
        dist_tgt = data.loc[select_tgt, marker]
        dist_ref = data.loc[select_ref, marker]
        if test == 'Mann-Whitney':
            mwu_stat, pval = mannwhitneyu(dist_tgt, dist_ref)
        if test == 'Welch':
            w_stat, pval = ttest_ind(dist_tgt, dist_ref, equal_var=False)
        if test == 'Kolmogorov-Smirnov': 
            ks_stat, pval = ks_2samp(dist_tgt, dist_ref)
        pvals.append(pval)
    pvals = pd.DataFrame(data=pvals, index=markers, columns=['pval'])

    if fdr_method is not None:
        rejected, pval_corr = fdrcorrection(pvals['pval'], method=fdr_method)
        pvals['pval_corr'] = pval_corr
    
    return pvals


def plot_distrib_groups(data, groups_labels, groups=None, pval_data=None, pval_col='pval_corr', pval_thresh=0.05, 
                        max_cols=-1, exclude_vars=None, id_vars=None, var_name='variable', value_name='value', 
                        multi_ind_to_col=False, figsize=(20, 6), fontsize=20, orientation=30, ax=None):
    """
    Plot the distribution of variables by groups.
    """

    # Select variables that will be plotted
    if groups is None:
        groups = data[groups_labels].unique()
    if len(groups) == 2 and pval_data is not False:
        if isinstance(pval_data, str) and pval_data == 'compute':
            pval_data = find_DE_markers(data, groups[0], groups[1], order=0)
        nb_vars = np.sum(pval_data[pval_col] <= pval_thresh)
        if max_cols > 0:
            nb_vars = min(nb_vars, max_cols)
        var_names = pval_data.sort_values(by=pval_col, ascending=True).head(nb_vars).index
    else:
        var_names = data.columns
    # filter variable_names if exclude_vars was given
    if exclude_vars is not None:
        var_names = [x for x in var_names if x not in exclude_vars]
    
    wide = data.loc[:, var_names]
    if multi_ind_to_col:
        if id_vars is None:
            id_vars = wide.index.names
        wide = wide.reset_index()
    
    # select desired groups
    select = np.any([wide[groups_labels] == i for i in groups], axis=0)
    wide = wide.loc[select, :]

    long = pd.melt(
        wide, 
        id_vars=id_vars, 
        value_vars=var_names,
        var_name=var_name, 
        value_name=value_name)
    select = np.isfinite(long[value_name])
    long = long.loc[select, :]

    if ax is None:
        ax_none = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax_none = False
    if len(groups) == 2:
        split = True
    else:
        split = False
    sns.violinplot(x=var_name, y=value_name, hue=groups_labels,
                   data=long, palette="Set2", split=split, ax=ax);
    plt.xticks(rotation=orientation, ha='right', fontsize=fontsize);
    plt.yticks(fontsize=fontsize);
    if ax_none:
        return fig, ax


def plot_heatmap(data, obs_labels=None, groups_labels=None, groups=None, use_col=None, z_score=1, 
                 palette=None, figsize=(10, 10), fontsize=10, xlabels_rotation=30, ax=None):

    data = data.copy(deep=True)
    if obs_labels is not None:
        data.index = data[obs_labels]
        data.drop(columns=[obs_labels], inplace=True)
    if use_col is None:
        skip_cols = [obs_labels, groups_labels]
        use_col = [x for x in data.columns if x not in skip_cols]
    else:
        data = data[use_col]

    if groups_labels is not None:
        if groups is None:
            groups = data[groups_labels].unique()        
        # select desired groups
        data = data.query(f'{groups_labels} in @groups')
        # make lut group <--> color
        if palette is None:
            palette = sns.color_palette()
        lut = dict(zip(groups, palette))
        # Make the vector of colors
        colors = data[groups_labels].map(lut)
        data.drop(columns=[groups_labels], inplace=True)
    else:
        colors = None
    
    g = sns.clustermap(data, z_score=z_score, figsize=figsize, row_colors=colors)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=xlabels_rotation, ha='right', fontsize=fontsize);
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=fontsize);
    if hasattr(g, 'ax_row_colors'):
        g.ax_row_colors.set_xticklabels(g.ax_row_colors.get_xticklabels(), rotation=xlabels_rotation, ha='right', fontsize=fontsize);
    return g


def color_val_inf(val, thresh=0.05, col='green', col_back='white'):
    """
    Takes a scalar and returns a string with
    the css property `'color: green'` for values
    below a threshold, black otherwise.
    """
    color = col if val < thresh else col_back
    return 'color: %s' % color


def find_survival_variable(surv, X, reverse_response=False, return_table=True, return_model=True, model_kwargs=None, model_fit=None):
    """
    Fit a CoxPH model for each single variable, and detect the ones
    that are statistically significant.
    """
    pass



# ------ Stepwise linear / logistic regression ------

def forward_regression(X, y,
                       learner=sm.OLS, # sm.Logit
                       threshold_in=0.05,
                       verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            try:
                model = learner(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            except np.linalg.LinAlgError:
                print(f"LinAlgError with column {new_column}")
                new_pval[new_column] = 1
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included


def backward_regression(X, y,
                        learner=sm.OLS,
                        threshold_out=0.05,
                        verbose=False):
    included=list(X.columns)
    while True:
        changed=False
        model = learner(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def stepwise_regression(X, y=None,
                        y_name=None,
                        y_values=None,
                        col_names=None,
                        learner=sm.OLS,
                        threshold_in=0.05,
                        threshold_out=0.05,
                        support=1,
                        verbose=False,
                        ignore_warnings=True,
                        kwargs_model={},
                        kwargs_fit={}):
    """
    Parameters
    ----------
    suport : int
        Minimal "support", i.e the minimal number of
        different values (avoid only 1s, etc...)
    """

    if y is None:
        X, y = extract_X_y(X, y_name, y_values)
    if col_names is not None:
        col_names = [x for x in X.columns if x in col_names]
        X = X[col_names]
    
    # drop variable that don't have enough different values
    if support > 0:
        drop_cols = []
        for col in X.columns:
            uniq = X[col].value_counts()
            if len(uniq) == 1:
                drop_cols.append(col)
            else:
                # drop variables with non-most numerous values are too few
                minority_total = uniq.sort_values(ascending=False).iloc[1:].sum()
                if minority_total < support:
                    drop_cols.append(col)
        if len(drop_cols) > 0:
            X.drop(columns=drop_cols, inplace=True)
            if verbose:
                print("Dropping variables with not enough support:\n", drop_cols)
        
    if ignore_warnings:
        warnings.filterwarnings("ignore")
    initial_list = []
    included = list(initial_list)
    # record of dropped columns to avoid infinite cycle of adding/dropping
    drop_history = []
    
    while True:
        changed = False
        # ------ Forward selection ------
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            try:
                model = learner(y, sm.add_constant(pd.DataFrame(X[included+[new_column]])), **kwargs_model).fit(**kwargs_fit)
                new_pval[new_column] = model.pvalues[new_column]
            except np.linalg.LinAlgError:
                print(f"LinAlgError with column {new_column}")
                new_pval[new_column] = 1
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
            
            # ------ Backward selection ------
            while True:
                back_changed = False
                model = learner(y, sm.add_constant(pd.DataFrame(X[included])), **kwargs_model).fit(**kwargs_fit)
                # use all coefs except intercept
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max() # null if pvalues is empty
                if worst_pval > threshold_out:
                    worst_feature = pvalues.idxmax()
                    if worst_feature in drop_history:
                        changed = False # escape the forward/backward selection
                        if verbose:
                            print('Variable "{:30}" already dropped once, escaping adding/dropping cycle.'.format(worst_feature))
                    else: 
                        back_changed = True
                        included.remove(worst_feature)
                        drop_history.append(worst_feature)
                        if verbose:
                            print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
                if not back_changed:
                    break
        
        if not changed:
            model = learner(y, sm.add_constant(pd.DataFrame(X[included])), **kwargs_model).fit(**kwargs_fit)
            return model, included



# ------ Risk ratios ------

def relative_risk(expo, nonexpo, alpha_risk=0.05):
    """
    Compute the relative risk between exposed and non exposed conditions.
    Diseases is coded as True or 1, healthy is codes as False or 0.
    alpha is the risk, default is 0.05.
    
    Example
    -------
    >>> expo = np.array([1, 1, 0, 0])
    >>> nonexpo = np.array([1, 0, 0, 0])
    >>> relative_risk(expo, nonexpo)
    """
        
    # number of exposed
    Ne = expo.size
    # number of diseased exposed
    De = expo.sum()
    # number of healthy exposed
    He = Ne - De
    # number of non-exposed
    Nn = nonexpo.size
    # number of diseased non-exposed
    Dn = nonexpo.sum()
    # number of healthy non-exposed
    Hn = Nn - Dn
    # relative risk
    RR = (De / Ne) / (Dn / Nn)
    
    # confidence interval
    eff = np.sqrt( He / (De * Ne) + Hn / (Dn + Nn))
    Z_alpha = np.array(stats.norm.interval(1 - alpha_risk, loc=0, scale=1))
    interv = np.exp( np.log(RR) + Z_alpha * eff)
    
    return RR, interv


def make_expo(control, test, filter_obs=None):
    """
    Make arrays of exposed and non-exposed samples from a control
    array defining the exposure (True or 1) and a test array defining
    disease status (True or 1).
    """
    
    # filter missing values
    select = np.logical_and(np.isfinite(control), np.isfinite(test))
    # combine with given selector
    if filter_obs is not None:
        select = np.logical_and(select, filter_obs)
    control = control[select]
    test = test[select]
    control = control.astype(bool)
    test = test.astype(bool)
    
    expo = test[control]
    nonexpo = test[~control]
    return expo, nonexpo


def make_risk_ratio_matrix(data, y_name=None, y_values=None, rows=None, columns=None, 
                           alpha_risk=0.5, col_filters={}):
    """
    Make the matrices of risk ratio and lower and upper bounds
    of confidence intervals.
    
    col_filters is a dictionnary to select observations for a given set of columns.
    the keys are the conditionnal columns, values are dictionaries which keys are either
    'all' to apply selector to all target columns of several taret columns names.
    """
    if y_name is not None:
        X, y = extract_X_y(data, y_name, y_values, binarize=False)
        X[y_name] = y
        data = X
    if rows is None:
        rows = data.columns
    if columns is None:
        columns = data.columns
    N = len(columns)
    rr = pd.DataFrame(data=np.zeros((N, N)), index=columns, columns=columns)
    rr_low = rr.copy()
    rr_high = rr.copy()

    for i in rows:
        for j in columns:
            # i tells what variable is used to define exposure
            # j is used to define disease status
            if i == j:
                rr.loc[i, j], (rr_low.loc[i, j], rr_high.loc[i, j]) = 1, (1, 1)
            else:
                if i in col_filters:
                    if 'all' in col_filters[i]:
                        filter_obs = col_filters[i]['all']
                        expo, nonexpo = make_expo(data[i], data[j], filter_obs=filter_obs)
                    elif j in col_filters[i]:
                        filter_obs = col_filters[i][j]
                        expo, nonexpo = make_expo(data[i], data[j], filter_obs=filter_obs)
                    else:
                        expo, nonexpo = make_expo(data[i], data[j])
                else:
                    expo, nonexpo = make_expo(data[i], data[j])
                rr.loc[i, j], (rr_low.loc[i, j], rr_high.loc[i, j]) = relative_risk(expo, nonexpo, alpha_risk=alpha_risk)
    # significance matrix
    rr_sig = (rr_low > 1) | (rr_high < 1)
    
    return rr, rr_low, rr_high, rr_sig