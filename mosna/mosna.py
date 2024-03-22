import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path
from time import time
import joblib
from itertools import combinations
from functools import partial
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy import stats
from scipy.stats import ttest_ind    # Welch's t-test
from scipy.stats import mannwhitneyu # Mann-Whitney rank test
from scipy.stats import ks_2samp     # Kolmogorov-Smirnov statistic
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import xgboost
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import hdbscan
import composition_stats as cs
import igraph as ig
import leidenalg as la
import colorcet as cc
import re
from typing import Optional, Any, List, Tuple, Union, Iterable, Callable, Dict, Set

from multiprocessing import cpu_count
from dask.distributed import Client, LocalCluster, progress
from dask import delayed
import dask

from tysserand import tysserand as ty

try:
    from cuml import UMAP
except:
    from umap import UMAP



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

def randomized_mixmat(
        nodes, 
        edges, 
        attributes, 
        n_shuffle=50, 
        parallel='max', 
        memory_limit='10GB',
        verbose=1,
        ):
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
        if verbose > 0:
            iterable = tqdm(range(n_shuffle), desc='randomization')
        else:
            iterable = range(n_shuffle)
        for i in iterable:
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
    # with warnings.simplefilter("ignore", category=RuntimeWarning):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zscore = (mat - rand_mean) / rand_std
    if return_stats:
        return rand_mean, rand_std, zscore
    else:
        return zscore
    
# DEPRECATED: batch computations will be performed by loading each individual network files
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
                         parallel='max', memory_limit='10GB', verbose=1):
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
    np.random.seed(0)
    mixmat_rand, assort_rand = randomized_mixmat(
        nodes, edges, attributes, 
        n_shuffle=n_shuffle, 
        parallel=parallel, 
        memory_limit=memory_limit,
        verbose=verbose)
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

# DEPRECATED: batch computations will be performed by loading each individual network files
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
                        parallel_groups='max', parallel_shuffle=False, memory_limit='50GB',
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
    parallel_groups : bool, int or str (default="max")
        How parallelization across groups is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
    parallel_shuffle : bool, int or str (default="False)
        How parallelization across shuffle rounds is performed.
        Parameter options are identical to `parallel_groups`.
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
                                        parallel_groups=False)
    """

    
    # TODO: add selection of subset
    if not isinstance(groups, pd.Series):
        groups = pd.Series(groups).copy()
    
    groups_data = []
 
    if parallel_groups is False:
        for group in tqdm(groups.unique(), desc='group'):
            # select nodes and edges of a specific group
            nodes_sel, edges_sel = _select_nodes_edges_from_group(nodes, edges, group, groups)
            # compute network statistics
            group_data = sample_assort_mixmat(nodes_sel, edges_sel, attributes, sample_id=group, 
                                              n_shuffle=n_shuffle, parallel=parallel_shuffle, memory_limit=memory_limit)
            if save_intermediate_results:
                group_data.to_csv(os.path.join(dir_save_interm, f'network_statistics_group_{group}.csv'), 
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
        if isinstance(parallel_groups, int):
            use_cores = min(parallel_groups, nb_cores)
        elif parallel_groups == 'max-1':
            use_cores = nb_cores - 1
        elif parallel_groups == 'max':
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
                                                       n_shuffle=n_shuffle, parallel=parallel_shuffle) 
            groups_data.append(group_data)
        # evaluate the parallel computation
        networks_stats = delayed(pd.concat)(groups_data, axis=0, ignore_index=True).compute()
    return networks_stats


def make_group_network_stats(
    net_dir,
    data_info,
    extension,
    read_fct,
    id_level_1,
    id_level_2=None,
    attributes_col=None,
    use_attributes=None, 
    make_onehot=False,
    n_shuffle=50,
    parallel_shuffle=False, 
    memory_limit='10GB',
    save_intermediate_results=False,
    dir_save_interm=None,
    verbose=1):
    """
    Load the network data of a specific sample group, i.e. a specific pair
    of  id_level_1 and id_level_2, and computes its mixing matrix elements
    and assortativity.    
    """

    # load nodes and edges of a specific group
    str_group = f'{id_level_1}-{data_info[0]}_{id_level_2}-{data_info[1]}'
    nodes = read_fct(net_dir / f'nodes_{str_group}.{extension}')
    edges = read_fct(net_dir / f'edges_{str_group}.{extension}')

    # make dummy variables for attributes (ex: phenotype) if needed
    if make_onehot:
        nodes = nodes.join(pd.get_dummies(nodes[attributes_col], prefix='', prefix_sep=''))
    if use_attributes is None:
        use_attributes = np.unique(nodes[attributes_col])
    # compute network statistics
    group_data = sample_assort_mixmat(
        nodes, edges, 
        attributes=use_attributes, 
        sample_id=str_group, 
        n_shuffle=n_shuffle, 
        parallel=parallel_shuffle, 
        memory_limit=memory_limit, 
        verbose=verbose)
    
    if save_intermediate_results:
        if dir_save_interm is None:
            dir_save_interm = net_dir / '.temp'
        dir_save_interm.mkdir(parents=True, exist_ok=True)
        group_data.to_parquet(dir_save_interm / f'network_statistics_{str_group}.parquet', index=False)
    
    return group_data


def groups_assort_mixmat(
    net_dir, 
    attributes_col,
    use_attributes=None, 
    make_onehot=False,
    id_level_1='patient',
    id_level_2='sample', 
    extension='parquet',
    data_index=None,
    n_shuffle=50,
    parallel_groups='max', 
    memory_limit='max',
    save_intermediate_results=False, 
    dir_save_interm=None,
    verbose=1):
    """
    Compute z-scored assortativity and mixing matrix elements for all
    samples in a batch, cohort or other kind of groups.
    
    Parameters
    ----------
    net_dir: str or path object
        Location of reconstructed networks data with nodes and edges files.
    attributes_col: str or list
        Column containing attributes, multiple columns if attributes are already one-hot encoded.
    use_attributes: list
        Categorical attributes considered in the mixing matrix (ex: phenotypes).
    make_onehot: bool
        If True, make one-hot encoded variables from `attributes_col`.
    id_level_1: str
        Label in filenames used to identify the first level of data (ex: patient_id).
    id_level_2: str or None
        Label in filenames used to identify the second level of data (ex: sample_id).
    extension: str
        Extension used to save network data. Either 'parquet' (default) or 'csv'.
    data_index: list(list) or list or None
        Index of all groups, i.e. patients and their samples, or genes and their loci.
        If None, the index is built from files in net_dir.
    n_shuffle : int (default=50)
        Number of attributes permutations.
    parallel_groups : bool, int or str (default="max")
        How parallelization across groups is performed.
        If False, no parallelization is done.
        If int, use this number of cores.
        If 'max', use the maximum number of cores.
        If 'max-1', use the max of cores minus 1.
    memory_limit : str (default='max')
        Dask memory limit for parallelization.
        If 'max, will use 95% of the available free memory.
    save_intermediate_results : bool (default=False)
        If True network statistics are saved for each group.
    dir_save_interm : str (default=None)
        Directory where intermediate group network statistics are saved.
        If None, data is saved in net_dir / '.temp'.
        
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
                                        parallel_groups=False)
    """

    net_dir = Path(net_dir)
    data_single_level = id_level_2 is None
    
    if isinstance(attributes_col, str):
        attributes_col = [attributes_col]
    if extension == 'parquet':
        read_fct = pd.read_parquet
    elif extension == 'csv':
        read_fct = pd.read_csv
    
    # build index of patients and samples files
    if data_index is None:
        data_index = []
        len_ext = len(extension) + 1
        len_l1 = len(id_level_1) + 1
        len_l2 = len(id_level_2) + 1
        files = net_dir.glob(f'edges_*.{extension}')
        if not data_single_level:
            for file in files:
                # print(file)
                # parse patient and sample description
                file_name = file.name[6:-len_ext]
                patient_info, sample_info = file_name.split('_')
                patient_id = patient_info[len_l1:]
                sample_id = sample_info[len_l2:]
                
                # add info to data index
                data_index.append((patient_id, sample_id))
    
    groups_data = []
    
    # redefine defaults values of the network analysis function
    use_group_network_stats = partial(
        make_group_network_stats,
        net_dir=net_dir,
        extension=extension,
        read_fct=read_fct,
        attributes_col=attributes_col,
        use_attributes=use_attributes, 
        make_onehot=make_onehot,
        id_level_1=id_level_1,
        id_level_2=id_level_2,
        n_shuffle=n_shuffle,
        parallel_shuffle=False,  # don't parallelize over iterations per network
        memory_limit=memory_limit,
        save_intermediate_results=save_intermediate_results,
        dir_save_interm=dir_save_interm,
        verbose=0)  # don't display iterations
    
    if parallel_groups is False:
        if verbose > 0:
            iterable = tqdm(data_index, desc='data')
        else:
            iterable = data_index
        for data_info in iterable:
            group_data = use_group_network_stats(data_info=data_info)
            groups_data.append(group_data)
        networks_stats = pd.concat(groups_data, axis=0)
    else:
        from multiprocessing import cpu_count
        from dask.distributed import Client, LocalCluster
        from dask import delayed
        from dask.diagnostics import ProgressBar
        
        # select the right number of cores
        nb_cores = cpu_count()
        if isinstance(parallel_groups, int):
            use_cores = min(parallel_groups, nb_cores)
        elif parallel_groups == 'max-1':
            use_cores = nb_cores - 1
        elif parallel_groups == 'max':
            use_cores = nb_cores
        if memory_limit == 'max':
            total_memory, used_memory, free_memory = map(
                int, os.popen('free -t -m').readlines()[-1].split()[1:])
            memory_limit = str(int(0.95 * free_memory/1000)) + 'GB'

        # set up cluster and workers
        with LocalCluster(
            n_workers=use_cores,
            processes=True,
            threads_per_worker=1,
            memory_limit=memory_limit,
            ) as cluster, Client(cluster) as client:
                # TODO: add dask's progressbar
                for data_info in data_index:
                    # select nodes and edges of a specific group
                    group_data = delayed(use_group_network_stats)(data_info=data_info)
                    groups_data.append(group_data)
                # evaluate the parallel computation
                # ProgressBar().register()
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

def make_features_NAS(X, pairs, order=1, var_names=None, stat_funcs='default', stat_names='default', var_sep=' '):
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
        Separation between variables names and statistical functions names.
        Default is ' '.

    Returns
    -------
    nas : dataframe
        Neighbors Aggregation Statistics of X.
        
    Examples
    --------
    >>> x = np.arange(5)
    >>> X = x[np.newaxis,:] + x[:,np.newaxis] * 10
    >>> pairs = np.array([[0, 1],
                          [2, 3],
                          [3, 4]])
    >>> nas = make_features_NAS(X, pairs, stat_funcs=[np.mean, np.max], stat_names=['mean', 'max'], var_sep=' - ')
    >>> nas.values
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
    elif not hasattr(stat_funcs, '__iter__'):
        # check if a single function has been passed
        stat_funcs = [stat_funcs]
    if stat_names == 'default':
        stat_names = ['mean', 'std']
    elif isinstance(stat_names, str):
        stat_names = [stat_names]
    nb_funcs = len(stat_funcs)
    nas = np.zeros((nb_obs, nb_var*nb_funcs))

    # check if other info as source and target are in pairs and clean array
    if pairs.shape[1] > 2:
        print("Trimmimg additonnal columns in `pairs`")
        pairs = pairs[:, :2].astype(int)
    
    for i in range(nb_obs):
        all_neigh = neighbors_k_order(pairs, n=i, order=order)
        neigh = flatten_neighbors(all_neigh)
        for j, (stat_func, stat_name) in enumerate(zip(stat_funcs, stat_names)):
            nas[i, j*nb_var : (j+1)*nb_var] = stat_func(X[neigh,:], axis=0)
        
    if var_names is None:
        var_names = [str(i) for i in range(nb_var)]
    columns = []
    for stat_name in stat_names:
        stat_str = var_sep + stat_name
        columns = columns + [var + stat_str for var in var_names]
    nas = pd.DataFrame(data=nas, columns=columns)
    
    return nas


def compute_NAS_single_network(
    net_dir: Union[str, Path], 
    data_info: List[str],
    extension: str,
    read_fct: Callable,
    id_level_1: str,
    id_level_2: Union[str, None] = None, 
    attributes_col: Union[Iterable, None] = None,
    use_attributes: Union[Iterable, None] = None, 
    make_onehot: bool = False,
    order: int = 1, 
    stat_funcs: Union[str, List[Callable]] = 'default', 
    stat_names: Union[str, List[str]] = 'default', 
    var_sep: str = ' ',
    save_intermediate_results: bool = False, 
    dir_save_interm: Union[str, Path, None] = None,
    add_sample_info: bool = True,
    verbose: int = 1,
    ) -> pd.DataFrame:
    """
    Compute the Neighbors Aggregation Statistics for a single network.

    Parameters
    ----------
    net_dir : Union[str, Path]
        Directory where network files are stored.
    data_info : List[str, str]
        Identifier IDs of sample, e.g. patient id and sample id.
    extension : str
        File format of network files.
    read_fct : Callable
        Function used to load network files.
    id_level_1 : str
        Identifier of the first level of the dataset, like
        'patient' or 'chromosome'.
    id_level_2 : Union[str, None], None
        Identifier of the second level of the dataset, like 
        'sample' or 'locus'.
    attributes_col : Union[Iterable, None]
        Unique columns storing attributes, like cell types, or
        list of columns used to aggregate variables. 
        If None, all columns are used.
    use_attributes : Union[Iterable, None], None
        If provided, subset of variables used for aggregation.
    make_onehot : bool, False
        If True, convert a single column into multiple columns.
    order : int, 1 
        Maximum order of neighborhoud for aggregation.
    stat_funcs : Union[str, List[Callable]], 'default'
        Statistics functions to use on aggregated data. 
        If 'default' np.mean and np.std are use.
        All functions are used with the `axis=0` argument.
    stat_names : Union[str, List[str]], 'default' 
        Names of the statistical functions used on aggregated data.
        If 'default' 'mean' and 'std' are used.
    var_sep : str, ' '
        Separation between variables names and statistical functions names.
    save_intermediate_results : bool, False 
        If True, save results for each network.
    dir_save_interm : Union[str, Path, None], None
        Directory of intermediate results.
    add_sample_info : bool, True
        If True, add sample information to the final NAS table.
    verbose : int, 1
        Level of information displayed.
    
    Returns
    -------
    nas : pd.DataFrame
        Table of Neighbors Aggregated Statistics.
    """

    # load nodes and edges of a specific group
    if len(data_info) == 1:
        str_group = f'{id_level_1}-{data_info[0]}'
    elif len(data_info) == 2:
        str_group = f'{id_level_1}-{data_info[0]}_{id_level_2}-{data_info[1]}'
    nodes = read_fct(net_dir / f'nodes_{str_group}.{extension}')
    edges = read_fct(net_dir / f'edges_{str_group}.{extension}')

    if attributes_col is None:
        attributes_col = nodes.columns

    # make dummy variables for attributes (ex: phenotype) if needed
    if make_onehot:
        assert len(attributes_col) == 1, "`attributes_col` has to be of length 1 to make dummy variables"
        nodes = nodes.join(pd.get_dummies(nodes[attributes_col], prefix='', prefix_sep=''))
    if use_attributes is None:
        if len(attributes_col) == 1:
            use_attributes = np.unique(nodes[attributes_col])
        else:
            use_attributes = attributes_col
    else:
        # handle missing columns
        if len(attributes_col) == 1:
           missing_cols = set(use_attributes).difference(np.unique(nodes[attributes_col]))
        else:
           missing_cols = set(use_attributes).difference(np.unique(attributes_col))
        for col in missing_cols:
            nodes[col] = 0
    
    # compute Neighbors Aggregation Statistics
    nas = make_features_NAS(
        nodes[use_attributes].astype(float).values, 
        edges.values, 
        order=order, 
        var_names=use_attributes, 
        stat_funcs=stat_funcs, 
        stat_names=stat_names, 
        var_sep=var_sep)
    if add_sample_info:
        nas[id_level_1] = data_info[0]
        nas[id_level_2] = data_info[1]
    
    if save_intermediate_results:
        if dir_save_interm is None:
            dir_save_interm = net_dir / '.temp'
        dir_save_interm.mkdir(parents=True, exist_ok=True)
        nas.to_parquet(dir_save_interm / f'nas_{str_group}.parquet', index=False)
    
    return nas


def compute_NAS_all_networks(
    net_dir: Union[str, Path],  
    attributes_col: Union[str, Iterable, None] = None,
    use_attributes: Union[Iterable, None] = None,  
    make_onehot: bool = False,
    order: int = 1, 
    id_level_1: str = 'patient',
    id_level_2: str = 'sample', 
    extension: str = 'parquet',
    data_index: Union[List[Tuple], None]=None,
    parallel_groups: Union[bool, int, str] = 'max', 
    memory_limit: str = 'max',
    save_intermediate_results: bool = False, 
    dir_save_interm: Union[str, Path, None] = None,
    add_sample_info: bool = True,
    verbose: int = 1,
    ) -> pd.DataFrame:
    """
    Compute the Neighbors Aggregation Statistics for all
    samples in a batch, cohort or other kind of groups.

    Parameters
    ----------
    net_dir : Union[str, Path]
        Directory where network files are stored.
    attributes_col : Union[str, Iterable, None], None
        Unique columns storing attributes, like cell types, or
        list of columns used to aggregate variables. 
        If None, all columns are used.
    use_attributes : Union[Iterable, None], None
        If provided, subset of variables used for aggregation.
    make_onehot : bool, False
        If True, convert a single column into multiple columns.
    order : int, 1 
        Maximum order of neighborhoud for aggregation.
    id_level_1 : str
        Identifier of the first level of the dataset, like
        'patient' or 'chromosome'.
    id_level_2 : Union[str, None], None
        Identifier of the second level of the dataset, like 
        'sample' or 'locus'.
    extension : str
        File format of network files.
    data_index : Union[List[Tuple], None], None
        List of identifier IDs of network files.
    parallel_groups : Union[bool, int, str], 'max'
        Computation is run on a single CPU if False, on the specified 
        number of CPU if it is an integer, or on the max or max-1 
        number of CPUS if it is a string.
    memory_limit : str, 'max'
        Maximum memory used by Dask during parallel computation.
        Use either 'max' or 'XX GB'.
    
    Returns
    -------
    nas : pd.DataFrame
        Table of Neighbors Aggregated Statistics.
    """

    net_dir = Path(net_dir)
    data_single_level = id_level_2 is None
    
    if isinstance(attributes_col, str):
        attributes_col = [attributes_col]
    if extension == 'parquet':
        read_fct = pd.read_parquet
    elif extension == 'csv':
        read_fct = pd.read_csv
    
    # build index of patients and samples files
    if data_index is None:
        data_index = []
        len_ext = len(extension) + 1
        len_l1 = len(id_level_1) + 1
        len_l2 = len(id_level_2) + 1
        files = net_dir.glob(f'edges_*.{extension}')
        if not data_single_level:
            for file in files:
                # print(file)
                # parse patient and sample description
                file_name = file.name[6:-len_ext]
                patient_info, sample_info = file_name.split('_')
                patient_id = patient_info[len_l1:]
                sample_id = sample_info[len_l2:]
                
                # add info to data index
                data_index.append((patient_id, sample_id))
    
    groups_data = []
    
    # redefine defaults values of the network analysis function
    use_compute_NAS_single_network = partial(
        compute_NAS_single_network,
        net_dir=net_dir,
        extension=extension,
        read_fct=read_fct,
        attributes_col=attributes_col,
        use_attributes=use_attributes, 
        make_onehot=make_onehot,
        order=order,
        id_level_1=id_level_1,
        id_level_2=id_level_2,
        save_intermediate_results=save_intermediate_results,
        dir_save_interm=dir_save_interm,
        add_sample_info=add_sample_info,
        verbose=0)  # don't display iterations
    
    if parallel_groups is False:
        if verbose > 0:
            iterable = tqdm(data_index, desc='data')
        else:
            iterable = data_index
        for data_info in iterable:
            group_data = use_compute_NAS_single_network(data_info=data_info)
            groups_data.append(group_data)
        nas = pd.concat(groups_data, axis=0)
    else:
        from multiprocessing import cpu_count
        from dask.distributed import Client, LocalCluster
        from dask import delayed
        
        # select the right number of cores
        nb_cores = cpu_count()
        if isinstance(parallel_groups, int):
            use_cores = min(parallel_groups, nb_cores)
        elif parallel_groups == 'max-1':
            use_cores = nb_cores - 1
        elif parallel_groups == 'max':
            use_cores = nb_cores
        if memory_limit == 'max':
            total_memory, used_memory, free_memory = map(
                int, os.popen('free -t -m').readlines()[-1].split()[1:])
            memory_limit = str(int(0.95 * free_memory/1000)) + 'GB'

        # set up cluster and workers
        with LocalCluster(
            n_workers=use_cores,
            processes=True,
            threads_per_worker=1,
            memory_limit=memory_limit,
            ) as cluster, Client(cluster) as client:
                # TODO: add dask's progressbar
                for data_info in data_index:
                    # select nodes and edges of a specific group
                    group_data = delayed(use_compute_NAS_single_network)(data_info=data_info)
                    groups_data.append(group_data)
                # evaluate the parallel computation
                # ProgressBar().register()
                nas = delayed(pd.concat)(groups_data, axis=0, ignore_index=True).compute()

    return nas


def screen_nas_parameters(X, pairs, markers, orders, dim_clusts, min_cluster_sizes, processed_dir, soft_clustering=True, 
                          var_type=None, downsample=False, aggreg_dir=None, save_dir=None, opt_str='', parallel_clustering=4,
                          n_neighbors=70, opt_args_reduc={}, opt_args_clust={}):
    """
    Try combinations of parameters for the Neighbors Aggregation Statistics method, 
    including the neighbors order of aggregation, the dimensionnality in which the aggregation
    is performed, and the minimum cluster size.
    Results and logs are saved as soon as an intermediate result is produced.
    
    Example
    -------
    processed_dir = Path('../data/processed/CODEX_CTCL')
    opt_str = '_samples-all_stat-mean-std'
    """


    if var_type is None:
        var_type = 'markers'
    if aggreg_dir is None:
        aggreg_dir = processed_dir / "nas"
    else:
        aggreg_dir = processed_dir / aggreg_dir
    if save_dir is None:
        save_dir = aggreg_dir / f"screening_dim_reduc_clustering_nas_on-{var_type}{opt_str}_n_neighbors-{n_neighbors}_downsample-{downsample}"
    else:
        save_dir = aggreg_dir / save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    default_args_reduc = {
        # 'n_neighbors': 30,
        'metric': 'euclidean',
        'min_dist': 0.0,
        'random_state': 0,
    }
    default_args_clust = {'metric': 'euclidean'}

    args_reduc = opt_args_reduc
    for key, val in default_args_reduc.items():
        if key not in args_reduc.keys():
            args_reduc[key] = default_args_reduc[key]
    args_clust = opt_args_clust
    for key, val in default_args_clust.items():
        if key not in args_clust.keys():
            args_clust[key] = default_args_clust[key]
    if downsample is False:
        downsample = 1

    # check if other info as source and target are in pairs and clean array
    if pairs.shape[1] > 2:
        print("Trimmimg additonnal columns in `pairs`")
        pairs = pairs[:, :2].astype(int)

    for order in orders:
        print("order: {}".format(order))
        
        # compute statistics on aggregated data across neighbors
        file_path = aggreg_dir / f'nas_on-{var_type}{opt_str}.csv'
        if os.path.exists(file_path):
            print("load var_aggreg")
            var_aggreg = pd.read_csv(file_path)
        else:
            print("computing var_aggreg...", end=' ')
            var_aggreg = make_features_NAS(X=X, pairs=pairs, order=order, var_names=markers)
            if not os.path.exists(aggreg_dir):
                os.makedirs(aggreg_dir)
            var_aggreg.to_csv(file_path, index=False)
            print("done")
        if downsample:
            var_aggreg = var_aggreg.loc[::downsample, :]

        # Dimension reduction for visualization
        title = f"umap_on-{var_type}{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-2"
        file_path = str(save_dir / title) + '.csv'
        if os.path.exists(file_path):
            print("load embed_viz")
            embed_viz = np.loadtxt(file_path, delimiter=',')
        else:
            print("computing embed_viz...", end=' ')
            embed_viz = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=0).fit_transform(var_aggreg)
            np.savetxt(file_path, embed_viz, fmt='%.18e', delimiter=',', newline='\n')
            print("done")
        
        # Dimension reduction for clustering
        for dim_clust in dim_clusts:
            print("    dim_clust: {}".format(dim_clust))
            
            title = f"umap_on-{var_type}{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-{dim_clust}"
            file_path = str(save_dir / title) + '.csv'
            if os.path.exists(file_path):
                print("    load embed_clust_orig...", end=' ')
                embed_clust_orig = np.loadtxt(file_path, delimiter=',')
            else:
                print("    computing embed_clust_orig...", end=' ')
                embed_clust_orig = UMAP(n_neighbors=n_neighbors, **args_reduc).fit_transform(var_aggreg)
                np.savetxt(file_path, embed_clust_orig, fmt='%.18e', delimiter=',', newline='\n')
                print("done")

            for min_cluster_size in min_cluster_sizes:
                print(f"        min_cluster_size: {min_cluster_size}")

                for sampling in [1]:
                    print(f"            sampling: {sampling}", end=', ')
                    # title = f"hdbscan_on-{var_type}_reducer-umap_nas{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-{dim_clust}_min_cluster_size-{min_cluster_size}_sampling-{sampling}"
                    title = f"hdbscan_reducer-umap_nas_on-{var_type}{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-{dim_clust}_min_cluster_size-{min_cluster_size}_sampling-{sampling}"
                    if not os.path.exists(str(save_dir / title) + '.csv'):
                        # downsample embedding
                        select = np.full(embed_clust_orig.shape[0], False)
                        select[::sampling] = True
                        embed_clust = embed_clust_orig[select, :]
                        
                        start = time()
                        # Clustering
                        if soft_clustering:
                            clusterer = hdbscan.HDBSCAN(
                                min_cluster_size=min_cluster_size, 
                                min_samples=None, 
                                prediction_data=True, 
                                core_dist_n_jobs=parallel_clustering,
                                **args_clust,
                            )
                            clusterer.fit(embed_clust)
                            soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
                            if len(soft_clusters.shape) > 1:
                                labels_hdbs = soft_clusters.argmax(axis=1)
                            else:
                                labels_hdbs = soft_clusters
                        else:
                            clusterer = hdbscan.HDBSCAN( 
                                min_cluster_size=min_cluster_size, 
                                min_samples=1,
                                core_dist_n_jobs=parallel_clustering,
                                **args_clust,
                            )
                            clusterer.fit(embed_clust)
                            labels_hdbs = clusterer.labels_
                        nb_clust_hdbs = labels_hdbs.max() # + 1
                        end = time()
                        duration = end-start
                        print("HDBSCAN has detected {} clusters in {:.2f}s".format(nb_clust_hdbs, duration))
                        
                        np.savetxt(str(save_dir / title) + '.csv',
                                    labels_hdbs, 
                                    fmt='%.18e', delimiter=',', newline='\n')
                        title = "clusterer-" + title
                        joblib.dump(clusterer, str(save_dir / title) + '.pkl')

    print("\n\nFinished!\n\n")
    return


def screen_nas_parameters_parallel(
    X, pairs, markers, orders, dim_clusts, min_cluster_sizes, processed_dir, soft_clustering=True, 
    var_type=None, downsample=False, aggreg_dir=None, save_dir=None, opt_str='', 
    parallel_dim='max', parallel_clust_size=False, parallel_clustering=8, memory_limit='50GB',
    opt_args_reduc={}, opt_args_clust={}):
    """
    Try combinations of parameters for the Neighbors Aggregation Statistics method, 
    including the neighbors order of aggregation, the dimensionnality in which the aggregation
    is performed, and the minimum cluster size.
    Results and logs are saved as soon as an intermediate result is produced.
    
    Example
    -------
    processed_dir = Path('../data/processed/CODEX_CTCL')
    opt_str = '_samples-all_stat-mean-std'
    """

    for order in orders:
        print("order: {}".format(order))
        
        # compute statistics on aggregated data across neighbors
        file_path = aggreg_dir / 'nas_on-{var_type}{opt_str}.csv'
        if os.path.exists(file_path):
            var_aggreg = pd.read_csv(file_path)
        else:
            var_aggreg = make_features_NAS(X=X, pairs=pairs, order=order, var_names=markers)
            var_aggreg.to_csv(file_path, index=False)
        if downsample:
            var_aggreg = var_aggreg.loc[::downsample, :]

        # Dimension reduction for visualization
        title = f"umap_on-{var_type}{opt_str}_order-{order}_dim_clust-2"
        file_path = str(save_dir / title) + '.csv'
        if os.path.exists(file_path):
            embed_viz = np.loadtxt(file_path, delimiter=',')
        else:
            embed_viz = UMAP(n_components=2, random_state=0).fit_transform(var_aggreg)
            np.savetxt(file_path, embed_viz, fmt='%.18e', delimiter=',', newline='\n')
        
        # Dimension reduction for clustering
        # select the right number of cores
        nb_cores = cpu_count()
        if isinstance(parallel_dim, int):
            use_cores = min(parallel_dim, nb_cores)
        elif parallel_dim == 'max-1':
            use_cores = nb_cores - 1
        elif parallel_dim == 'max':
            use_cores = nb_cores
        print(f"using {use_cores} cores")
        # # set up cluster and workers
        cluster = LocalCluster(n_workers=use_cores, 
                                threads_per_worker=1,
                                memory_limit=memory_limit)
        client = Client(cluster)
        # client = Client(threads_per_worker=2, n_workers=nb_cores)

        # dummy output for Dask
        lazy_output = []
        for dim_clust in dim_clusts:
            mini_out = delayed(screen_nas_parameters)(
                X, pairs, markers, orders=[order], dim_clusts=[dim_clust], min_cluster_sizes=min_cluster_sizes, 
                processed_dir=processed_dir, soft_clustering=soft_clustering, var_type=var_type, downsample=downsample, 
                aggreg_dir=aggreg_dir, save_dir=save_dir, opt_str=opt_str,parallel_clustering=parallel_clustering) 
                # f'Screening finished for dim_clust {dim_clust}.'
            lazy_output.append(mini_out)
        # evaluate the parallel computation
        # log_final = delayed(np.unique)(lazy_output).compute()
        dask.compute(*lazy_output)
        # lazy_output.compute()
        # close workers and cluster
        client.close()
        cluster.close()

    print("\n\nFinished!\n\n")
    return


def plot_screened_parameters(obj, cell_pos_cols, cell_type_col, orders, dim_clusts, processed_dir,
                             min_cluster_sizes, filter_samples=None, all_edges=None, sampling=False, var_type=None, 
                             n_neighbors=70, downsample=False, aggreg_dir=None, load_dir=None, save_dir=None, opt_str=''):
    """
    
    Example
    -------
    >>> processed_dir = Path('../data/processed/CODEX_CTCL')
    """

    # from skimage import color
    import colorcet as cc
  
    if var_type is None:
        var_type = 'markers'
    if aggreg_dir is None:
        aggreg_dir = processed_dir / "nas"
    if load_dir is None:
        load_dir = aggreg_dir / f"screening_dim_reduc_clustering_nas_on-{var_type}{opt_str}_n_neighbors-{n_neighbors}_downsample-{downsample}"
    if save_dir is None:
        save_dir = load_dir

    sample_ids = obj['Patients'].sort_values().unique()

    # For the visualization
    plots_marker = '.'
    size_points = 10
    if sampling is False:
        sampling = 1

    for order in orders:
        title = f"umap_on-{var_type}{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-2"
        file_path = str(save_dir / title) + '.csv'
        embed_viz = np.loadtxt(file_path, delimiter=',')

        n_cell_types = obj[cell_type_col].unique().size
        palette = sns.color_palette(cc.glasbey, n_colors=n_cell_types).as_hex()
        palette = [mpl.colors.rgb2hex(x) for x in mpl.cm.get_cmap('tab20').colors]

        for dim_clust in dim_clusts:
            print("dim_clust: {}".format(dim_clust))
            for min_cluster_size in min_cluster_sizes:
                print("    min_cluster_size: {}".format(min_cluster_size))

                # title = f"hdbscan_on-{var_type}_reducer-umap_nas{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-{dim_clust}_min_cluster_size-{min_cluster_size}_sampling-{sampling}"
                title = f"hdbscan_reducer-umap_nas_on-{var_type}{opt_str}_order-{order}_n_neighbors-{n_neighbors}_dim_clust-{dim_clust}_min_cluster_size-{min_cluster_size}_sampling-{sampling}"
                labels_hdbs = np.loadtxt(str(load_dir / title) + '.csv', delimiter=',')

                # Histogram of classes
                fig = plt.figure()
                class_id, class_count = np.unique(labels_hdbs, return_counts=True)
                plt.bar(class_id, class_count, width=0.8);
                plt.title('Clusters histogram');
                title = f"Clusters histogram - on {var_type} - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size}"
                plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight', facecolor='white')
                plt.show(block=False)
                plt.close()   
                
                # make a cohort-wide cmap
                hdbs_cmap = cc.palette["glasbey_category10"]
                # make color mapper
                # series to sort by decreasing order
                uniq = pd.Series(labels_hdbs).value_counts().index.astype(int)
                n_colors = len(hdbs_cmap)
                labels_color_mapper = {x: hdbs_cmap[i % n_colors] for i, x in enumerate(uniq)}
                # make list of colors
                labels_colors = [labels_color_mapper[x] for x in labels_hdbs]
                labels_colors = pd.Series(labels_colors)

                for sample in sample_ids:
                    print("        sample: {}".format(sample))
                    select_sample = obj['Patients'] == sample
                    filenames = obj.loc[select_sample, 'FileName'].unique()

                    for filename in filenames:
                        if filter_samples is None or filename in filter_samples:
                            print("            filename: {}".format(filename))
                            select_file = obj['FileName'] == filename
                            select = np.logical_and(select_sample, select_file)

                            # load nodes and edges
                            if isinstance(all_edges, str):
                                file_path = processed_dir / all_edges / f'edges_sample-{filename}.csv'
                                pairs = pd.read_csv(file_path, dtype=int).values
                            else:
                                coords = obj.loc[select, cell_pos_cols].values
                                pairs = ty.build_delaunay(coords)
                                pairs = ty.link_solitaries(coords, pairs, method='knn', k=2)
                            # we drop z for the 2D representation
                            coords = obj.loc[select, ['x', 'y']].values

                            # Big summary plot
                            fig, ax = plt.subplots(1, 4, figsize=(int(7*4)+1, 7), tight_layout=False)
                            i = 0
                            ty.plot_network(coords, pairs, labels=obj.loc[select, 'ClusterName'], cmap_nodes=palette, marker=plots_marker, size_nodes=size_points, ax=ax[0])
                            ax[i].set_title('Spatial map of phenotypes', fontsize=14);

                            i += 1
                            ax[i].scatter(coords[:, 0], coords[:, 1], c=labels_colors[select], marker=plots_marker, s=size_points)
                            ax[i].set_title('Spatial map of detected areas', fontsize=14);
                            ax[i].set_aspect('equal')

                            i += 1
                            ax[i].scatter(embed_viz[select, 0], embed_viz[select, 1], c=labels_colors[select], s=5);
                            ax[i].set_title("HDBSCAN clustering on NAS", fontsize=14);
                            ax[i].set_aspect('equal')

                            i += 1
                            ax[i].scatter(embed_viz[:, 0], embed_viz[:, 1], c=labels_colors);
                            ax[i].set_title("HDBSCAN clustering on NAS of all samples", fontsize=14);
                            ax[i].set_aspect('equal')
                            
                            # make plot limits equal
                            ax[i-1].set_xlim(ax[i].get_xlim())
                            ax[i-1].set_ylim(ax[i].get_ylim())

                            suptitle = f"Spatial omics data and detected areas - mean and std - order {order} - dim_clust {dim_clust} - min_cluster_size {min_cluster_size} - sample {sample} - file {filename}";
                            fig.suptitle(suptitle, fontsize=18)

                            fig.savefig(save_dir / suptitle, bbox_inches='tight', facecolor='white', dpi=200)
                            plt.show(block=False)
                            plt.close()


def make_features_STARGATE(
    X: np.array, 
    pairs: np.array, 
    var_names: Union[Iterable[str], None] = None,
    ) -> pd.DataFrame:
    """
    Compute feature vectors of each node in a network
    given the STARGATE method.

    Parameters
    ----------
    X : array_like
        Nodes' attributes on which features are computed.
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    var_names : list
        Names of variables of X.

    Returns
    -------
    feats : dataframe
        Features computed with the STARGATE method.
    """
    # code here
    pass


def make_cluster_cmap(labels, grey_pos='end', saturated_first=True, as_mpl_cmap=False):
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

    cmap = ['#1F77B4', '#FF7F0E',  '#2CA02C', '#D62728', '#9467BD',
            '#8C564B', '#17BECF', '#E377C2', '#BCBD22', '#7F7F7F']
    if grey_pos == 'start':
        cmap[0], cmap[-1] = cmap[-1], cmap[0]
    if len(labels) > 10:
        cmap_next = ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
                      '#C49C94', '#9EDAE5', '#F7B6D2', '#DBDB8D', '#C7C7C7']
        if grey_pos == 'start':
            cmap_next[0], cmap_next[-1] = cmap_next[-1], cmap_next[0]
        # select as few as lowly saturated colors as possible
        if len(labels) < 20:
            cmap_next = cmap_next[:len(labels)-10]
        if saturated_first:
            cmap = cmap + cmap_next
        else:
            # put the saturated colors at the end
            cmap = cmap_next + cmap
    if len(labels) > 20:
        cmap_end = ['#00FFFF', '#00FF00', '#FF00FF', '#FF007F']
        cmap = cmap + cmap_end
    if as_mpl_cmap:
        # TODO: check if hex convert to tuple of size 3
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap)
    
    return cmap


def make_niches_composition(var, niches, var_label='variable', normalize='total'):
    """
    Make a matrix plot of cell types composition of niches.
    """
    df = pd.DataFrame({var_label: var,
                       'niches': niches})
    df['counts'] = np.arange(df.shape[0])
    counts = df.groupby([var_label, 'niches']).count()
    counts = counts.reset_index().pivot(var_label, 'niches', 'counts').fillna(0)
    if normalize == 'total':
        counts = counts / df.shape[0]
    elif normalize == 'obs':
        # pandas has some unconvenient bradcasting behaviour otherwise
        counts = counts.div(counts.sum(axis=1), axis=0)
    elif normalize == 'niche':
        counts = counts / counts.sum(axis=0)
    elif normalize == 'clr':
        X = counts.values
        # avoid null values
        X[X == 0] = X.max() / 100000
        # CLR tranformation
        X_clr = cs.clr(cs.closure(X))
        counts.loc[:, :] = X_clr
    
    return counts


def plot_niches_composition(counts=None, var=None, niches=None, var_label='variable', normalize='total'):
    """
    Make a matrix plot of cell types composition of niches.
    """
    if counts is None:
        counts = make_niches_composition(var, niches, var_label='variable', normalize=normalize)
    
    plt.figure()
    fig = sns.heatmap(counts, linewidths=.5, cmap=sns.color_palette("Blues", as_cmap=True),
                      xticklabels=True, yticklabels=True)
    return fig


###### Survival and response statistics ######

def clean_data(
        data, 
        method='mixed', 
        thresh=1, 
        cat_cols=None, 
        modify_infs=True, 
        axis=0, 
        verbose=1):
    """
    Delete or impute missing or non finite data.
    During imputation, they are replaced by continuous values, not by binary values.
    We correct them into int values

    Parameters
    ----------
    data : dataframe
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
    axis : int
        Axis over which elements are dropped.
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
            perc_nan = 100 * nb_nan / to_nan.size
            print(f"There are {nb_nan} non finite values ({perc_nan:.1f}%)")
        # set also inf values to nan
        if modify_infs:
            data[to_nan] = np.nan
        # convert proportion threshold into absolute number of variables threshold
        if method in ['drop', 'mixed'] and (0 < thresh <= 1):
            thresh = thresh * data.shape[axis]
        if method in ['drop', 'mixed']:
            # we use a custom code instead of pandas.dropna to return the boolean selector
            count_nans = to_nan.sum(axis=axis)
            select = count_nans <= thresh
            if axis == 0:
                data = data.loc[:, select].copy()
            else:
                data = data.loc[select, :].copy()
        # impute non finite values (nan, +/-inf)
        if method in ['knn', 'mixed']:
            if verbose > 0:
                print('Imputing data')
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            data.loc[:, :] = imputer.fit_transform(data.values)
            # set to intergers the imputed int-coded categorical variables
            # note: that's fine for 0/1 variables, avoid imputing on non binary categories
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
    data : dataframe
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


def extract_X_y(data, y_name, y_values=None, col_names=None, col_exclude=None, binarize=True):
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
    col_exclude : list(str) or None
        Columns to ignore.
    binarize : bool
        If true and `y_values` has 2 elements, the vector `y` is
        binarized, with the 1st and 2nd elements of `y_values`
        tranformed into 0 and 1 respectivelly.
    
    Returns
    -------
    X : dataframe
        Data corresponding to specific target y values.
    y : array
        The y values related to X.
    """

    # if the y variable is in a pandas multiindex:
    if y_name not in data.columns and y_name in data.index.names:
        X = data.reset_index()
    else:
        X = deepcopy(data)
    if y_values is None:
        y_values = X[y_name].unique()
    if col_exclude is None:
        col_exclude = []
    col_exclude.append(y_name)
    if col_names is None:
        col_names = [x for x in data.columns if x not in col_exclude]

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
    When producing ratios of variables, ratios of identical variables and
    reverse ratios are avoided, e.g. a/b, but no a/a nor b/a.
    When producing ratios of ratios of variables (order=2), equivalent and
    inverse ratios are avoided, e.g. (a/b)/(c/d), but no (a/b)/(a/d), and
    no (a/c)/(b/d).

    Example
    -------
    >>> df = pd.DataFrame({
            'a': [24, 24, 24],
            'b': [12, 8, 8],
            'c': [6, 4, 3],
            'd': [3, 4, 1],
        })
    >>> mosna.make_composed_variables(df)
       a / b  a / c  a / d     b / c  b / d  c / d  (a / b) / (c / d)
    0    2.0    4.0    8.0  2.000000    4.0    2.0                1.0
    1    3.0    6.0    6.0  2.000000    2.0    1.0                3.0
    2    3.0    8.0   24.0  2.666667    8.0    3.0                1.0                   
    """
    
    if use_col is None:
        use_col = data.columns
    if method == 'ratio':
        # ratio of variables
        new_vars = {}
        for i, var_1 in enumerate(use_col):
            for var_2 in use_col[i+1:]:
                new_var_name = f"{var_1} / {var_2}"
                new_vars[new_var_name] = data[var_1] / data[var_2]
        new_data = pd.DataFrame(data=new_vars)
    
        if order == 2:
            # ratios of ratios of variables
            new_vars = {}
            for i, var_1 in enumerate(use_col):
                for j, var_2 in enumerate(use_col[i+1:]):
                    for k, var_3 in enumerate(use_col[i+j+2:]):
                        for var_4 in use_col[i+j+k+3:]:
                            pair_1 = [var_1, var_2]
                            pair_2 = [var_3, var_4]
                            new_var_name = f"({var_1} / {var_2}) / ({var_3} / {var_4})"
                            new_vars[new_var_name] = (data[var_1] / data[var_2]) / (data[var_3] / data[var_4])
            next_data = pd.DataFrame(data=new_vars)
            new_data = pd.concat([new_data, next_data], axis=1)

    return new_data


def find_DE_markers(data, group_ref, group_tgt, group_var, markers=None, exclude_vars=None, composed_vars=False, 
                    composed_order=2, test='Mann-Whitney', fdr_method='indep', alpha=0.05):
    

    if composed_vars:
        data = pd.concat([data, make_composed_variables(data, order=composed_order)], axis=1)
    if markers is None:
        markers = data.columns
    if isinstance(group_var, str):
        if group_var in data.columns:
            if exclude_vars is None:
                exclude_vars = [group_var]
            else:
                exclude_vars = exclude_vars + [group_var]
            group_var = data[group_var].values
        elif group_var in data.index.names:
            group_var = data.index.to_frame()[group_var]
        else:
            raise ValueError('The name of the group variable is not in columns nor in the index.')

    select_tgt = group_var == group_tgt
    if group_ref == 'other':
        select_ref = group_var != group_tgt
    elif not isinstance(group_ref, list):
        select_ref = group_var == group_ref
    else:
        select_ref = group_var == group_ref[0]
        for ref_id in group_ref[1:]:
            select_ref = np.logical_or(select_ref, group_var == ref_id)
    if isinstance(select_tgt, pd.Series):
        select_tgt = select_tgt.values
        select_ref = select_ref.values

    pvals = []
    # filter variable_names if exclude_vars was given
    if exclude_vars is not None:
        markers = [x for x in markers if x not in exclude_vars]
    used_markers = []
    for marker in markers:
        dist_tgt = data.loc[select_tgt, marker].dropna()
        dist_ref = data.loc[select_ref, marker].dropna()
        # select = np.logical_and(np.isfinite(dist_tgt), np.isfinite(dist_ref))
        # dist_tgt = dist_tgt[select]
        # dist_ref = dist_ref[select]
        if len(dist_tgt) > 0 and len(dist_ref) > 0:
            if test == 'Mann-Whitney':
                mwu_stat, pval = mannwhitneyu(dist_tgt, dist_ref)
            if test == 'Welch':
                w_stat, pval = ttest_ind(dist_tgt, dist_ref, equal_var=False)
            if test == 'Kolmogorov-Smirnov': 
                ks_stat, pval = ks_2samp(dist_tgt, dist_ref)
            pvals.append(pval)
            used_markers.append(marker)
    pvals = pd.DataFrame(data=pvals, index=used_markers, columns=['pval'])
    pvals = pvals.sort_values(by='pval', ascending=True)

    if fdr_method is not None:
        rejected, pval_corr = fdrcorrection(pvals['pval'], method=fdr_method)
        pvals['pval_corr'] = pval_corr
    
    return pvals


def plot_distrib_groups(data, group_var, groups=None, pval_data=None, pval_col='pval_corr', pval_thresh=0.05, 
                        max_cols=-1, exclude_vars=None, id_vars=None, var_name='variable', value_name='value', 
                        multi_ind_to_col=False, figsize=(20, 6), fontsize=20, orientation=30, ax=None,
                        plot_type='boxplot', add_points=True):
    """
    Plot the distribution of variables by groups.
    """

    # Select variables that will be plotted
    if groups is None:
        groups = data[group_var].unique()
    if len(groups) == 2 and pval_data is not None:
        if isinstance(pval_data, str) and pval_data == 'compute':
            pval_data = find_DE_markers(data, groups[0], groups[1], group_var=group_var, composed_order=0)
        nb_vars = np.sum(pval_data[pval_col] <= pval_thresh)
        print(f'There are {nb_vars} significant variables in `{pval_col}`')
        if nb_vars == 0:
            nb_vars = len(pval_data)
        if max_cols > 0:
            nb_vars = min(nb_vars, max_cols)
        marker_vars = pval_data.sort_values(by=pval_col, ascending=True).head(nb_vars).index.tolist()
    else:
        marker_vars = data.columns.tolist()
        if max_cols > 0:
            marker_vars = marker_vars[:max_cols]
    # filter variable_names if exclude_vars was given
    if group_var in data.columns:
        gp_in_cols = [group_var] # exclude column of groups anyway
        if exclude_vars is None:
            exclude_vars = [group_var]
        else:
            exclude_vars = exclude_vars + [group_var]
    else:
        gp_in_cols = []
    if exclude_vars is not None:
        marker_vars = [x for x in marker_vars if x not in exclude_vars]
    
    # TODO: utility function to put id variables in multi-index into columns if not already in cols
    wide = data.loc[:, gp_in_cols + marker_vars]
    if id_vars is None:
        list_id_vars = list(wide.index.names) + gp_in_cols
        id_vars = [x for x in list_id_vars if x is not None]
    if multi_ind_to_col:
        wide = wide.reset_index()

    # select desired groups
    select = np.any([wide[group_var] == i for i in groups], axis=0)
    wide = wide.loc[select, :]

    long = pd.melt(
        wide, 
        id_vars=id_vars, 
        value_vars=marker_vars,
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
    
    # TODO: display variables on different axes if the have very differents ranges
    if plot_type == 'boxplot':
        sns.boxplot(x=var_name, y=value_name, hue=group_var, 
                    data=long, palette="Set2", ax=ax);
    elif plot_type == 'violinplot':
        sns.violinplot(x=var_name, y=value_name, hue=group_var, 
                       data=long, palette="Set2", split=split, ax=ax);
    if add_points:
        sns.stripplot(long, x=var_name, y=value_name, hue=group_var, 
                      dodge=True, size=4, palette='dark:.3', legend=None);
    plt.xticks(rotation=orientation, ha='right', fontsize=fontsize);
    plt.yticks(fontsize=fontsize);
    if ax_none:
        return fig, ax


def plot_heatmap(data, obs_labels=None, group_var=None, groups=None, 
                 use_col=None, skip_cols=[], z_score=1, drop_unique=True, 
                 cmap=None, center=None, row_cluster=True, col_cluster=True,
                 palette=None, figsize=(10, 10), fontsize=10, 
                 colors_ratio=0.03, dendrogram_ratio=0.2, cbar_kws=None,
                 cbar_pos=(0.02, 0.8, 0.05, 0.18),
                 xlabels_rotation=30, ax=None, return_data=False):

    data = data.copy(deep=True)
    if obs_labels is not None:
        data.index = data[obs_labels]
        data.drop(columns=[obs_labels], inplace=True)
    if use_col is None:
        skip_cols = skip_cols + [obs_labels, group_var]
        use_col = [x for x in data.columns if x not in skip_cols]
    else:
        data = data[use_col]
    if drop_unique:
        drop_cols = []
        keep_cols = []
        for col in use_col:
            n_uniq = len(data[col].unique())
            if n_uniq > 0:
                keep_cols.append(col)
            else:
                drop_cols.append(col)
        if len(drop_cols) > 0:
            print("Dropping colunms with unique value:")
            print(drop_cols)
            data = data.loc[:, keep_cols]
            use_col = keep_cols

    if group_var is not None:
        if groups is None:
            groups = data[group_var].unique()        
        # select desired groups
        data = data.query(f'{group_var} in @groups')
        # make lut group <--> color
        if palette is None:
            palette = sns.color_palette()
        lut = dict(zip(groups, palette))
        # Make the vector of colors
        colors = data[group_var].map(lut)
        data.drop(columns=[group_var], inplace=True)
    else:
        colors = None
    
    if cmap is None:
        if z_score is not None or (data.values.min() < 0 and data.values.max() > 0):
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            center = 0
        else:
            cmap = sns.light_palette("#C25539", as_cmap=True)
            center = None
    g = sns.clustermap(data, z_score=z_score, figsize=figsize, 
                       row_colors=colors, cmap=cmap, center=center,
                       row_cluster=row_cluster, col_cluster=col_cluster,
                       colors_ratio=colors_ratio, dendrogram_ratio=dendrogram_ratio,
                       cbar_pos=cbar_pos, cbar_kws=cbar_kws)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=xlabels_rotation, ha='right', fontsize=fontsize);
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=fontsize);
    if hasattr(g, 'ax_row_colors') and colors is not None:
        g.ax_row_colors.set_xticklabels(g.ax_row_colors.get_xticklabels(), rotation=xlabels_rotation, ha='right', fontsize=fontsize);
    if return_data:
        return g, data
    else:
        return g


def color_val_inf(val, thresh=0.05, col='green', col_back='white'):
    """
    Takes a scalar and returns a string with
    the css property `'color: green'` for values
    below a threshold, black otherwise.
    """
    color = col if val < thresh else col_back
    return 'color: %s' % color


def get_significant_coefficients(
    df: pd.DataFrame, 
    lower_col: str = '95% lower-bound', 
    upper_col: str = '95% upper-bound', 
    variables_in: str = 'index') -> Iterable[str]:
    """"
    Pick variables with significant coefficients in a predictive model.

    Parameters
    ----------
    df : dataframe
        Coefficients with lower and upper confidence intervals.
    lower_col : str
        Column storing the lower bound
    upper_col : str
        Column storing the upper bound
    variables_in : str
        If 'index', variables are selected from the index, otherwise
        they are selected from the column indicated by variables_in.
    
    Returns
    -------
    variables : Iterable[str]
        Variables with significant coefficients.
    """
    select = np.logical_or(
        np.logical_and(df[lower_col] < 0, df[upper_col] < 0),
        np.logical_and(df[lower_col] > 0, df[upper_col] > 0))
    if variables_in == 'index':
        variables = df.index.values[select]
    else:
        variables = df.loc[select, variables_in]
    return variables


def find_best_survival_threshold(
    data: pd.DataFrame, 
    variable_name: str, 
    duration_col: str, 
    event_col: str, 
    perc_range: Tuple[int, int, int] = (10, 91, 10)
    ) -> Tuple[float, int, float]:
    """
    Find the threshold that minimizes the p-value of the log-rank test.

    Parameters
    ----------
    data : pd.DataFrame
        Survival data.
    variable_name : str
        Column to use in data. 
    duration_col : str
        Column for survival duration.
    event_col : str
        Column for survival event (death).
    perc_range : Tuple[int, int, int]
        Parameters to generate the percentiles used as potential thresholds.
    
    Returns
    -------
    best_thresh : float
        Best threshold.
    best_perc : int
        Best percentile.
    best_pval : float
        Best p-value. 
    """
    
    variable = data[variable_name]
    T = data[duration_col]
    E = data[event_col]
    
    all_perc = np.arange(*perc_range)
    all_thresh = []
    all_pvals = []
    for perc in all_perc:
        thresh = np.percentile(variable, perc)
        select = (variable > thresh)
        p_val = logrank_test(T[select], T[~select], E[select], E[~select], alpha=.99).p_value
        all_thresh.append(thresh)
        all_pvals.append(p_val)

    best_idx = np.argmin(all_pvals)
    best_thresh = all_thresh[best_idx]
    best_perc = all_perc[best_idx]
    best_pval = all_pvals[best_idx]

    return best_thresh, best_perc, best_pval


def plot_survival_threshold(
    data: pd.DataFrame, 
    variable_name: str, 
    duration_col: str, 
    event_col: str, 
    thresh: float, 
    with_confidence: bool = True,
    ax: plt.Axes = None
    ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Kaplan-Meier curves of observations discriminated by a threshold.

    Parameters
    ----------
    data : pd.DataFrame
        Survival data.
    variable_name : str
        Column to use in data. 
    duration_col : str
        Column for survival duration.
    event_col : str
        Column for survival event (death).
    thresh : float
        Threshold applied on variable.
    with_confidence : bool
        If True, KM curves are plotted with estimated confidence intervals.
    ax : plt.Axes
        Existing pyplot ax if any to draw KM curves.
    
    Returns
    -------
    fig : plt.Figure
        Figure of KM curves.
    ax : plt.Axes
        Axes of KM curves.
    """
    
    kmf_1 = KaplanMeierFitter()
    kmf_2 = KaplanMeierFitter()

    variable = data[variable_name]
    T = data[duration_col]
    E = data[event_col]
    select = (variable > thresh)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    kmf_1.fit(T[select], event_observed=E[select], label=f">   {thresh:.3g}")
    kmf_2.fit(T[~select], event_observed=E[~select], label=f"<= {thresh:.3g}")
    if with_confidence:
        kmf_1.plot_survival_function(ax=ax)
        kmf_2.plot_survival_function(ax=ax)
    else:
        kmf_1.survival_function_.plot(ax=ax)
        kmf_2.survival_function_.plot(ax=ax)

    ax.set_title(f"Survival given {variable_name}")
    return fig, ax


def find_survival_variable(surv, X, reverse_response=False, return_table=True, return_model=True, model_kwargs=None, model_fit=None):
    """
    Fit a CoxPH model for each single variable, and detect the ones
    that are statistically significant.
    """
    pass


def get_reducer(data, save_dir, reducer_type='umap', n_components=2, 
                n_neighbors=15, metric='manhattan', min_dist=0.0, 
                save_reducer=False, random_state=None, verbose=1):
    """
    Generate or load a dimensionality reduction (DR) model and transformed (reduced) data.

    Parameters
    ----------
    data : ndarray
        Dataset on which we want to apply the DR method.
    save_dir : str or pathlib Path object
        Directory where the DR model and transformed data are stored.
    reducer_type : str
        DR method, can be 'umap', for now, other ones coming soon.
    n_components : int
        Number of final dimensions.
    n-neighbors : int
        Number of closest neighbors used in various DR methods.
    metric : str
        Type of distance used, like 'manhattan', 'euclidean' or 'cosine'.
    min_dist : float
        Minimum distance between DRed data, we usually want 0.
    save_reducer : bool
        Whether the reducer object is saved. 
    random_state : int
        Controls the random initialization of the DR model.
    
    Returns
    -------
    embedding : ndarray
        Reduced coordinates of the dataset.
    reducer : object
        The DR model, its type depends on the choosen DR method.

    Example
    -------
    In the *mosna* pipeline, `var_aggreg` is the array of aggregated statistics:
    >>> embedding, reducer = get_reducer(data=var_aggreg, save_dir=nas_dir)
    """

    reducer_name = f"reducer-{reducer_type}_dim-{n_components}_nneigh-{n_neighbors}_metric-{metric}_min_dist-{min_dist}"
    save_dir = Path(save_dir) / reducer_name
    file_path = save_dir / "embedding"
    if os.path.exists(str(file_path) + '.npy'):
        if verbose > 0: print("Loading reducer object and reduced coordinates")
        embedding = np.load(str(file_path) + '.npy')
        if os.path.exists(str(save_dir / "reducer") + '.pkl'):
            reducer = joblib.load(str(save_dir / "reducer") + '.pkl')
        else:
            reducer = None
    else:
        if verbose > 0: print("Computing dimensionality reduction")
        if reducer_type == 'umap':
            reducer = UMAP(
                random_state=random_state,
                n_components=n_components,
                n_neighbors=n_neighbors,
                metric=metric,
                min_dist=min_dist,
                )
        embedding = reducer.fit_transform(data)
        # save reduced coordinates
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(file_path) + '.npy', embedding, allow_pickle=False, fix_imports=False)
        if save_reducer:
            # save the reducer object
            joblib.dump(reducer, str(save_dir / "reducer") + '.pkl')
    
    return embedding, reducer


def get_clusterer(
        data,
        save_dir,
        reducer_type='umap', 
        n_neighbors=15, 
        metric='manhattan', 
        min_dist=0.0,
        clusterer_type='leiden', 
        dim_clust=2, 
        k_cluster=15, 
        resolution_parameter=0.005,
        force_recompute=False, 
        verbose=1,
        ):
    """
    Generate or load a clustering model and cluster labels.

    Parameters
    ----------
    data : ndarray
        Dataset on which we want to apply the DR method.
    save_dir : str or pathlib Path object
        Directory where the DR model and transformed data are stored.
    reducer_type : str
        DR method, can be 'umap', for now, other ones coming soon.
    n_components : int
        Number of final dimensions.
    n-neighbors : int
        Number of closest neighbors used in various DR methods.
    metric : str
        Type of distance used.
    min_dist : float
        Minimum distance between DRed data, we usually want 0.
    clusterer_type : str
        Clustering algorithm to partition data.
    dim_clust : int
        Dimensionality of the reducede space in which data is clustered.
        A higher number allows for more complex cluster shapes, but introduces outliers.
    k_cluster : int
        Number of neighbors considered during the clustering.
    resolution_parameter : float
        Level of details of the clustering. A higher number increases the level of details.
    force_recompute : bool
        Whether computation occurs even if results already exist in `save_dir`.
    
    Returns
    -------
    embedding : ndarray
        Reduced coordinates of the dataset.
    reducer : object
        The DR model, its type depends on the choosen DR method.

    Example
    -------
    >>> np.random.seed(0)
    >>> data = np.random.rand(800, 4)
    >>> cluster_labels, cluster_dir, nb_clust, G = get_clusterer(data, "test")
    """

    reducer_name = f"reducer-{reducer_type}_dim-{dim_clust}_nneigh-{n_neighbors}_metric-{metric}_min_dist-{min_dist}"
    reducer_dir = Path(save_dir) / reducer_name
    cluster_dir = reducer_dir / f"clusterer-{clusterer_type}_n_neighbors-{k_cluster}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if clusterer_type == "leiden":
        clusterer_name = f"partition-{'RBConfigurationVertexPartition'}_resolution-{resolution_parameter}"
    file_path = cluster_dir / clusterer_name

    if os.path.exists(str(file_path) + '_labels.npy') and not force_recompute:
        if verbose > 0: print("Loading clusterer object and cluster labels")
        cluster_labels = np.load(str(file_path) + '_labels.npy')
        if clusterer_type == "leiden":
            G = joblib.load(str(file_path) + '_network.ig')
        #     partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution_parameter, seed=0)
        #     # extract labels from partition
        #     cluster_labels = np.array(partition.membership)
        nb_clust = cluster_labels.max()
        if verbose > 0: print(f"There are {nb_clust} clusters")
    else:
        if verbose > 0: print("Performing clustering")
        # get the embedding of data
        embedding, _ = get_reducer(data, save_dir, reducer_type, dim_clust, n_neighbors, metric, min_dist, verbose=verbose)
        if clusterer_type == "leiden":
            # build knn graph
            embedding_pairs = ty.build_knn(embedding, k=k_cluster)
            # convert into iGraph object
            G = ty.to_iGraph(embedding, embedding_pairs)
            # perform clustering
            partition = la.find_partition(G, la.RBConfigurationVertexPartition, resolution_parameter=resolution_parameter, seed=0)
            # partition = la.find_partition(G, la.RBERVertexPartition, resolution_parameter=resolution_parameter)
            cluster_labels = np.array(partition.membership)
        nb_clust = cluster_labels.max()
        if verbose > 0: print(f"Found {nb_clust} clusters")
        # save cluster labels
        np.save(str(file_path) + '_labels.npy', cluster_labels, allow_pickle=False, fix_imports=False)
    if clusterer_type == "leiden":
        # save the iGraph object
        joblib.dump(G, str(file_path) + '_network.ig')
        return cluster_labels, cluster_dir, nb_clust, G


def relabel_clusters(
    clusters: np.array,
) -> np.array:
    """
    Relabel the N clusters to have ids between 0 and N-1.

    Parameters
    ----------
    clusters : np.array
        Cluster labels.
    
    Returns
    -------
    new_clusters : np.array
        Potentially relabelled clusters.
    """
    bins, counts = np.unique(clusters, return_counts=True)
    if len(bins) == bins.max() - 1:
        return clusters
    else:
        new_bins = np.arange(len(bins))
        new_clusters = np.zeros_like(clusters)
        for i in range(len(new_bins)):
            new_clusters[clusters == bins[i]] = new_bins[i]
        return new_clusters


def merge_clusters(
    clusters: np.array,
    coords: np.ndarray,
    size_thresh: Union[int, None] = None,
    size_perc: int = 25,
    ratio_size: float = 0.1,
    n_neigh_max: int = 10,
    force_merge: bool = False,
    verbose: int = 1,
) -> Tuple[np.array, bool]:
    """
    Merge the smallest cluster to it's closest cluster if its size
    is lower than a given size.

    Parameters
    ----------
    clusters : np.array
        Cluster labels.
    coords : np.ndarray
        Coordinates of points in clusters.
    size_thresh : Union[int, None], None
        If provided, the size threshold to merge the smallest cluster.
        If none, computed as the base size * ratio_size
    size_perc : int, 25
        Percentile of cluster size as the base size used for the size threshold.
    ratio_size : float, 0.1
        Ratio to compute size_thresh.
    n_neigh_max : int, 10
        Number of points to consider inside the cluster to merge for the closest neighbor.
    force_merge : bool, False
        If True, the smallest cluster is merged even if it is big enough.
    verbose : int, 1
        Verbosity level.
    
    Returns
    -------
    clusters : np.array
        Potentially merged clusters.
    merged : bool
        Whether a merge occured.
    """

    merged = False
    cluster_ids, cluster_sizes = np.unique(clusters, return_counts=True)
    smallest_id = np.argmin(cluster_sizes)
    if size_thresh is None:
        size_thresh = np.percentile(cluster_sizes, size_perc) * ratio_size
    if force_merge or cluster_sizes[smallest_id] < size_thresh:
        select = clusters == cluster_ids[smallest_id]
        coords_in = coords[select, :]
        coords_out = coords[~select, :]
        clusters_out = clusters[~select]
        n_neigh_max = min(n_neigh_max, cluster_sizes[smallest_id])

        kdt = cKDTree(coords_out)
        # closest point id
        dist, pairs = kdt.query(x=coords_in, k=1)

        # Get the closest points to another cluster (avoid inner points)
        closest_neigh_ids = np.argsort(dist)[:n_neigh_max]
        closest_clusters = clusters_out[pairs][closest_neigh_ids]
        val, counts = np.unique(closest_clusters, return_counts=True)
        closest_cluster = val[np.argmax(counts)]
        # make a copy of clusters
        clusters = np.array(clusters)
        clusters[select] = closest_cluster
        if verbose:
            print(f'cluster {cluster_ids[smallest_id]} merged with cluster {closest_cluster}')
        merged = True
    return clusters, merged


def merge_clusters_until(
    clusters: np.array,
    coords: np.ndarray,
    cond_n_clust: Union[int, None] = None,
    force_n_clust: bool = False,
    size_thresh: Union[int, None] = None,
    size_perc: int = 25,
    ratio_size: float = 0.1,
    n_neigh_max: int = 10,
    relabel_clusters_ids: bool = True,
    verbose: int = 1,
) -> np.array:
    """
    Merge iteratively the smallest cluster to it's closest cluster 
    if its size is lower than a given size, until no further merging
    occurs or until a condition is reached.

    Parameters
    ----------
    clusters : np.array
        Cluster labels.
    coords : np.ndarray
        Coordinates of points in clusters.
    cond_n_clust: Union[int, None], None
        Sufficient condition on the number of clusters below which the iterative
        merge stops.
    force_n_clust : bool, False
        If True, force merging until the desired number of clusters is reached.
        It overides the 'until no further merging occurs' condition.
    size_thresh : Union[int, None], None
        If provided, the size threshold to merge the smallest cluster.
        If none, computed as the base size * ratio_size
    size_perc : int, 25
        Percentile of cluster size as the base size used for the size threshold.
    ratio_size : float, 0.1
        Ratio to compute size_thresh.
    n_neigh_max : int, 10
        Number of points to consider inside the cluster to merge for the closest neighbor.
    relabel_clusters_ids : bool, True
        If True, relabel N clusters between 0 and N-1.
    verbose : int, 1
        Verbosity level.
    
    Returns
    -------
    clusters : np.array
        Potentially merged clusters.
    """

    keep_merging = True
    while keep_merging:
        clusters, merged = merge_clusters(
            clusters=clusters,
            coords=coords,
            size_thresh=size_thresh,
            size_perc=size_perc,
            ratio_size=ratio_size,
            n_neigh_max=n_neigh_max,
            force_merge=force_n_clust,
            verbose=verbose,
        )

        if not merged and not force_n_clust:
            keep_merging = False
            if verbose > 0:
                print('no further merging can occur')
        else:
            bins = np.unique(clusters)
            if cond_n_clust is not None and len(bins) <= cond_n_clust:
                keep_merging = False
                if verbose > 0:
                    print(f'maximum number of clusters {cond_n_clust} reached')
    if relabel_clusters_ids:
        clusters = relabel_clusters(clusters)
    return clusters




def plot_clusters(embed_viz, 
                  cluster_labels=None, 
                  save_dir=None, 
                  cluster_params=None, 
                  extra_str='', 
                  show_id=True,
                  cluster_colors=None,
                  aspect='equal',
                  return_cmap=False, 
                  figsize=(10,10),
                  ):
    """
    Plots clustered data on its 2D projection.

    Parameters
    ----------
    embed_viz : ndarray
        Mx2 array of points coordinates in the 2D projection.
    cluster_labels : array
        Cluster label ids of data points.
    save_dir : str or pathlib Path object
        Directory where the vizualisation is stored.
    cluster_params : dic
        Parameters used to generate the 2D projection and clustering that are included in the file name.
    extra_str : str
        Additional string to add in the file name to indicate manual curation
        of clustering for instance.
    
    Returns
    -------
    fig, ax : matplotlib figure objects.
    """

    if cluster_colors is None:
        if cluster_labels is not None:
            nb_clust = cluster_labels.max()
            uniq = pd.Series(cluster_labels).value_counts().index

            # choose colormap
            clusters_cmap = make_cluster_cmap(uniq)
            # make color mapper
            # series to sort by decreasing order
            n_colors = len(clusters_cmap)
            labels_color_mapper = {x: clusters_cmap[i % n_colors] for i, x in enumerate(uniq)}
            # labels_color_mapper = {x: clusters_cmap[i] for i, x in enumerate(uniq)}
            # labels_color_mapper[-1] = (0.5, 0.5, 0.5)
            cluster_colors = [labels_color_mapper[x] for x in cluster_labels]
        else:
            cluster_colors = 'royalblue'

    fig, ax = plt.subplots(figsize=figsize)
    plt.scatter(embed_viz[:, 0], embed_viz[:, 1], c=cluster_colors, marker='.');
    plt.axis('off')
    if aspect == 'equal':
        ax.set_aspect('equal')

    if cluster_labels is not None and show_id:
        for clust_id in np.unique(cluster_labels):
            clust_targ = cluster_labels == clust_id
            x_mean = embed_viz[clust_targ, 0].mean()
            y_mean = embed_viz[clust_targ, 1].mean()
            plt.text(x_mean, y_mean, str(clust_id))

    if save_dir is not None:
        if cluster_params is None:
            str_params = ''
        else:
            str_params = '_' + '_'.join([str(key) + '-' + str(val) for key, val in cluster_params.items()])
        figname =  f'cluster_labels{str_params}{extra_str}.png'
        plt.savefig(save_dir / figname, dpi=150)

    if return_cmap:
        return fig, ax, labels_color_mapper
    return fig, ax


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


def logistic_regression(
    data, 
    y=None,
    y_name=None,
    y_values=None,
    col_drop=None, 
    cv_train=5, 
    cv_adapt=True, 
    cv_max=10, 
    l1_ratios_list='auto', 
    dir_save=None,
    plot_coefs=True,
    save_coefs=False,
    save_scores=False,
    save_plot_figures=False,
    figsize=(8, 8),
    verbose=1,
    ):
    """
    Train logistic regression models looking for the best hyperparameters
    for the ElasticNet penalization, and dislay or save results and models.

    Parameters
    ----------
    data : DataFrame
        Table containing predictive variables.
    y : array_like, optional
        Response / target variable.
    y_name : str, optional
        If `y`is not provided, it is used to extract the response from `data`.
    y_values : list, optional
        List of accepted conditions to extract observations
    col_drop : iterable, optional
        Columns to ignore in `data`

    Returns
    -------
    models : dict
        Record of scikit-learn's models and their associated performance and
        coefficients for each l1 ratios list.
    
    Example
    -------
    >>> col_drop = ['Patients', 'Spots']
    >>> y_name = 'Groups'
    """

    # Elasticnet logistic regression
    # l1_ratio = 0 the penalty is an L2 penalty (Ridge)
    # l1_ratio = 1 the penalty is an L1 penalty (Lasso)

    # Test either one of those combinations
    if l1_ratios_list == 'auto':
        l1_ratios_list = [
            ['default', [0.5]],
            ['naive', np.linspace(0, 1, 21)],           # naive param grid
            ['advised', [.1, .5, .7, .9, .95, .99, 1]], # advised in scikit-learn documentation
        ]

    score_labels = [
        'ROC AUC', # Receiver Operating Characteristic Area Under the Curve
        'AP',      # Average Precision
        'MCC',     # Matthews Correlation Coefficient
    ]

    # /!\ not related to `X = obj[aggreg_vars].values`
    if y is None:
        X, y = extract_X_y(data, y_name, y_values)

    X = data.reset_index()
    for col in col_drop:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
    X = X.loc[(X.loc[:, y_name] == 1) | (X.loc[:, y_name] == 2), :]
    y = X[y_name].values - 1  # to have resp=0, non-resp=1
    X = X.drop(columns=[y_name])
    var_idx = X.columns
    # X = X.values

    start = time()

    models = {}
    for l1_name, l1_ratios in l1_ratios_list:

        score_split = {}
        # stratify train / test by response
        np.random.seed(0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.25, 
            random_state=0, 
            shuffle=True, 
        )
        # Standardize data to give same weight to regularization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        training_succeeded = False
        cv_used = cv_train
        while not training_succeeded and cv_used <= cv_max:
            np.random.seed(0)
            clf = linear_model.LogisticRegressionCV(
                cv=cv_used,
                Cs=20, 
                penalty='elasticnet', 
                # scoring='neg_log_loss', 
                scoring='roc_auc', 
                solver='saga', 
                l1_ratios=l1_ratios,
                max_iter=10000,
                n_jobs=-1,  # or n_jobs-1 to leave one core available
            )
            clf = clf.fit(X_train, y_train)
            training_succeeded = not np.all(clf.coef_ == 0)
            if not training_succeeded:
                if cv_adapt:
                    cv_used += 1
                    print(f"        training failed, trying with cv = {cv_used}")
                else:
                    print(f"        training failed")
                    break
            
        if training_succeeded:
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)
            score = {
                'ROC AUC': metrics.roc_auc_score(y_test, y_pred_proba),
                'AP' : metrics.average_precision_score(y_test, y_pred_proba),
                'MCC': metrics.matthews_corrcoef(y_test, y_pred),
            }

            # Save model coefficients and plots
            l1_ratio = np.round(clf.l1_ratio_[0], decimals=4)
            C = np.round(clf.C_[0], decimals=4)

            coef = pd.DataFrame({'coef': clf.coef_.flatten()}, index=var_idx)
            coef['abs coef'] = coef['coef'].abs()
            coef = coef.sort_values(by='abs coef', ascending=False)
            coef['% total'] = coef['abs coef'] / coef['abs coef'].sum()
            coef['cum % total'] = coef['% total'].cumsum()
            coef['coef OR'] = np.exp(coef['coef'])
            nb_coef = coef.shape[0]
            if save_coefs:
                coef.to_csv(dir_save / f"LogisticRegressionCV_coefficients.csv")
            
            if plot_coefs:
                nb_coef_plot = min(20, nb_coef)
                labels = coef.index[:nb_coef_plot]

                fig, ax = plt.subplots(figsize=figsize)
                ax = coef.loc[labels, 'coef'].to_frame().plot.bar(ax=ax)
                ax.hlines(y=0, xmin=0, xmax=nb_coef_plot-1, colors='gray', linestyles='dashed')
                ticks_pos = np.linspace(start=0, stop=nb_coef_plot-1, num=nb_coef_plot)
                # ticks_label = np.round(ticks_label, decimals=2)
                ax.set_xticks(ticks_pos);
                # ax.set_xticklabels(ticks_label)
                ax.set_xticklabels(labels, rotation=45, ha='right');
                ax.set_xlabel('variables')
                ax.set_ylabel('coef')
                ax.set_title(f" l1_ratio {l1_ratio}, C {C}, AUC {score['ROC AUC']}")
                if save_plot_figures:
                    fig.savefig(dir_save / f"coef.png", bbox_inches='tight', facecolor='white')

        else:
            score = {
                'ROC AUC': np.nan,
                'AP' : np.nan,
                'MCC': np.nan,
            }
            coef = None
            print(f"        training failed with cv <= {cv_max}")
                
        models[l1_name] = {
            'model': clf,
            'score': score,
            'coef': coef,
        }

        scores = pd.DataFrame.from_dict(score, orient='index')
        if save_scores:
            scores.to_csv(dir_save / 'scores_niches.csv')

    end = time()
    duration = end - start
    if verbose > 0:
        print(f"Training took {duration:.3f}s")

    return models


def train_XGBoost(
    data: pd.DataFrame, 
    y: Iterable = None,
    ) -> xgboost.core.Booster:
    """
    Train an XGBoost model.

    Parameters
    ----------
    data : DataFrame
        Table containing predictive variables.
    y : array_like, optional
        Response / target variable.

    Returns
    -------
    model : xgboost
        Trained XGBoost model
    """
    # training code here
    pass


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