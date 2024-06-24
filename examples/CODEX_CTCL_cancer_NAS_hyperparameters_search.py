# For WIP and avoid re-executing individual cells, to be deleted...

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import time
import joblib
from pathlib import Path
from time import time
from tqdm import tqdm
import copy
import matplotlib as mpl
import napari
import colorcet as cc
import composition_stats as cs
from sklearn.impute import KNNImputer

from tysserand import tysserand as ty
from mosna import mosna

import matplotlib as mpl
mpl.rcParams["figure.facecolor"] = 'white'
mpl.rcParams["axes.facecolor"] = 'white'
mpl.rcParams["savefig.facecolor"] = 'white'

# If need to reload modules after their modification
from importlib import reload
ty = reload(ty)
mosna = reload(mosna)

print("Loading and processing data")

data_dir = Path("../data/raw/CODEX_CTCL")
try:
    objects_path = data_dir / "41467_2021_26974_MOESM3_ESM_-_Objects.parquet"
    obj = pd.read_parquet(data_dir / "41467_2021_26974_MOESM3_ESM_-_Objects.parquet")
except FileNotFoundError:
    objects_path = data_dir / "41467_2021_26974_MOESM3_ESM_-_Objects.xlsx"
    obj = pd.read_excel(objects_path, skiprows=2)
    # for latter use
    obj.to_parquet(data_dir / "41467_2021_26974_MOESM3_ESM_-_Objects.parquet")

# sample_cols = obj.columns[:10]
obj.rename(columns={'X': 'x', 'Y': 'y', 'Z':'z', 'ClusterName': 'cell type'}, inplace=True)
patient_col = 'Patients'
sample_col = 'FileName'
group_col = 'Groups'
pos_cols = ['x', 'y']
pheno_col = 'cell type'  # unique columns with all cell types
sample_cols = [sample_col, pheno_col, patient_col, 'Spots' ,group_col]
marker_cols = obj.columns[9:67]
marker_cols = [x for x in marker_cols if x != 'HOECHST1:Cyc_1_ch_1']
# pos_cols = obj.columns[69:76]
# slices are 5µm thick, so data is pseudo-2D, including z leads to spatial networks reconstruction artifacts
cell_type_cols = list(obj.columns[75:96]) # one-hot encoded cell types
marker_posit_cols = list(obj.columns[100:-1])
all_cols = sample_cols + marker_cols + pos_cols + cell_type_cols + marker_posit_cols
net_cols = pos_cols + [pheno_col] + cell_type_cols + marker_cols
nb_clusters = obj[pheno_col].unique().size

# Select only pre-treatment samples
obj = obj.loc[obj[group_col] < 3, :]
obj.index = np.arange(len(obj))

code_groups = {
  1: 'Responder, pre-treatment',
  2: 'Non-responder, pre-treatment',
  3: 'Responder, post-treatment',
  4: 'Non-responder, post-treatment',
} 

status_path = data_dir / "41467_2021_26974_MOESM5_ESM - Patients_spots_conditons.xlsx"
# status = pd.read_excel(status_path, skiprows=2, usecols=[patient_col, group_col, 'Spots'])
# # samples 5, 25, 36, 37, 40, 44 are missing
# # now inferred from updated data at https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-26974-6/MediaObjects/41467_2021_26974_MOESM1_ESM.pdf
# updated_data = [
#     [2, 1, 5],
#     [7, 2, 25],
#     [9, 1, 36],
#     [9, 3, 37],
#     [10, 1, 40],
#     [11, 3, 44],
#     ]
# status = pd.concat([status, pd.DataFrame(updated_data, columns=status.columns)], axis=0)

# status[sample_col] = status['Spots'].apply(lambda x: f'reg{x:03}')
# # pd.set_option('display.max_rows', 100)

# if sample_col in status.columns:
#     status.index = status[sample_col]
#     status.drop(columns=[sample_col], inplace=True)
#     status.index.name = 'id'
# status.sort_values('Spots')

# When status had too many missing values after first data release:
# we make a status dataframe from the objects dataframe
status = obj[[sample_col, patient_col, group_col, 'Count', 'Spots']].groupby([patient_col, group_col, 'Spots', sample_col]).count().reset_index()
if sample_col in status.columns:
    status.index = status[sample_col]
    status.drop(columns=[sample_col], inplace=True)
    status.index.name = sample_col

# Drop sample with too few cells
# net_size_threshold = 300
net_size_threshold = 0    # no filtering
status = status.loc[status['Count'] >= net_size_threshold, :]
status.sort_values('Spots')
# status.drop(columns=['Count'], inplace=True)

# update the `object` dataframe
uniq_filenames = set(status.index)
obj = obj.query("FileName in @uniq_filenames")


survival_path = data_dir / "cohort_response.ods"
surv = pd.read_excel(survival_path)
surv.index = surv['Patient ID']
surv.drop(columns=['Patient ID'], inplace=True)
surv.index.name = patient_col


# make common color mapper
cycle_cmap = False

nodes_labels = obj[pheno_col]
uniq = pd.Series(nodes_labels).value_counts().index

processed_dir = Path('../data/processed/CODEX_CTCL')
dir_fig_save = processed_dir / 'figures'


trim_dist = 200 # or 'percentile_size'
min_neighbors = 3 # number of nearest nodes to connect for each unconneted node

reconst_dir = processed_dir / f"pretreatment_samples_networks_xy_min_size-{net_size_threshold}_solitary-{min_neighbors}"
save_dir = reconst_dir / "networks_images_common_cmap_saturated_first"
save_dir.mkdir(parents=True, exist_ok=True)
edges_dir = reconst_dir
nodes_dir = reconst_dir

uniq_patients = obj[patient_col].unique()
uniq_samples = obj[sample_col].unique()
n_uniq_patients = len(uniq_patients)
n_uniq_samples = len(uniq_samples)

# we add attributes to nodes to color network figures
nodes_all = obj[pos_cols + [pheno_col]].copy()
nodes_all = nodes_all.join(pd.get_dummies(obj[pheno_col]))
uniq_phenotypes = nodes_all[pheno_col].unique() 

count_types = obj[[patient_col, group_col, 'Count']].join(nodes_all[pheno_col]).groupby([patient_col, group_col, pheno_col]).count().unstack()
count_types.columns = count_types.columns.droplevel()
count_types = count_types.fillna(value=0).astype(int)
total_count_types = count_types.sum().sort_values(ascending=False)
prop_types = count_types.div(count_types.sum(axis=1), axis=0)

# With this dataset, when we filter networks without enough nodes
# we can't split patients in 5 groups having at least one occurrence
# of both groups (one patient of group 2 is deleted).

if net_size_threshold == 300:
    cv_train = 4
    cv_adapt = False
    cv_max = 4
else:
    cv_train = 5
    cv_adapt = False
    cv_max = 5

nodes_dir = Path('../data/processed/CODEX_CTCL/pretreatment_samples_networks_xy_min_size-0_solitary-3/transfo-clr/batch_correction-scanorama_on-patient/')





method = 'NAS'
# method = 'SCAN-IT'

order = 1
var_type = 'markers'
stat_funcs='default'
stat_names='default'
# stat_funcs = np.mean
# stat_names = 'mean'
# aggreg_vars = marker_cols

# directory of spatial omic features
if method == 'NAS':
    sof_dir = nodes_dir / f"nas_on-{var_type}_stats-{stat_names}_order-{order}"    
    sof_dir.mkdir(parents=True, exist_ok=True)
elif method == 'SCAN-IT':
    sof_dir = nodes_dir / f"scanit_on-{var_type}"    
    sof_dir.mkdir(parents=True, exist_ok=True)

# For UMAP visualization
marker = '.'
size_points = 10

# read column names in stored nodes files
node_file = list(nodes_dir.glob('nodes_*.parquet'))[0]
attributes_col = pd.read_parquet(node_file).columns[:-2]


filename = sof_dir / f'aggregation_statistics.parquet'

if filename.exists():
    print('Load aggregation statistics')
    var_aggreg = pd.read_parquet(filename)
else:
    var_aggreg = mosna.compute_spatial_omic_features_all_networks(
        method=method,
        nodes_dir=nodes_dir,
        edges_dir=edges_dir, 
        attributes_col=attributes_col, 
        use_attributes=marker_cols, # use all attributes 
        make_onehot=False, 
        stat_funcs=stat_funcs,
        stat_names=stat_names,
        id_level_1='patient',
        save_intermediate_results=False, 
        dir_save_interm=None,
        verbose=1,
        # memory_limit='1GB',
        )
    var_aggreg.to_parquet(filename, index=False)

# retrieve network info and remove it from NAS table
var_aggreg_samples_info = var_aggreg[['patient', 'sample']]
var_aggreg.drop(columns=['patient', 'sample'], inplace=True)

# ------ Screen NAS parameters ------

group_remapper = {2: 0, # non-responders
                  1: 1} # responders
group_cat_mapper = {0: 'Non-responder',
                    1: 'Responder'}
group_cat_mapper_rev = {val: key for key, val in group_cat_mapper.items()}


# predict_key = 'patient'
predict_key = 'sample'
group_col_cat = 'Response status'

if predict_key == 'patient':
    var_label = patient_col
    status_pred = surv[[group_col_cat]]
    status_pred[group_col] = status_pred[group_col_cat].map(group_cat_mapper_rev)
elif predict_key == 'sample':
    var_label = sample_col
    status_pred = status.copy()
    status_pred[group_col] = status_pred[group_col].map(group_remapper)
    status_pred[group_col_cat] = status_pred[group_col].map(group_cat_mapper)


#%% Define space of hyperparameters search

plot_heatmap = True
plot_alphas = False
plot_best_model_coefs = False
train_model = True
recompute = False
verbose = 1

DEBUG = False
if DEBUG:
    verbose = 2
    
show_progress = False
n_jobs_gridsearch = -1

columns = ['dim_clust', 'n_neighbors', 'metric', 'clusterer_type', 'k_cluster', 'clust_size_param', 'n_clusters', 'normalize', 'l1_ratio', 'alpha', 'score_roc_auc', 'score_ap', 'score_mcc']
col_types = {
    'dim_clust': int,
    'n_neighbors': int,
    'metric': 'category',
    'k_cluster': int,
    'clusterer_type': 'category',
    'clust_size_param': float,
    'n_clusters': int,
    'normalize': 'category',
    'l1_ratio': float,
    'alpha': float,
    'score_roc_auc': float, 
    'score_ap': float, 
    'score_mcc': float,
    }

l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
min_alpha = 0.001
dir_save_interm = sof_dir / f'search_LogReg_on_{predict_key}'
dir_save_interm.mkdir(parents=True, exist_ok=True)


RUN_LONG = True

if RUN_LONG:
    print('searching hyperparameters')
    all_models = []
    # screen NAS parameters
    iter_dim_clust = [2, 3, 4, 5]
    iter_dim_clust = [3, 4, 5]
    if show_progress:
        iter_dim_clust = tqdm(iter_dim_clust, leave=False)
    for dim_clust in iter_dim_clust:
        iter_n_neighbors = [15, 45, 75, 100, 200]
        if show_progress:
            iter_n_neighbors = tqdm(iter_n_neighbors, leave=False)
        for n_neighbors in iter_n_neighbors:
            iter_metric = ['manhattan', 'euclidean', 'cosine']
            if show_progress:
                iter_metric = tqdm(iter_metric, leave=False)
            for metric in iter_metric:
                iter_k_cluster = [x for x in iter_n_neighbors if x <= n_neighbors]
                if show_progress:
                    iter_k_cluster = tqdm(iter_k_cluster, leave=False)
                for k_cluster in iter_k_cluster:
                    # iter_clusterer_type = ['hdbscan', 'spectral', 'ecg', 'leiden', 'gmm']
                    # iter_clusterer_type = ['spectral', 'ecg', 'leiden', 'gmm']
                    iter_clusterer_type = ['leiden']
                    if show_progress:
                        iter_clusterer_type = tqdm(iter_clusterer_type, leave=False)
                    for clusterer_type in iter_clusterer_type:

                        if clusterer_type == 'spectral':
                            clust_size_param_name = 'n_clusters'
                            if DEBUG:
                                iter_clust_size_param = range(3, 6)
                            else:
                                iter_clust_size_param = range(3, 20)
                        elif clusterer_type == 'leiden':
                            clust_size_param_name = 'min_cluster_size'
                            iter_clust_size_param = [0.1, 0.03, 0.01, 0.003, 0.001]
                        elif clusterer_type == 'hdbscan':
                            clust_size_param_name = 'resolution'
                            iter_clust_size_param = [0.1, 0.03, 0.01, 0.003, 0.001]
                        elif clusterer_type == 'ecg':
                            clust_size_param_name = 'ecg_ensemble_size'
                            iter_clust_size_param = [20]
                        if clusterer_type == 'gmm':
                            clust_size_param_name = 'n_clusters'
                            iter_clust_size_param = range(3, 20)
                        
                        if show_progress:
                            iter_clust_size_param = tqdm(iter_clust_size_param, leave=False)
                        for clust_size_param in iter_clust_size_param:
                            cluster_params = {
                                'reducer_type': 'umap', 
                                # 'reducer_type': 'none', 
                                'n_neighbors': n_neighbors, 
                                'metric': metric,
                                'min_dist': 0.0,
                                'clusterer_type': clusterer_type, 
                                'dim_clust': dim_clust, 
                                'k_cluster': k_cluster, 
                                # 'flavor': 'CellCharter',
                                clust_size_param_name: clust_size_param,
                            }
                            str_params = '_'.join([str(key) + '-' + str(val) for key, val in cluster_params.items()])
                            print(str_params)

                            # try:
                            cluster_labels, cluster_dir, nb_clust, _ = mosna.get_clusterer(var_aggreg, sof_dir, verbose=verbose, **cluster_params)
                            n_clusters = len(np.unique(cluster_labels))
                            
                            # Survival analysis (just heatmap for now)
                            niches = cluster_labels
                            if n_clusters > 1:
                                for normalize in ['total', 'niche', 'obs', 'clr', 'niche&obs']:
                                    str_params = '_'.join([str(key) + '-' + str(val) for key, val in cluster_params.items()])
                                    str_params = str_params + f'_normalize-{normalize}'

                                    results_path = dir_save_interm / f'{str_params}.parquet'
                                    new_model = None
                                    if results_path.exists() and not recompute:
                                        if verbose > 1:
                                            print(f'load {results_path.stem}')
                                        new_model = pd.read_parquet(results_path)
                                    else:
                                        if train_model and n_clusters < 200:
                                            if verbose > 1:
                                                print(f'compute {results_path.stem}')
                                    
                                            var_aggreg_niches = var_aggreg_samples_info.copy()
                                            var_aggreg_niches['niche'] = np.array(niches)

                                            counts = mosna.make_niches_composition(var_aggreg_niches[predict_key], niches, var_label=var_label, normalize=normalize)
                                            counts.index = counts.index.astype(status_pred.index.dtype)
                                            exo_vars = counts.columns.astype(str).tolist()
                                            df_surv = pd.concat([status_pred, counts], axis=1, join='inner').fillna(0)
                                            # alternative aggregation
                                            # df_surv = counts.merge(status_pred, how='inner', on=var_label) 
                                            df_surv.columns = df_surv.columns.astype(str)
                                            df_surv.index.name = var_label

                                            # to model data, not to predict from it, do:
                                            split_train_test = False
                                            models = mosna.logistic_regression(
                                                df_surv[exo_vars + [group_col]],
                                                y_name=group_col,
                                                col_drop=[var_label],
                                                cv_train=cv_train, 
                                                cv_adapt=cv_adapt, 
                                                cv_max=cv_max,
                                                plot_coefs=False,
                                                split_train_test=split_train_test,
                                                )
                                            
                                            score_roc_auc = np.nanmax([models[model_type]['score']['ROC AUC'] for model_type in models.keys()])
                                            score_ap = np.nanmax([models[model_type]['score']['AP'] for model_type in models.keys()])
                                            score_mcc = np.nanmax([models[model_type]['score']['MCC'] for model_type in models.keys()])
                                            print(f'score_roc_auc: {score_roc_auc:.3f}')
                                            
                                            if score_roc_auc >= 0.7:
                                            
                                                best_id = np.argmax([models[model_type]['score']['ROC AUC'] for model_type in models.keys()])
                                                l1_ratio = [models[model_type]['model'].l1_ratio_[0] for model_type in models.keys()][best_id]
                                                alpha = [models[model_type]['model'].C_[0] for model_type in models.keys()][best_id]

                                                if plot_heatmap:
                                                    # make folder to save figures
                                                    path_parts = cluster_dir.parts[-2:]
                                                    dir_save_figures = dir_save_interm
                                                    for part in path_parts:
                                                        dir_save_figures = dir_save_figures / part
                                                    dir_save_figures.mkdir(parents=True, exist_ok=True)

                                                    try:
                                                        g, d = mosna.plot_heatmap(
                                                            # TODO: use group_col_cat instead?
                                                            df_surv[exo_vars + [group_col]].reset_index(), 
                                                            obs_labels=var_label, 
                                                            group_var=group_col, 
                                                            groups=[0, 1],
                                                            group_names=group_cat_mapper,
                                                            figsize=(10, 10),
                                                            z_score=False,
                                                            cmap=sns.color_palette("Reds", as_cmap=True),
                                                            return_data=True,
                                                            )
                                                        figname = f"biclustering_{str_params}_roc_auc-{score_roc_auc:.3f}.jpg"
                                                        plt.savefig(dir_save_figures / figname, dpi=150)
                                                        plt.close()

                                                        g, d = mosna.plot_heatmap(
                                                            df_surv[exo_vars + [group_col]].reset_index(), 
                                                            obs_labels=var_label, 
                                                            group_var=group_col, 
                                                            groups=[0, 1],
                                                            group_names=group_cat_mapper,
                                                            figsize=(10, 10),
                                                            z_score=1,
                                                            cmap=sns.color_palette("Reds", as_cmap=True),
                                                            return_data=True,
                                                            )
                                                        figname = f"biclustering_{str_params}_roc_auc-{score_roc_auc:.3f}_col_zscored.jpg"
                                                        plt.savefig(dir_save_figures / figname, dpi=150)
                                                        plt.close()
                                                    except:
                                                        pass
                                                
                                                new_model = [dim_clust, n_neighbors, metric, clusterer_type, k_cluster, clust_size_param, n_clusters, normalize, l1_ratio, alpha, score_roc_auc, score_ap, score_mcc]
                                        if new_model is None:
                                            # not trained or training failed
                                            new_model = [dim_clust, n_neighbors, metric, clusterer_type, k_cluster, clust_size_param, n_clusters, normalize, -1, -1, -1, -1, -1]
                                        new_model = pd.DataFrame(data=np.array(new_model).reshape((1, -1)), columns=columns)
                                        new_model = new_model.astype(col_types)
                                        new_model.to_parquet(results_path)
                                    
                                    all_models.append(new_model.values)
                                # except Exception as e:
                                #     print(e)
    all_models = pd.DataFrame(all_models, columns=columns)
    all_models = all_models.astype(col_types)
    all_models.to_parquet(dir_save_interm / 'all_models.parquet')
    print('done')
else:
    aggregated_file_path = dir_save_interm / 'all_models.parquet'
    if aggregated_file_path.exists():
        print('Load NAS hyperparameters search results')
        all_models = pd.read_parquet()
    else:
        print('Aggregate NAS hyperparameters search results')
        all_models = [pd.read_parquet(file_path) for file_path in dir_save_interm.glob('*.parquet')]
        all_models = pd.concat(all_models, axis=0).astype(col_types)
        all_models.index = np.arange(len(all_models))

