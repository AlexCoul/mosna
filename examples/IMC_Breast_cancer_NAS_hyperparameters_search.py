import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
warnings.simplefilter('ignore', FitFailedWarning)
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from time import time
import warnings
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
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tysserand import tysserand as ty
from mosna import mosna

import matplotlib as mpl
mpl.rcParams["figure.facecolor"] = 'white'
mpl.rcParams["axes.facecolor"] = 'white'
mpl.rcParams["savefig.facecolor"] = 'white'

print('loading data')
data_dir = Path("../data/raw/IMC_Breast_cancer_Danenberg_2022")
objects_path = data_dir / "SingleCells.csv"

if objects_path.with_suffix('.parquet').exists():
    obj = pd.read_parquet(objects_path.with_suffix('.parquet'))
else:
    obj = pd.read_csv(objects_path)
    # for latter use
    obj.to_parquet(objects_path.with_suffix('.parquet'))

obj.rename(columns={'Location_Center_X': 'x', 'Location_Center_Y': 'y'}, inplace=True)
sample_col = 'ImageNumber'  # != from `sample_cols`
patient_col = 'metabric_id'
pheno_col = 'cellPhenotype'
sample_cols = [sample_col, patient_col, pheno_col, 'ObjectNumber']
# this file contains 'c-Caspase3c-PARP' instead of 'c-Caspase3c' as in the dataframe
# all_epitopes = pd.read_csv(data_dir / 'markerStackOrder.csv').iloc[:, 1].values
# so instead we directly do:
all_epitopes = obj.columns[11:50].values
# remove Histone H3 and DNA markers
marker_cols = list(all_epitopes[1:-2])
pos_cols = ['x', 'y']
cell_type_cols = [
    'is_epithelial',
    'is_tumour',
    'is_normal',
    'is_dcis',
    'is_interface',
    'is_perivascular',
    'is_hotAggregate',
    ]
nb_phenotypes = obj[pheno_col].unique().size
all_cols = sample_cols + marker_cols + pos_cols + cell_type_cols + [pheno_col]
# columns we want to include in network data
net_cols = pos_cols + [pheno_col] + cell_type_cols + marker_cols

print(f'nb phenotypes: {nb_phenotypes}')
print(f'nb used markers: {len(marker_cols)}')

# Show number of cells per sample
sample_sizes = obj[['ImageNumber', 'ObjectNumber']].groupby(['ImageNumber']).count()

# # aggregate unique pairs of patients and image IDs
# sample_patient_counts = obj[sample_cols].groupby([sample_col, patient_col]).count()
# print(sample_patient_counts)

# # count occurrences of patient IDs in patient / image pairs
# occ = sample_patient_counts.index.get_level_values(1).value_counts()
# print(occ)

# n_patient_multiple = np.sum(occ.values != 1)
# print(f'There are {n_patient_multiple} patients with multiple samples')

# sample_patient_mapper = dict(sample_patient_counts.index.values)

# ------ Survival data ------

survival_path = data_dir / "IMCClinical.csv"
surv = pd.read_csv(survival_path, index_col=0)

net_size_threshold = 150

select = sample_sizes['ObjectNumber'] >= net_size_threshold
print(f"there are {select.sum()} samples with > {net_size_threshold} cells, discarding {len(select) - select.sum()} samples")
sample_sizes = sample_sizes.loc[select, :]
uniq_samples = sample_sizes.index.values

print(f"filtering small samples, full dataframe dropping from {len(obj)} cells to", end=' ')
obj = obj.query("ImageNumber in @uniq_samples")
print(f'{len(obj)} cells')

print(f"survival data dropping from {len(surv)} patients to", end=' ')
uniq_patients = obj['metabric_id'].unique()
surv = surv.query("metabric_id in @uniq_patients")
print(f'{len(surv)} patients')

# make common color mapper
cycle_cmap = False

nodes_labels = obj[pheno_col]
uniq = pd.Series(nodes_labels).value_counts().index

if nodes_labels is not None:
    nb_clust = nodes_labels.max()
    uniq = pd.Series(nodes_labels).value_counts().index

    # choose colormap
    clusters_cmap = mosna.make_cluster_cmap(uniq)
    # make color mapper
    # series to sort by decreasing order
    n_colors = len(clusters_cmap)
    celltypes_color_mapper = {x: clusters_cmap[i % n_colors] for i, x in enumerate(uniq)}


# ------ All samples network reconstruction ------

processed_dir = Path('../data/processed/IMC_breast_cancer')
dir_fig_save = processed_dir / 'figures'

trim_dist = 200 # or 'percentile_size'
min_neighbors = 3 # number of nearest nodes to connect for each unconneted node

reconst_dir = processed_dir / f"samples_networks_xy_min_size-{net_size_threshold}_solitary-{min_neighbors}"
save_dir = reconst_dir / "networks_images_common_cmap_saturated_first"
save_dir.mkdir(parents=True, exist_ok=True)
edges_dir = reconst_dir
nodes_dir = reconst_dir

n_uniq_patients = len(uniq_patients)
n_uniq_samples = len(uniq_samples)

# ------ Response groups characteristics and survival analysis ------

# we add attributes to nodes to color network figures
nodes_all = obj[pos_cols + [pheno_col]].copy()
nodes_all = nodes_all.join(pd.get_dummies(obj[pheno_col]))
uniq_phenotypes = nodes_all[pheno_col].unique()  # 'attribute_uniq' in other notebooks

# here we don't use `sample_col` but `patient_col` in the group / unstack procedure to aggregate 
# statistics per patient and condition rather than per sample.

count_types = obj[[patient_col, pheno_col, sample_col]].groupby([patient_col, pheno_col]).count().unstack()
count_types.columns = count_types.columns.droplevel()
count_types = count_types.fillna(value=0).astype(int)
# count_types.index.name = 'sample'
count_types.to_csv(save_dir / 'count_types_per_patient.csv')

total_count_types = count_types.sum().sort_values(ascending=False)

prop_types = count_types.div(count_types.sum(axis=1), axis=0)

surv_orig = surv.copy()
surv = surv.loc[~surv['isValidation'], :]
surv.drop(columns=['isValidation'], inplace=True)

# replace some values with numbers for fitting
mapper_str = {
    'pos': 1,
    'neg': 0,
    np.nan: np.nan,
}
surv.loc[:, 'ERStatus'] = surv['ERStatus'].map(mapper_str)

mapper_bool = {
    True: 1,
    False: 0,
    np.nan: np.nan,
}
surv.loc[:, 'ERBB2_pos'] = surv['ERBB2_pos'].map(mapper_bool)

duration_col = 'yearsToStatus'
event_col = 'DeathBreast'
covariates = ['ERStatus'] #, 'ERBB2_pos'] # else []
strata = 'ERStatus'
drop_last_column = True   # because we use proportions last column is fully determined
drop_nan = True
if drop_last_column:
    df_surv = prop_types.iloc[:, :-1].join(surv[[duration_col, event_col] + covariates], how='inner')
else:
    df_surv = prop_types.join(surv[[duration_col, event_col] + covariates], how='inner')
if drop_nan:
    n_obs_orig = len(df_surv)
    df_surv.dropna(axis=0, inplace=True)
    n_obs = len(df_surv)
    if n_obs != n_obs_orig:
        print(f'discarded {n_obs_orig - n_obs} / {n_obs_orig} observations with NaN')


# %%
# ------------------------------------
# ------ Cellular neighborhoods ------
# ------------------------------------

print('performing / loading NAS data')
method = 'NAS'
# method = 'SCAN-IT'

order = 2
var_type = 'cell-types'
# stat_names = 'mean-std'
stat_funcs = np.mean
stat_names = 'mean'
# aggreg_vars = pheno_cols

# as we use cell types, we don't need to save data in a folder dedicated to a particular nodes data transformation
nodes_dir = edges_dir

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


filename = sof_dir / f'aggregation_statistics.parquet'

if filename.exists():
    var_aggreg = pd.read_parquet(filename)
else:
    var_aggreg = mosna.compute_spatial_omic_features_all_networks(
        method=method,
        nodes_dir=nodes_dir,
        edges_dir=edges_dir, 
        attributes_col=pheno_col,
        use_attributes=uniq_phenotypes, 
        make_onehot=True,
        stat_funcs=stat_funcs,
        stat_names=stat_names,
        id_level_1='patient',
        id_level_2='sample', 
        parallel_groups=False, #'max', 
        memory_limit='max',
        save_intermediate_results=False, 
        dir_save_interm=None,
        verbose=1,
        )
    var_aggreg.to_parquet(filename, index=False)

var_aggreg_samples_info = var_aggreg[['patient', 'sample']]
var_aggreg.drop(columns=['patient', 'sample'], inplace=True)

print('sof_dir:', sof_dir)

def surv_col_to_numpy(df_surv, event_col, duration_col):
    y_df = df_surv[[event_col, duration_col]].copy()
    y_df.loc[:, event_col] = y_df.loc[:, event_col].astype(bool)
    records = y_df.to_records(index=False)
    y = np.array(records, dtype = records.dtype.descr)
    return y

RUN_LONG = True

#%% Define space of hyperparameters search

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

columns = ['dim_clust', 'n_neighbors', 'metric', 'clusterer_type', 'clust_size_param', 'n_clusters', 'normalize', 'l1_ratio', 'alpha', 'n_coeffs', 'score']
col_types = {
    'dim_clust': int,
    'n_neighbors': int,
    'metric': 'category',
    'clusterer_type': 'category',
    'clust_size_param': float,
    'n_clusters': int,
    'normalize': 'category',
    'l1_ratio': float,
    'alpha': float,
    'n_coeffs': int,
    'score': float,
    }

l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
min_alpha = 0.001
dir_save_interm = sof_dir / 'search_CoxPH'
dir_save_interm.mkdir(parents=True, exist_ok=True)


#%% ------ Hyperparameters search ------

print('searching hyperparameters')
all_models = []

if RUN_LONG:
    # screen NAS parameters
    # iter_dim_clust = [2, 3, 4, 5, 6]
    iter_dim_clust = [0]
    if show_progress:
        iter_dim_clust = tqdm(iter_dim_clust)
    for dim_clust in iter_dim_clust:
        iter_n_neighbors = [15, 45, 75, 100]
        if show_progress:
            iter_n_neighbors = tqdm(iter_n_neighbors, leave=False)
        for n_neighbors in iter_n_neighbors:
            iter_metric = ['manhattan', 'euclidean', 'cosine']
            if show_progress:
                iter_metric = tqdm(iter_metric, leave=False)
            for metric in iter_metric:
                # iter_clusterer_type = ['hdbscan', 'spectral', 'ecg', 'leiden', 'gmm']
                # iter_clusterer_type = ['spectral', 'ecg', 'leiden', 'gmm']
                iter_clusterer_type = ['spectral', 'gmm']
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
                            # 'reducer_type': 'umap', 
                            'reducer_type': 'none', 
                            'n_neighbors': n_neighbors, 
                            'k_cluster': n_neighbors,
                            'metric': metric,
                            'min_dist': 0.0,
                            'clusterer_type': clusterer_type, 
                            'dim_clust': dim_clust, 
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
                        for normalize in ['none', 'total', 'niche', 'obs', 'clr', 'niche&obs']:
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
                                    counts = mosna.make_niches_composition(var_aggreg_niches['patient'], niches, var_label=patient_col, normalize=normalize)
                                    
                                    duration_col = 'yearsToStatus'
                                    event_col = 'DeathBreast'
                                    covariates = [] # ['ERStatus'] #, 'ERBB2_pos'] # else []
                                    strata = None #'ERStatus'
                                    status = surv[[duration_col, event_col] + covariates]

                                    drop_nan = True
                                    df_surv = counts.merge(status, how='inner', on=patient_col)
                                    if drop_nan:
                                        n_obs_orig = len(df_surv)
                                        df_surv.dropna(axis=0, inplace=True)
                                        n_obs = len(df_surv)
                                

                                    non_pred_cols = [patient_col, sample_col, event_col, duration_col]
                                    pred_cols = [x for x in df_surv if x not in non_pred_cols]
                                    # Xt = OneHotEncoder().fit_transform(X)
                                    Xt = df_surv[pred_cols]
                                    Xt.columns = Xt.columns.astype(str)
                                    y = surv_col_to_numpy(df_surv, event_col, duration_col)

                                    # Search best CoxPH model
                                    models = []
                                    scores = []
                                    all_cv_results = []
                                    for l1_ratio in l1_ratios:
                                        if verbose > 1:
                                            print(f'l1_ratio: {l1_ratio}', end='; ')
                                        
                                        try:
                                            coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=min_alpha, max_iter=100))
                                            coxnet_pipe.fit(Xt, y)

                                            # retrieve best alpha
                                            estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
                                            # estimated_alphas = [0.1, 0.01]

                                            cv = KFold(n_splits=5, shuffle=True, random_state=0)
                                            gcv = GridSearchCV(
                                                make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=l1_ratio)),
                                                param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
                                                cv=cv,
                                                error_score=0.5,
                                                n_jobs=n_jobs_gridsearch,
                                            ).fit(Xt, y)

                                            cv_results = pd.DataFrame(gcv.cv_results_)

                                            # retrieve best model
                                            best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]

                                            models.append(best_model)
                                            scores.append(best_model.score(Xt, y))
                                            all_cv_results.append(cv_results)
                                        except Exception as e:
                                            print(e)
                                    
                                    if len(scores) > 0:
                                        best_score_id = np.argmax(scores)
                                        best_model = models[best_score_id]
                                        best_cv = all_cv_results[best_score_id]
                                        score = scores[best_score_id]
                                        l1_ratio = best_model.l1_ratio
                                        alpha = best_model.alphas[0]
                                        best_coefs = pd.DataFrame(best_model.coef_, index=Xt.columns, columns=["coefficient"])
                                        non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
                                        # print(f"Number of non-zero coefficients: {non_zero}")
                                        non_zero_coefs = best_coefs.query("coefficient != 0")
                                        coef_order = non_zero_coefs.abs().sort_values("coefficient").index
                                        n_coefs = len(non_zero_coefs)

                                        if plot_alphas:
                                            alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
                                            mean = cv_results.mean_test_score
                                            std = cv_results.std_test_score

                                            fig, ax = plt.subplots(figsize=(9, 6))
                                            ax.plot(alphas, mean)
                                            ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
                                            ax.set_xscale("log")
                                            ax.set_ylabel("concordance index")
                                            ax.set_xlabel("alpha")
                                            ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
                                            ax.axhline(0.5, color="grey", linestyle="--")
                                            ax.grid(True)

                                        if plot_best_model_coefs:
                                            _, ax = plt.subplots(figsize=(6, 8))
                                            non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
                                            ax.set_xlabel("coefficient")
                                            ax.set_title(f'l1_ratio: {l1_ratio}  alpha: {alpha:.3g}  score: {score:.3g}')
                                            ax.grid(True)
                                        
                                        # str_params_model = f'{str_params}_l1_ratio-{l1_ratio:.3g}_alpha-{alpha:.3g}'
                                        # print(str_params_model)
                                        new_model = [dim_clust, n_neighbors, metric, clusterer_type, clust_size_param, n_clusters, normalize, l1_ratio, alpha, n_coefs, score]
                                if new_model is None:
                                    # not trained or training failed
                                    new_model = [dim_clust, n_neighbors, metric, clusterer_type, clust_size_param, n_clusters, normalize, -1, -1, -1, -1]
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
    print('Load NAS hyperparameters search results')
    all_models = pd.read_parquet(dir_save_interm / 'all_models.parquet')

