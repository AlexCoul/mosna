import warnings
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
warnings.simplefilter('ignore', FitFailedWarning)
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)
import numpy as np
import pandas as pd
import os
from time import time
import joblib
from pathlib import Path
from time import time
from tqdm import tqdm
import copy
import matplotlib as mpl

from tysserand import tysserand as ty
from mosna import mosna

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


method = 'NAS'
# method = 'SCAN-IT'

order = 1
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

from umap import UMAP
from umap.umap_ import nearest_neighbors
reducer_type = 'umap'
min_dist = 0.0
max_neigh = 200
subsample = True
sub = 100


global_start = time()
# for metric in ['manhattan', 'euclidean', 'cosine']:
for metric in ['euclidean']:
    print(f'metric: {metric}')

    for n_neighbors in [50]:
        print(f'    n_neighbors: {n_neighbors}')
        for dim_clust in [2]:
        # for dim_clust in [2, 3]:
            str_sp = ' ' * 7
            print(f'{str_sp} dim_clust: {dim_clust}')
            
            # perform dimensionality reduction
            reducer_name = f"reducer-{reducer_type}_metric-{metric}_nneigh-{n_neighbors}_dim-{dim_clust}_min_dist-{min_dist}"
            reducer_dir = Path(sof_dir) / reducer_name
            reducer_dir.mkdir(parents=True, exist_ok=True)
            embedding_path = reducer_dir / 'embedding.npy'
            reducer_path = reducer_dir / 'reducer.joblib'

            if embedding_path.exists():
                print(f'{str_sp} skipping embedding recomputation')
            else:
                print(f'{str_sp} computing embedding', end=' ')

                # JIT warmup
                reducer_warmup = UMAP(
                    random_state=None,
                    n_components=dim_clust,
                    n_neighbors=n_neighbors,
                    metric=metric,
                    min_dist=min_dist,
                    )
                embedding_warmup = reducer_warmup.fit_transform(var_aggreg.values[:int(max_neigh*2), :])

                # actual embedding
                start = time()
                reducer = UMAP(
                    random_state=None,
                    n_components=dim_clust,
                    n_neighbors=n_neighbors,
                    metric=metric,
                    min_dist=min_dist,
                    )
                if subsample:
                    embedding = reducer.fit_transform(var_aggreg[::sub].values)
                else:
                    embedding = reducer.fit_transform(var_aggreg.values)
                duration = time() - start
                print(f'took {duration:.2f} s')
                np.save(embedding_path, embedding, allow_pickle=False, fix_imports=False)
                joblib.dump(reducer, reducer_path)

global_duration = time() - global_start
print(f'done in {global_duration:.2f} s')
