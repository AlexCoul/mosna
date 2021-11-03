import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from time import time
import copy
from skimage import color
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform

import umap
# if not installed run: conda install -c conda-forge umap-learn
import hdbscan
from sklearn.cluster import OPTICS, cluster_optics_dbscan

from skimage import io

from scipy.stats import ttest_ind    # Welch's t-test
from scipy.stats import mannwhitneyu # Mann-Whitney rank test
from scipy.stats import ks_2samp     # Kolmogorov-Smirnov statistic

import sys
sys.path.extend([
    '../../tysserand',
    '../../mosna',
])

from tysserand import tysserand as ty
from mosna import mosna

# If need to reload modules after their modification
from importlib import reload
# ty = reload(ty)
# mosna = reload(mosna)
# sns = reload(sns)
# plt = reload(plt)

data_dir = Path("../../Commons/Common_data/MIBI-TOF__Triple_Negative_Breast_Cancer__Angelo_lab/processed_data")
patients_path = data_dir / "patient_class.csv"
objects_path = data_dir / "cellData.csv"
images_path = list(data_dir.glob('*.tiff'))

# relate image paths to patient numbers
import re

img_path_patients = [int(re.search('/processed_data/p(.+?)_labeledcell', str(s)).group(1)) for s in images_path]
pat_img = pd.Series(images_path, index=img_path_patients)

##### Patients data

patients = pd.read_csv(patients_path, index_col=0, header=None, names=['patient', 'response'])

### Objects data

obj = pd.read_csv(objects_path)

markers = obj.columns[3:-5]
# avoid control isotopes
markers = [
    'Vimentin', 'SMA', 'B7H3', 'FoxP3', 'Lag3',
    'CD4', 'CD16', 'CD56', 'OX40', 'PD1', 'CD31', 'PD-L1', 'EGFR', 'Ki67',
    'CD209', 'CD11c', 'CD138', 'CD163', 'CD68', 'CSF-1R', 'CD8', 'CD3',
    'IDO', 'Keratin17', 'CD63', 'CD45RO', 'CD20', 'p53', 'Beta catenin',
    'HLA-DR', 'CD11b', 'CD45', 'H3K9ac', 'Pan-Keratin', 'H3K27me3',
    'phospho-S6', 'MPO', 'Keratin6', 'HLA_Class_1',
]

other_cols = [x for x in obj.columns if x not in markers]

def clean_from_presence(im, values, target_val=0):
    """
    Set to a target values elements in im that are
    not present in values
    
    Example:
    >>> im = clean_from_presence(im, values=exp['cellLabelInImage'])
    """
    
    unim = pd.Series(np.unique(im))
    select = ~unim.isin(values)
    val_miss = unim.values[select]
    for i in val_miss:
        im[im == i] = target_val
    return im

batch_start = time()

processed_dir = Path('../data/processed/MIBI-TOF_TNBC')
save_dir = processed_dir / "all_samples_reconstruction_parameters"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

obj_size_thresh = 1250
k_sol = 2
dist_threshold = 50

for sample in obj['SampleID'].unique():
    print(f"processing sample {sample}")
    
    # ------ Open and pre-process sample image and objects data ------
    im = io.imread(pat_img[sample])
    exp = obj[obj['SampleID'] == sample]
    plt.figure()
    exp['cellSize'].hist(bins=50);
    title = f"sample-{sample}_histogram_objects_size"
    plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()

    
    obj_drop = exp.loc[exp['cellSize'] > obj_size_thresh, 'cellLabelInImage']
    # float image just for visualization with nan values
    im_drop = im.astype(float)
    for i in obj_drop.values:
        select = im_drop == i
        im_drop[select] = np.nan
        im[select] = 0
    current_cmap = mpl.cm.get_cmap()
    current_cmap.set_bad(color='red')
    ty.showim(im_drop);
    title = f"sample-{sample}_image_bad_objects"
    plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()
    im[im_drop == i] = 0

    # Values in images that are absent in cellLabelInImage were previously detected as background
    im = clean_from_presence(im, values=exp['cellLabelInImage'])
    ty.showim(im);
    title = f"sample-{sample}_image_cleaned"
    plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Contacting areas
    r_med = 5
    for r in [2, 5]:
        start = time()
        pairs = ty.build_contacting(im, r=r)
        end = time()
        print(f"running time: {end-start}s for {np.unique(im)[1:].size} objects")

        # reencode the coordinates to match node positions with their respective areas
        coords = ty.mask_val_coord(im)
        coords, pairs = ty.refactor_coords_pairs(coords, pairs)
        
        title = f"sample-{sample}_coords"
        np.savetxt(str(save_dir / title) + '.csv',
                   coords, 
                   fmt='%.18e', delimiter=',', newline='\n', header='x,y')
        title = f"sample-{sample}_pairs_contacting-area_r-{r}"
        np.savetxt(str(save_dir / title) + '.csv',
                   pairs,
                   fmt='%u', delimiter=',', newline='\n', header='source,target')
        distances = ty.distance_neighbors(coords, pairs)
        ty.plot_network_distances(coords, pairs, distances, figsize=(18,15), size_nodes=10);
        plt.axis('off');
        title = f"sample-{sample}_network_contacting-area_r-{r}"
        plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()
        
        # link solitary nodes to their closest neighbors
        pairs = ty.link_solitaries(coords, pairs, k=k_sol)
        title = f"sample-{sample}_pairs_contacting-area_r-{r}_k_sol-{k_sol}"
        np.savetxt(str(save_dir / title) + '.csv',
                   pairs,
                   fmt='%u', delimiter=',', newline='\n', header='source,target')
        distances = ty.distance_neighbors(coords, pairs)
        ty.plot_network_distances(coords, pairs, distances, figsize=(18,15), size_nodes=10);
        plt.axis('off');
        title = f"sample-{sample}_network_contacting-area_r-{r}_k_sol-{k_sol}"
        plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()
    
    # Delaunay triangulation
    pairs = ty.build_delaunay(coords, trim_dist=False)
    distances = ty.distance_neighbors(coords, pairs)

    ty.plot_network_distances(coords, pairs, distances, figsize=(18,15), size_nodes=10);
    plt.axis('off');
    title = f"sample-{sample}_network_delaunay_all_distances"
    plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    
    plt.figure()
    plt.hist(distances, bins=50);
    title = f"sample-{sample}_histogram_delaunay_distances"
    plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # trim edges with 'percentile_size' method
    trim_dist = ty.find_trim_dist(dist=distances, method='percentile_size', nb_nodes=coords.shape[0])
    
    select = distances < trim_dist
    pairs_selected = pairs[select,:]
    title = f"sample-{sample}_pairs_delaunay_trim-percentile_size-{trim_dist}"
    np.savetxt(str(save_dir / title) + '.csv',
               pairs_selected,
               fmt='%u', delimiter=',', newline='\n', header='source,target')
    ty.plot_network_distances(coords, pairs_selected, distances[select], figsize=(18,15), size_nodes=10);
    plt.axis('off');
    title = f"sample-{sample}_network_delaunay_trim-percentile_size-{trim_dist}"
    plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
    
    # trim edges with fixed distance threshold
    
    select = distances < dist_threshold
    pairs_selected = pairs[select,:]
    title = f"sample-{sample}_pairs_delaunay_trim-fixed-{dist_threshold}"
    np.savetxt(str(save_dir / title) + '.csv',
               pairs_selected,
               fmt='%u', delimiter=',', newline='\n', header='source,target')
    ty.plot_network_distances(coords, pairs_selected, distances[select], figsize=(18,15), size_nodes=10);
    plt.axis('off');
    title = f"sample-{sample}_network_delaunay_trim-fixed-{dist_threshold}"
    plt.savefig(str(save_dir / title) + '.png', bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()
   


batch_end = time()
print(f"batch running time: {batch_end-batch_start}s")
print("-" * 20)
print("-" * 6, Finished, "*" * 6)
print("-" * 20)

