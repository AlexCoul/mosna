# MOSNA: Multi‑Omics Spatial Network Analysis

**`mosna`** is a Python package for **spatial omics data analysis**, designed to extract clinically relevant features from single-cell spatial measurements (transcriptomics, proteomics, multiplexed imaging). MOSNA reconstructs spatial networks, computes interaction and neighborhood statistics, and trains predictive models integrating clinical outcomes.


## Overview of the Analysis Pipeline

MOSNA’s workflow progresses through increasing analytical complexity, starting with broad hypotheses on **cell proportions**, then moving to **interaction-based metrics** (aka "assortativity"), and finally **discovering spatial neighboroods** (also called niches), leveraging **machine learning** at each step for predictive modeling.

The pipeline stages:

| Stage                                         | Description                                                        |
| --------------------------------------------- | ------------------------------------------------------------------ |
| **1. Proportions**                            | Quantify global cell-type frequencies                              |
| **2. Composed Variables**                     | Ratios of proportions                                              |
| **3. Assortativity and mixing matrices**      | Preferential interaction between cell types                        |
| **4. Neighbors Aggregation Statistics (NAS)** | Cell neighborhoods (niches) from local omics data                  |


## 🔧 Installation

### Using Conda (GPU-enabled)

```bash
conda create -n mosna-gpu -c rapidsai -c conda-forge -c nvidia -c pytorch \
  rapids=23.04.01 python=3.10 cuda-version=11.2 pytorch=1.12.1 scanpy
conda activate mosna-gpu
pip install ipykernel ipywidgets tysserand scipy==1.13
cd /path/to/mosna
pip install -e .
```

### Without GPU

```bash
conda create -n mosna -c conda-forge python=3.10 scanpy
conda activate mosna
pip install ipykernel ipywidgets tysserand scipy==1.13
pip install -e .
```

## Detailed Steps and Analyses

### 1. **Cell-type Proportions**

* MOSNA computes proportions of cell types per sample.
* Performs statistical tests (e.g., Mann-Whitney U) to identify differences between groups (e.g., responder vs non-responder), correcting with Benjamini-Hochberg.

### 2. **Composed Variables (Ratios)**

* Computes first-order (cell-type ratios) and higher-order ratios.
* Enables detection of enriched features not obvious from absolute counts.

### 3. **Spatial Interaction Metrics**

* Uses **tysserand** to build spatial cell networks (e.g., Delaunay triangulation).
* Computes **mixing matrices** and **assortativity coefficients (AC)** for categorical attributes (cell types or marker positivity).
* Applies network attribute randomization and z‑scoring to account for cell type proportions bias.
* Identifies interacting cell-type pairs predictive of response (e.g., neutrophil self-interactions, CD8–B cell, etc...).

### 4. **Neighbors Aggregation Statistics (NAS) & Niches**

* Aggregates features (omics data) of each cell and its neighbors to compute local statistics (means, variances, proportions, etc...).
* Clusters these aggregated features (e.g., UMAP + k‑means) to define **niches** (local microenvironments).

### 5. **Model Training and Feature Importance**

* At each stage, MOSNA allows training **elastic‑net penalized logistic regression** (for binary outcome) **or Cox Proportional Hazard** (for survival) models with cross-validation.
* It reports peformance metrics, model coefficients, and identifies top predictive features.
* Typically, as feature sets become more refined (e.g., cell types interactions or niche proportions), predictive power increases.

<!-- ---

## 📁 Example Notebooks

MOSNA includes Jupyter notebooks illustrating the analyses on **CODEX CTCL spatial proteomics data**:

* Step-by-step demonstration: from network building → metric computations → modeling.
* Visualizations include heatmaps, interaction matrices, clustering outputs, ROC curves, and feature importance plots. -->


## References

* The core methodology and pipeline are published in [**bioRxiv**](https://www.biorxiv.org/content/10.1101/2023.03.16.532947v2), demonstrating its use on a spatial proteomics dataset of Cutaneous T-Cell Lymphoma (CTCL), generated with the CODEX technology with binary response, and on a a spatial proteomics dataset of breast cancer generated with the IMC technology, with survival (right-censored) data.
* MOSNA leverages [**tysserand**](https://github.com/VeraPancaldiLab/tysserand) for fast network construction applicable to any spatial omics data.


## Contributions & License

* Contributions are welcome via GitHub pull requests and issues.
* **Licensed under GPL‑3.0**, involving open-access reuse conditions.

---

## Summary

MOSNA provides:

* A reproducible, interpretable pipeline for spatial network analysis.
* Multiple layers of feature computation: proportions → interactions → niches.
* Predictive modeling tightly integrated with exploratory analysis.
* Support for diverse spatial omics platforms such as CODEX, MERFISH, CyCIF, Visium, etc.

Explore the example notebooks to get hands-on experience and adapt workflows to your own spatial omics datasets.