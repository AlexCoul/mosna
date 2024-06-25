# mosna: Multi-Omics Spatial Network Analysis

A package to find patterns and communities in spatial networks of nodes with attributes of possibly high dimension.  
It works on any type of omics data (proteomics, transcriptomics, ...) and technology (MERFISH, CODEX, MIBI-TOF, CyCIF, ...).  
*mosna* allows to train various machine learning models to predict clinical outcome on gradually more complex data (cell type proportions, interactions, niches), while looking for the best hyperparameters for the models.  

For an example of how to use mosna, see the notebook on the [CODEX CTCL data](./examples/_CODEX_CTCL_xy_only_pretreatment.ipynb).

## Installation

To use mosna with GPU-compatible libraries, you can try:
```bash
conda create --solver=libmamba -n mosna-gpu -c rapidsai -c conda-forge -c nvidia -c pytorch rapids=23.04.01 python=3.10 cuda-version=11.2 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 scanpy
conda activate mosna-gpu
```

without GPU you can do:
```bash
conda create --solver=libmamba -n mosna-c conda-forge python=3.10 scanpy
conda activate mosna
```

then do:
```bash
pip install ipykernel ipywidgets
pip install tysserand
cd /path/to/mosna_benchmark/
pip install -e .
```