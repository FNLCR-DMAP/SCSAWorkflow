import scipy
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
import typing as tp
import parmap
import anndata
from tqdm import tqdm

# leiden docs - https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html
# anndata docs - https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html

# read the docs and make a comment about what sc.tl.leiden is doing

def preprocess(adata):
    ad = adata.copy()
    sc.tl.pca(ad)
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    return ad

def leiden_only_clustering(adata, resolution=1.0, random_state=0, n_iterations=-1, key_added="leiden_clusters"):
    ad = adata.copy()
    sc.tl.leiden(ad, 
                 resolution=resolution,
                 random_state=random_state,
                 n_iterations=n_iterations,
                 key_added=key_added
                )
    return ad

def plot(adata, color="leiden_clusters", title=None, save=None, palette=None, size=None):
    sc.pl.umap(adata,
               color=color,
               title=title,
               save=save,
               palette=palette,
               size=size
              )

''' plan:
        - feature addition
            - resolution, random_state, n_iterations, key_added
            - neighbors_key/obsp, use_weights, directed
            - partition_type, flavor, restrict_to
        - tune for other ml/clustering/normalization models
        - write unit tests & justify
'''