import scipy
import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
import typing as tp
import parmap
import anndata
from tqdm import tqdm


def preprocess(adata):
    """
    Prepares an AnnData object for Leiden clustering by computing PCA,
    nearest neighbor graph, and UMAP embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells as rows and genes/features as columns.
        Raw or normalized expression data expected.

    Returns
    -------
    ad : AnnData
        Copy of the input with PCA, neighbor graph, and UMAP added.
        Original adata is not modified.
    """
    ad = adata.copy()
    sc.tl.pca(ad)
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)
    return ad

def leiden_only_clustering(
    adata,
    resolution=1.0,
    random_state=0,
    n_iterations=-1,
    key_added="leiden_clusters"
):
    """
    Performs Leiden community detection on a preprocessed AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells as rows and genes/features as columns.
        Must be preprocessed with PCA and nearest neighbor graph computed.
    resolution : float
        Controls the coarseness of the clustering. Higher values produce more,
        smaller clusters (finer granularity); lower values produce fewer, larger
        clusters (broader cell populations). Default is 1.0.
    random_state : int
        Seed for the random number generator. Set to a fixed value for reproducible
        clustering results across runs. Default is 0.
    n_iterations : int
        Number of iterations to run the Leiden algorithm. Set to -1 to run until
        the algorithm converges to an optimal partition. Higher values may improve
        cluster stability at the cost of compute time. Default is -1.
    key_added : str
        Column name added to `adata.obs` where cluster labels will be stored.
        Access results via `adata.obs[key_added]` after clustering.
        Default is "leiden_clusters".
    
    Returns
    -------
    adata : AnnData
        Annotated data matrix with Leiden cluster assignments for each cell
        stored in `adata.obs[key_added]`.
    """
    ad = adata.copy()
    # Preprocess if neighbors haven't been computed yet
    if 'neighbors' not in ad.uns:
        ad = preprocess(ad)

    sc.tl.leiden(ad, 
                 resolution=resolution,
                 random_state=random_state,
                 n_iterations=n_iterations,
                 key_added=key_added
                )
    return ad

def plot(
    adata,
    color="leiden_clusters",
    title=None,
    save=None,
    palette=None,
    size=None,
    ax=None,
    legend_loc=None
):
    """
    Plots a UMAP embedding of the AnnData object colored by Leiden cluster labels.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with cells as rows and genes/features as columns.
        Must be preprocessed with PCA, nearest neighbor graph, and UMAP computed.
    color : str, optional
        Column name in `adata.obs` used to color the UMAP plot.
        Default is "leiden_clusters".
    title : str, optional
        Title displayed above the plot.
        If None, no title is shown. Default is None.
    save : str or bool, optional
        If a string, saves the plot to a file with that name (e.g. "my_plot.png").
        If True, saves with a default filename. If None, the plot is not saved.
        Default is None.
    palette : list or str, optional
        Color palette for cluster labels. Accepts a list of hex color codes,
        a matplotlib colormap name, or None to use the default palette.
        Default is None.
    size : float, optional
        Size of each cell dot in the UMAP plot. Increase for sparse plots,
        decrease for dense plots to reduce overlap. Default is None.
    ax : matplotlib.axes.Axes, optional
        Existing axes object to draw the plot onto. Useful for embedding this
        plot inside a multi-panel figure. If None, a new figure is created.
        Default is None.
    legend_loc : str, optional
        Location of the legend on the plot. Accepts standard matplotlib legend
        position strings such as "right margin", "on data", or "best".
        If None, the default location is used. Default is None.

    Returns
    -------
    None
        Displays the UMAP plot. If `save` is provided, also writes the plot to disk.
    """
    sc.pl.umap(
        adata,
        color=color,
        title=title,
        save=save,
        palette=palette,
        size=size,
        ax=ax,
        legend_loc=legend_loc
    )