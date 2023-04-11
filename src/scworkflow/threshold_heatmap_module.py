import pandas as pd
import numpy as np
import anndata
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import scanpy as sc


def threshold_heatmap(adata, marker_cutoffs, phenotype):
    """
    Creates a heatmap for each marker, categorizing intensities into low, medium, and
    high based on provided cutoffs.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the marker intensities in .X attribute.

    marker_cutoffs : dict
        Dictionary with marker names as keys and tuples with two intensity cutoffs
        as values.

    phenotype : str
        Column name in .obs DataFrame that contains the phenotype used for grouping.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object representing the heatmap with discretized intensity values
        (0, 1, or 2) for each marker, grouped by the specified phenotype.
        
    """

    # Save marker_cutoffs in the AnnData object
    adata.uns['marker_cutoffs'] = marker_cutoffs  

    intensity_df = pd.DataFrame(index=adata.obs_names, columns=marker_cutoffs.keys())

    for marker, cutoffs in marker_cutoffs.items():
        low_cutoff, high_cutoff = cutoffs
        marker_values = adata[:, marker].X.flatten()
        intensity_df.loc[marker_values <= low_cutoff, marker] = 0
        intensity_df.loc[(marker_values > low_cutoff) & (marker_values <= high_cutoff), marker] = 1
        intensity_df.loc[marker_values > high_cutoff, marker] = 2

    intensity_df = intensity_df.astype(int)

    # Add the intensity_df to adata as an AnnData layer
    adata.layers["intensity"] = intensity_df.to_numpy()

    # Convert the phenotype column to categorical
    adata.obs[phenotype] = adata.obs[phenotype].astype('category')

    # Create a custom color map for the heatmap
    color_map = {
        0: (0/255, 0/255, 139/255),
        1: 'green',
        2: 'yellow',
    }
    colors = [color_map[i] for i in range(3)]
    cmap = ListedColormap(colors)

    # Plot the heatmap using scanpy.pl.heatmap
    heatmap_plot = sc.pl.heatmap(adata,
                  var_names=intensity_df.columns,
                  groupby=phenotype,
                  use_raw=False,
                  layer='intensity',
                  cmap=cmap,
                  swap_axes=True, 
                  show=False)

    return heatmap_plot


"""
# Current function returns a Matplotlib figure object.
# Use the code below to display the heatmap when the function is called:

heatmap_figure = threshold_heatmap(adata, marker_cutoffs, phenotype)
plt.show()

"""

