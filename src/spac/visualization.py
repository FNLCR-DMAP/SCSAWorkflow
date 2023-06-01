import seaborn
import scanpy as sc
import pandas as pd
import numpy as np
import anndata
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def histogram(adata, column, group_by=None, together=False, **kwargs):
    """
    Plot the histogram of cells based specific column.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    column : str
        Name of member of adata.obs to plot the histogram.

    group_by : str, default None
        Choose either to group the histogram by another column.

    together : bool, default False
        If True, and if group_by !=None  create one plot for all groups.
        Otherwise, divide every histogram by the number of elements.

    **kwargs
        Parameters passed to matplotlib hist function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes of the histogram plot.

    fig : matplotlib.figure.Figure
        The created figure for the plot.

    """
    n_bins = len(adata.obs[column].unique()) - 1
    print("nbins=", n_bins)

    arrays = []
    labels = []

    if group_by is not None:
        groups = adata.obs[group_by].unique().tolist()
        observations = pd.concat(
            [adata.obs[column], adata.obs[group_by]],
            axis=1)

        for group in groups:
            group_cells = (observations[observations[group_by] ==
                           group][column].to_numpy())

            arrays.append(group_cells)
            labels.append(group)

        if together:
            fig, ax = plt.subplots()
            ax.hist(arrays, n_bins, label=labels, **kwargs)
            ax.legend(
                prop={'size': 10},
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.)
            ax.set_title(column)
            return ax, fig

        else:
            n_groups = len(groups)
            fig, ax = plt.subplots(n_groups)
            fig.tight_layout(pad=1)
            fig.set_figwidth(5)
            fig.set_figheight(5*n_groups)

            for group, ax_id in zip(groups, range(n_groups)):
                ax[ax_id].hist(arrays[ax_id], n_bins, **kwargs)
                ax[ax_id].set_title(group)
            return ax, fig

    else:
        fig, ax = plt.subplots()
        array = adata.obs[column].to_numpy()
        plt.hist(array, n_bins, label=column, **kwargs)
        ax.set_title(column)
        return ax, fig


def heatmap(adata, column, layer=None, **kwargs):
    """
    Plot the heatmap of the mean feature of cells that belong to a `column`.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    column : str
        Name of member of adata.obs to plot the histogram.

    layer : str, default None
        The name of the `adata` layer to use to calculate the mean feature.

    **kwargs:
        Parameters passed to seaborn heatmap function.

    Returns
    -------
    pandas.DataFrame
        A dataframe tha has the labels as indexes the mean feature for every
        marker.

    matplotlib.figure.Figure
        The figure of the heatmap.

    matplotlib.axes._subplots.AxesSubplot
        The AsxesSubplot of the heatmap.

    """
    features = adata.to_df(layer=layer)
    labels = adata.obs[column]
    grouped = pd.concat([features, labels], axis=1).groupby(column)
    mean_feature = grouped.mean()

    n_rows = len(mean_feature)
    n_cols = len(mean_feature.columns)
    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
    seaborn.heatmap(
        mean_feature,
        annot=True,
        cmap="Blues",
        square=True,
        ax=ax,
        fmt=".1f",
        cbar_kws=dict(use_gridspec=False, location="top"),
        linewidth=.5,
        annot_kws={"fontsize": 10},
        **kwargs)

    ax.tick_params(axis='both', labelsize=25)
    ax.set_ylabel(column, size=25)

    return mean_feature, fig, ax


def hierarchical_heatmap(
        adata,
        column,
        layer=None,
        dendrogram=True,
        standard_scale=None,
        **kwargs):
    """
    Plot a hierarchical clustering heatmap of the mean
    feature of cells that belong to a `column' using
    scanpy.tl.dendrogram and sc.pl.matrixplot.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    column : str
        Name of the column in adata.obs to group by and
        calculate mean feature.
    layer : str, optional, default: None
        The name of the `adata` layer to use to calculate the mean feature.
    dendrogram : bool, optional, default: True
        If True, a dendrogram based on the hierarchical clustering between
        the `column` categories is computed and plotted.
    **kwargs:
        Additional parameters passed to sc.pl.matrixplot function.

    Returns
    ----------
    feature, matrixplot

    """

    """
    # An example to call this function:
    mean_feature, matrixplot = hierarchical_heatmap(all_data,
    "phenograph", layer=None, standard_scale='var')

    # Display the figure
    #matrixplot.show()
    """

    # Calculate mean feature
    features = adata.to_df(layer=layer)
    labels = adata.obs[column]
    grouped = pd.concat([features, labels], axis=1).groupby(column)
    mean_feature = grouped.mean()

    # Reset the index of mean_feature
    mean_feature = mean_feature.reset_index()

    # Convert mean_feature to AnnData
    mean_feature_adata = sc.AnnData(
        X=mean_feature.iloc[:, 1:].values,
        obs=pd.DataFrame(
            index=mean_feature.index,
            data={column: mean_feature.iloc[:, 0].astype('category').values}
        ),
        var=pd.DataFrame(index=mean_feature.columns[1:]))

    # Compute dendrogram if needed
    if dendrogram:
        sc.tl.dendrogram(
            mean_feature_adata,
            groupby=column,
            var_names=mean_feature_adata.var_names,
            n_pcs=None)

    # Create the matrix plot
    matrixplot = sc.pl.matrixplot(
        mean_feature_adata,
        var_names=mean_feature_adata.var_names,
        groupby=column,
        use_raw=False,
        dendrogram=dendrogram,
        standard_scale=standard_scale,
        cmap="viridis",
        return_fig=True,
        **kwargs)

    return mean_feature, matrixplot


def threshold_heatmap(adata, feature_cutoffs, observation):
    """
    Creates a heatmap for each feature, categorizing intensities into low,
    medium, and high based on provided cutoffs.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the feature intensities in .X attribute.
    feature_cutoffs : dict
        Dictionary with feature names as keys and tuples with two intensity
        cutoffs as values.  observation : str Column name in .obs DataFrame
        that contains the observation used for grouping.

    Returns
    -------
    Dictionary of :class:`~matplotlib.axes.Axes`
        A dictionary contains the axes of figures generated in the scanpy
        heatmap function.
        Consistent Key: 'heatmap_ax'
        Potential Keys includes: 'groupby_ax', 'dendrogram_ax', and
        'gene_groups_ax'.

    """

    assert isinstance(feature_cutoffs, dict),\
        "feature_cutoffs should be a dictionary."
    for key, value in feature_cutoffs.items():
        assert isinstance(value, tuple) and len(value) == 2,\
            "Each value in feature_cutoffs should be a tuple of two elements."

    adata.uns['feature_cutoffs'] = feature_cutoffs

    intensity_df = pd.DataFrame(index=adata.obs_names,
                                columns=feature_cutoffs.keys())

    for feature, cutoffs in feature_cutoffs.items():
        low_cutoff, high_cutoff = cutoffs
        feature_values = adata[:, feature].X.flatten()
        intensity_df.loc[feature_values <= low_cutoff, feature] = 0
        intensity_df.loc[(feature_values > low_cutoff) &
                         (feature_values <= high_cutoff), feature] = 1
        intensity_df.loc[feature_values > high_cutoff, feature] = 2

    intensity_df = intensity_df.astype(int)
    adata.layers["intensity"] = intensity_df.to_numpy()
    adata.obs[observation] = adata.obs[observation].astype('category')

    color_map = {0: (0/255, 0/255, 139/255), 1: 'green', 2: 'yellow'}
    colors = [color_map[i] for i in range(3)]
    cmap = ListedColormap(colors)

    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    heatmap_plot = sc.pl.heatmap(
        adata,
        var_names=intensity_df.columns,
        groupby=observation,
        use_raw=False,
        layer='intensity',
        cmap=cmap,
        norm=norm,
        swap_axes=True,
        show=False
    )

    colorbar = plt.gcf().axes[-1]
    colorbar.set_yticks([0, 1, 2])
    colorbar.set_yticklabels(['low', 'medium', 'high'])

    return heatmap_plot
