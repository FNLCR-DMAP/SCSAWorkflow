
import anndata as ad
import scanpy.external as sce
import scanpy as sc
import pandas as pd
from scworkflow.normalization import subtract_min_quantile
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn


def ingest_cells(dataframe, regex_str, x_col=None, y_col=None, region=None):
    """
    Read the csv file into an anndata object.

    The function will also intialize intensities and spatial coordiantes.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data frame that contains cells as rows, and cells informations as
        columns

    regex_str : str
        A string representing python regular expression for the intensities
        columns in the data frame

    x_col : str
        The column name for the x coordinate of the cell

    y_col : str
        The column name for the y coordinate of the cell

    region : str
        The column name for the region that the cells

    Returns
    -------
    anndata.AnnData
        The generated AnnData object
    """
    intensities_regex = re.compile(regex_str)
    all_intensities = list(
        filter(intensities_regex.match, list(dataframe.columns)))

    intensities_df = dataframe[all_intensities]
    adata = ad.AnnData(
        intensities_df,
        dtype=intensities_df[all_intensities[0]].dtype)

    if region is not None:
        # As selecting one column of the dataframe returns a series which
        # AnnData converts to NaN, then I convert it to list before assignment
        adata.obs["region"] = dataframe[region].tolist()

    if x_col is not None and y_col is not None:
        numpy_array = dataframe[[x_col, y_col]].to_numpy().astype('float32')
        adata.obsm["spatial"] = numpy_array
    return adata


def concatinate_regions(regions):
    """
    Concatinate data from multiple regions and create new indexes.

    Parameters
    ----------
    regions : list of anndata.AnnData
        AnnData objects to be concatinated.

    Returns
    -------
    anndata.AnnData
        New AnddData object with the concatinated values in AnnData.X

    """
    all_adata = ad.concat(regions)
    all_adata.obs_names_make_unique()
    return all_adata


def rescale_intensities(intensities, min_quantile=0.01, max_quantile=0.99):
    """
    Clip and rescale intensities outside the minimum and maximum quantile.

    The rescaled intensities will be between 0 and 1.

    Parameters
    ----------
    intensities : pandas.Dataframe
        The DataRrame of intensities.

    min_quantile : float
        The minimum quantile to be consider zero.

    max_quantile: float
        The maximum quantile to be considerd 1.

    Returns
    -------
    pandas.DataFrame
        The created DataFrame with normalized intensities.
    """
    markers_max_quantile = intensities.quantile(max_quantile)
    markers_min_quantile = intensities.quantile(min_quantile)

    intensities_clipped = intensities.clip(
        markers_min_quantile,
        markers_max_quantile,
        axis=1)

    scaler = MinMaxScaler()
    np_intensities_scaled = scaler.fit_transform(
        intensities_clipped.to_numpy())

    intensities_scaled = pd.DataFrame(
        np_intensities_scaled,
        columns=intensities_clipped.columns)

    return intensities_scaled


def add_rescaled_intensity(adata, min_quantile, max_quantile, layer):
    """
    Clip and rescale the intensities matrix.

    The results will be added into a new layer in the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    min_quantile : float
        The minimum quantile to rescale to zero.

    max_quantile : float
        The maximum quantile to rescale to one.

    layer : str
        The name of the new layer to add to the anndata object.
    """

    original = adata.to_df()
    rescaled = rescale_intensities(original, min_quantile, max_quantile)
    adata.layers[layer] = rescaled


def pheongraph_clustering(adata, features, layer, k=30):
    """
    Calculate automatic phenotypes using phenograph.

    The function will add these two attributes to `adata`:
    `.obs["phenograph"]`
        The assigned int64 class by phenograph

    `.uns["phenograph_features"]`
        The features used to calculate the phenograph clusters



    Parameters
    ----------
    adata : anndata.AnnData
       The AnnData object.

    features : list of str
        The variables that would be included in creating the phenograph
        clusters.

    layer : str
        The layer to be used in calculating the phengraph clusters.

    k : int
        The number of nearest neighbor to be used in creating the graph.

    """
    phenograph_df = adata.to_df(layer=layer)[features]
    phenograph_out = sce.tl.phenograph(
        phenograph_df,
        clustering_algo="louvain",
        k=k)

    adata.obs["phenograph"] = phenograph_out[0].astype(np.int64)
    adata.uns["phenograph_features"] = features


def tsne(adata, layer=None):
    """
    Plot t-SNE from a specific layer information.

    Parameters
    ----------
    adata : anndatra.AnnData
       The AnnData object.

    layer : str
        The layer to be used in calculating the phengraph clusters.
    """
    # As scanpy.tl.tsne works on either X, obsm, or PCA, then I will copy the
    # layer data to an obsm if it is not the default X
    if layer is not None:
        X_tsne = adata.to_df(layer=layer)
        tsne_obsm_name = layer + "_tsne"
        adata.obsm[tsne_obsm_name] = X_tsne
    else:
        tsne_obsm_name = None

    sc.tl.tsne(adata, use_rep=tsne_obsm_name, random_state=7)


def subtract_min_per_region(adata, layer, min_quantile=0.01):
    """
    Substract the minimum quantile of every marker per region.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    min_quantile : float
        The minimum quantile to rescale to zero.

    layer : str
        The name of the new layer to add to the AnnData object.
    """
    regions = adata.obs['region'].unique().tolist()
    original = adata.to_df()

    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs['region'] == region]
        new_intensities = subtract_min_quantile(region_cells, min_quantile)
        new_df_list.append(new_intensities)

    new_df = pd.concat(new_df_list)
    adata.layers[layer] = new_df


def normalize(adata, layer, method="median", log=False):
    """
    Adjust the intensity of every marker using a normalization method.

    The normalization methods are summarized here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8723144/
    Adds the normalized values in
    `.layers[`layer`]`

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    layer : str
        The name of the new layer to add to the anndata object.

    method : {"median", "Q50", "Q75}
        The normlalization method to use.

    log : bool, default False
        If True, take the log2 of intensities before normalization.

    """
    allowed_methods = ["median", "Q50", "Q75"]
    regions = adata.obs['region'].unique().tolist()
    original = adata.to_df()

    if log:
        original = np.log2(1+original)

    if method == "median" or method == "Q50":
        all_regions_quantile = original.quantile(q=0.5)
    elif method == "Q75":
        all_regions_quantile = original.quantile(q=0.75)
    else:
        raise Exception(
            "Unsupported normalization {0}, allowed methods = {1]",
            method, allowed_methods)

    # Place holder for normalized dataframes per region
    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs['region'] == region]

        if method == "median":
            region_median = region_cells.quantile(q=0.5)
            new_intensities = region_cells + \
                (all_regions_quantile - region_median)

        if method == "Q50":
            region_median = region_cells.quantile(q=0.5)
            new_intensities = (region_cells
                               * all_regions_quantile
                               / region_median)

        if method == "Q75":
            region_75quantile = region_cells.quantile(q=0.75)
            new_intensities = (region_cells
                               * all_regions_quantile
                               / region_75quantile)

        new_df_list.append(new_intensities)

    new_df = pd.concat(new_df_list)
    adata.layers[layer] = new_df


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
    Plot the heatmap of the mean intensity of cells that belong to a `column`.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    column : str
        Name of member of adata.obs to plot the histogram.

    layer : str, default None
        The name of the `adata` layer to use to calculate the mean intensity.

    **kwargs:
        Parameters passed to seaborn heatmap function.

    Returns
    -------
    pandas.DataFrame
        A dataframe tha has the labels as indexes the mean intensity for every
        marker.

    matplotlib.figure.Figure
        The figure of the heatmap.

    matplotlib.axes._subplots.AxesSubplot
        The AsxesSubplot of the heatmap.

    """
    intensities = adata.to_df(layer=layer)
    labels = adata.obs[column]
    grouped = pd.concat([intensities, labels], axis=1).groupby(column)
    mean_intensity = grouped.mean()

    n_rows = len(mean_intensity)
    n_cols = len(mean_intensity.columns)
    fig, ax = plt.subplots(figsize=(n_cols * 1.5, n_rows * 1.5))
    seaborn.heatmap(
        mean_intensity,
        annot=True,
        cmap="Blues",
        square=True,
        ax=ax,
        fmt=".1f",
        cbar_kws=dict(use_gridspec=False, location="top"),
        linewidth=.5,
        annot_kws={"fontsize": 20},
        **kwargs)

    ax.tick_params(axis='both', labelsize=25)
    ax.set_ylabel(column, size=25)

    return mean_intensity, fig, ax
