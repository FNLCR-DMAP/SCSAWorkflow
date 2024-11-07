import seaborn as sns
import seaborn
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import math
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap, BoundaryNorm
from spac.utils import check_table, check_annotation
from spac.utils import check_feature, annotation_category_relations
from spac.utils import color_mapping
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def visualize_2D_scatter(
    x, y, adata=None, point_size=None, theme=None,
    ax=None, annotate_centers=False,
    x_axis_title='Component 1', y_axis_title='Component 2', plot_title=None,
    color_representation=None,pin_color=None, **kwargs
):
    """
    Visualize 2D data using plt.scatter.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the data.
    adata : anndata.AnnData, optional
        AnnData object containing data and metadata, including color mappings.
    point_size : float, optional
        Size of the points. If None, it will be automatically determined.
    theme : str, optional
        Color theme for the plot. Defaults to 'viridis' if theme is not recognized.
    ax : matplotlib.axes.Axes, optional (default: None)
        Matplotlib axis object. If None, a new one is created.
    annotate_centers : bool, optional (default: False)
        Annotate the centers of clusters if labels are categorical.
    x_axis_title : str, optional
        Title for the x-axis.
    y_axis_title : str, optional
        Title for the y-axis.
    plot_title : str, optional
        Title for the plot.
    color_representation : str, optional
        Description of what the colors represent.
    pin_color : str, optional
        Provides color dictionary for scatterplot, defaults to _spac_colors
    **kwargs
        Additional keyword arguments passed to plt.scatter.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure of the plot.
    ax : matplotlib.axes.Axes
        The axes of the plot.
    """

    # Input validation
    if not hasattr(x, "__iter__") or not hasattr(y, "__iter__"):
        raise ValueError("x and y must be array-like.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Define color themes
    themes = {
        'fire': plt.get_cmap('inferno'),
        'viridis': plt.get_cmap('viridis'),
        'inferno': plt.get_cmap('inferno'),
        'blue': plt.get_cmap('Blues'),
        'red': plt.get_cmap('Reds'),
        'green': plt.get_cmap('Greens'),
        'darkblue': ListedColormap(['#00008B']),
        'darkred': ListedColormap(['#8B0000']),
        'darkgreen': ListedColormap(['#006400'])
    }

    if theme and theme not in themes:
        error_msg = f"Theme '{theme}' not recognized. Please use a valid theme."
        raise ValueError(error_msg)

    cmap = themes.get(theme, plt.get_cmap('viridis'))

    # Determine point size
    num_points = len(x)
    if point_size is None:
        point_size = 5000 / num_points

    # Get figure size from kwargs or set defaults
    fig_width = kwargs.get('fig_width', 10)
    fig_height = kwargs.get('fig_height', 8)
    fontsize = kwargs.get('fontsize', 12)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure

    labels = kwargs.pop('labels', None)
    if adata is not None and (pin_color is not None or '_spac_colors' in adata.uns):
        if pin_color is not None:
            color_map = adata.uns.get(pin_color, {})
        else:
            color_map = adata.uns['_spac_colors']

        # Check if the color_representation exists in adata.obs
        if color_representation in adata.obs:
            annotation_colors = adata.obs[color_representation].astype(str)  
            colors = [color_map.get(label, 'gray') for label in annotation_colors]
            scatter = ax.scatter(x, y, c=colors, s=point_size, **kwargs)
        else:
            scatter = ax.scatter(x, y, c='gray', s=point_size, **kwargs)
    elif labels is not None:
        # Check if labels are categorical
        if pd.api.types.is_categorical_dtype(labels):
            if isinstance(labels, pd.Series):
                unique_clusters = labels.cat.categories
            elif isinstance(labels, pd.Categorical):
                unique_clusters = labels.categories
            else:
                raise TypeError(
                    "Expected labels to be of type Series[Categorical] or "
                    "Categorical."
                )
            cmap = plt.get_cmap('tab20', len(unique_clusters))
            for idx, cluster in enumerate(unique_clusters):
                mask = np.array(labels) == cluster
                ax.scatter(
                    x[mask], y[mask],
                    color=cmap(idx),
                    label=cluster,
                    s=point_size
                )

                # Annotate cluster centers if required
                if annotate_centers:
                    center_x = np.mean(x[mask])
                    center_y = np.mean(y[mask])
                    ax.text(
                        center_x, center_y, cluster,
                        fontsize=9, ha='center', va='center'
                    )
    else:
        scatter = ax.scatter(x, y, c='gray', s=point_size, **kwargs)

    # Set axis labels and title
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    if plot_title is not None:
        ax.set_title(plot_title)

    return fig, ax


def dimensionality_reduction_plot(adata, method, annotation=None, feature=None,
                                  layer=None, ax=None, **kwargs):
    """
    Visualize scatter plot in PCA, t-SNE or UMAP basis.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with coordinates precomputed by the 'tsne' or 'UMAP'
        function and stored in 'adata.obsm["X_tsne"]' or 'adata.obsm["X_umap"]'
    method : str
        Dimensionality reduction method to visualize.
        Choose from {'tsne', 'umap', 'pca'}.
    annotation : str, optional
        The name of the column in `adata.obs` to use for coloring
        the scatter plot points based on cell annotations.
    feature : str, optional
        The name of the gene or feature in `adata.var_names` to use
        for coloring the scatter plot points based on feature expression.
    layer : str, optional
        The name of the data layer in `adata.layers` to use for visualization.
        If None, the main data matrix `adata.X` is used.
    ax : matplotlib.axes.Axes, optional (default: None)
        A matplotlib axes object to plot on.
        If not provided, a new figure and axes will be created.
    **kwargs
        Parameters passed to visualize_2D_scatter function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure for the plot.
    ax : matplotlib.axes.Axes
        The axes of the plot.
    """

    # Check if both annotation and feature are specified, raise error if so
    if annotation and feature:
        raise ValueError(
            "Please specify either an annotation or a feature for coloring, "
            "not both.")

    # Use utility functions for input validation
    if layer:
        check_table(adata, tables=layer)
    if annotation:
        check_annotation(adata, annotations=annotation)
    if feature:
        check_feature(adata, features=[feature])

    # Validate the method and check if the necessary data exists in adata.obsm
    valid_methods = ['tsne', 'umap', 'pca']
    if method not in valid_methods:
        raise ValueError("Method should be one of {'tsne', 'umap', 'pca'}.")

    key = f'X_{method}'
    if key not in adata.obsm.keys():
        raise ValueError(
            f"{key} coordinates not found in adata.obsm. "
            f"Please run {method.upper()} before calling this function."
        )

    # Extract the 2D coordinates
    x, y = adata.obsm[key].T

    # Determine coloring scheme
    if annotation:
        color_values = adata.obs[annotation].astype('category').values
    elif feature:
        data_source = adata.layers[layer] if layer else adata.X
        color_values = data_source[:, adata.var_names == feature].squeeze()
    else:
        color_values = None

    fig, ax = visualize_2D_scatter(x, y, ax=ax, labels=color_values, **kwargs)

    return fig, ax


def tsne_plot(adata, color_column=None, ax=None, **kwargs):
    """
    Visualize scatter plot in tSNE basis.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with t-SNE coordinates precomputed by the 'tsne'
        function and stored in 'adata.obsm["X_tsne"]'.
    color_column : str, optional
        The name of the column to use for coloring the scatter plot points.
    ax : matplotlib.axes.Axes, optional (default: None)
        A matplotlib axes object to plot on.
        If not provided, a new figure and axes will be created.
    **kwargs
        Parameters passed to scanpy.pl.tsne function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure for the plot.
    ax : matplotlib.axes.Axes
        The axes of the tsne plot.
    """
    if not isinstance(adata, anndata.AnnData):
        raise ValueError("adata must be an AnnData object.")

    if 'X_tsne' not in adata.obsm:
        err_msg = ("adata.obsm does not contain 'X_tsne', "
                   "perform t-SNE transformation first.")
        raise ValueError(err_msg)

    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if color_column and (color_column not in adata.obs.columns and
                         color_column not in adata.var.columns):
        err_msg = f"'{color_column}' not found in adata.obs or adata.var."
        raise KeyError(err_msg)

    # Add color column to the kwargs for the scanpy plot
    if color_column:
        kwargs['color'] = color_column

    # Plot the t-SNE
    sc.pl.tsne(adata, ax=ax, **kwargs)

    return fig, ax


def histogram(adata, feature=None, annotation=None, layer=None,
              group_by=None, together=False, ax=None, **kwargs):
    """
    Plot the histogram of cells based on a specific feature from adata.X
    or annotation from adata.obs.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.

    feature : str, optional
        Name of continuous feature from adata.X to plot its histogram.

    annotation : str, optional
        Name of the annotation from adata.obs to plot its histogram.

    layer : str, optional
        Name of the layer in adata.layers to plot its histogram.

    group_by : str, default None
        Choose either to group the histogram by another column.

    together : bool, default False
        If True, and if group_by != None, create one plot combining all groups.
        If False, create separate histograms for each group.
        The appearance of combined histograms can be controlled using the
        `multiple` and `element` parameters in **kwargs.
        To control how the histograms are normalized (e.g., to divide the
        histogram by the number of elements in every group), use the `stat`
        parameter in **kwargs. For example, set `stat="probability"` to show
        the relative frequencies of each group.

    ax : matplotlib.axes.Axes, optional
        An existing Axes object to draw the plot onto, optional.

    **kwargs
        Additional keyword arguments passed to seaborn histplot function.
        Key arguments include:
        - `multiple`: Determines how the subsets of data are displayed
           on the same axes. Options include:
            * "layer": Draws each subset on top of the other
               without adjustments.
            * "dodge": Dodges bars for each subset side by side.
            * "stack": Stacks bars for each subset on top of each other.
            * "fill": Adjusts bar heights to fill the axes.
        - `element`: Determines the visual representation of the bins.
           Options include:
            * "bars": Displays the typical bar-style histogram (default).
            * "step": Creates a step line plot without bars.
            * "poly": Creates a polygon where the bottom edge represents
               the x-axis and the top edge the histogram's bins.
        - `log_scale`: Determines if the data should be plotted on
           a logarithmic scale.
        - `stat`: Determines the statistical transformation to use on the data
           for the histogram. Options include:
            * "count": Show the counts of observations in each bin.
            * "frequency": Show the number of observations divided
              by the bin width.
            * "density": Normalize such that the total area of the histogram
               equals 1.
            * "probability": Normalize such that each bar's height reflects
               the probability of observing that bin.
        - `bins`: Specification of hist bins.
            Can be a number (indicating the number of bins) or a list
            (indicating bin edges). For example, `bins=10` will create 10 bins,
            while `bins=[0, 1, 2, 3]` will create bins [0,1), [1,2), [2,3].
            If not provided, the binning will be determined automatically.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure for the plot.

    axs : list[matplotlib.axes.Axes]
        List of the axes of the histogram plots.

    """

    # If no feature or annotation is specified, apply default behavior
    if feature is None and annotation is None:
        # Default to the first feature in adata.var_names
        feature = adata.var_names[0]
        warnings.warn(
            "No feature or annotation specified. "
            "Defaulting to the first feature: "
            f"'{feature}'.",
            UserWarning
        )

    # Use utility functions for input validation
    if layer:
        check_table(adata, tables=layer)
    if annotation:
        check_annotation(adata, annotations=annotation)
    if feature:
        check_feature(adata, features=feature)
    if group_by:
        check_annotation(adata, annotations=group_by)

    # If layer is specified, get the data from that layer
    if layer:
        df = pd.DataFrame(
            adata.layers[layer], index=adata.obs.index, columns=adata.var_names
        )
    else:
        df = pd.DataFrame(
             adata.X, index=adata.obs.index, columns=adata.var_names
        )

    df = pd.concat([df, adata.obs], axis=1)

    if feature and annotation:
        raise ValueError("Cannot pass both feature and annotation,"
                         " choose one.")

    data_column = feature if feature else annotation

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    axs = []

    if group_by:
        groups = df[group_by].dropna().unique().tolist()
        n_groups = len(groups)
        if n_groups == 0:
            raise ValueError("There must be at least one group to create a"
                             " histogram.")

        if together:
            # Set default values if not provided in kwargs
            kwargs.setdefault("multiple", "stack")
            kwargs.setdefault("element", "bars")

            sns.histplot(data=df.dropna(), x=data_column, hue=group_by,
                         ax=ax, **kwargs)
            axs.append(ax)
        else:
            fig, ax_array = plt.subplots(
                n_groups, 1, figsize=(5, 5 * n_groups)
            )

            # Convert a single Axes object to a list
            # Ensure ax_array is always iterable
            if n_groups == 1:
                ax_array = [ax_array]

            for i, ax_i in enumerate(ax_array):
                sns.histplot(data=df[df[group_by] == groups[i]].dropna(),
                             x=data_column, ax=ax_i, **kwargs)
                ax_i.set_title(groups[i])
                axs.append(ax_i)
    else:
        sns.histplot(data=df, x=data_column, ax=ax, **kwargs)
        axs.append(ax)

    return fig, axs


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


def hierarchical_heatmap(adata, annotation, features=None, layer=None,
                         cluster_feature=False, cluster_annotations=False,
                         standard_scale=None, z_score="annotation",
                         swap_axes=False, rotate_label=False, **kwargs):

    """
    Generates a hierarchical clustering heatmap and dendrogram.
    By default, the dataset is assumed to have features as columns and
    annotations as rows. Cells are grouped by annotation (e.g., phenotype),
    and for each group, the average expression intensity of each feature
    (e.g., protein or marker) is computed. The heatmap is plotted using
    seaborn's clustermap.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    annotation : str
        Name of the annotation in adata.obs to group by and calculate mean
        intensity.
    features : list or None, optional
        List of feature names (e.g., markers) to be included in the
        visualization. If None, all features are used. Default is None.
    layer : str, optional
        The name of the `adata` layer to use to calculate the mean intensity.
        If not provided, uses the main matrix. Default is None.
    cluster_feature : bool, optional
        If True, perform hierarchical clustering on the feature axis.
        Default is False.
    cluster_annotations : bool, optional
        If True, perform hierarchical clustering on the annotations axis.
        Default is False.
    standard_scale : int or None, optional
        Whether to standard scale data (0: row-wise or 1: column-wise).
        Default is None.
    z_score : str, optional
        Specifies the axis for z-score normalization. Can be "feature" or
        "annotation". Default is "annotation".
    swap_axes : bool, optional
        If True, switches the axes of the heatmap, effectively transposing
        the dataset. By default (when False), annotations are on the vertical
        axis (rows) and features are on the horizontal axis (columns).
        When set to True, features will be on the vertical axis and
        annotations on the horizontal axis. Default is False.
    rotate_label : bool, optional
        If True, rotate x-axis labels by 45 degrees. Default is False.
    **kwargs:
        Additional parameters passed to `sns.clustermap` function or its
        underlying functions. Some essential parameters include:
        - `cmap` : colormap
          Colormap to use for the heatmap. It's an argument for the underlying
          `sns.heatmap()` used within `sns.clustermap()`. Examples include
          "viridis", "plasma", "coolwarm", etc.
        - `{row,col}_colors` : Lists or DataFrames
          Colors to use for annotating the rows/columns. Useful for visualizing
          additional categorical information alongside the main heatmap.
        - `{dendrogram,colors}_ratio` : tuple(float)
          Control the size proportions of the dendrogram and the color labels
          relative to the main heatmap.
        - `cbar_pos` : tuple(float) or None
          Specify the position and size of the colorbar in the figure. If set
          to None, no colorbar will be added.
        - `tree_kws` : dict
          Customize the appearance of the dendrogram tree. Passes additional
          keyword arguments to the underlying
          `matplotlib.collections.LineCollection`.
        - `method` : str
          The linkage algorithm to use for the hierarchical clustering.
          Defaults to 'centroid' in the function, but can be changed.
        - `metric` : str
          The distance metric to use for the hierarchy. Defaults to 'euclidean'
          in the function.

    Returns
    -------
    mean_intensity : pandas.DataFrame
        A DataFrame containing the mean intensity of cells for each annotation.
    clustergrid : seaborn.matrix.ClusterGrid
        The seaborn ClusterGrid object representing the heatmap and
        dendrograms.
    dendrogram_data : dict
        A dictionary containing hierarchical clustering linkage data for both
        rows and columns. These linkage matrices can be used to generate
        dendrograms with tools like scipy's dendrogram function. This offers
        flexibility in customizing and plotting dendrograms as needed.

    Examples
    --------
    import matplotlib.pyplot as plt
    import pandas as pd
    import anndata
    from spac.visualization import hierarchical_heatmap
    X = pd.DataFrame([[1, 2], [3, 4]], columns=['gene1', 'gene2'])
    annotation = pd.DataFrame(['type1', 'type2'], columns=['cell_type'])
    all_data = anndata.AnnData(X=X, obs=annotation)

    mean_intensity, clustergrid, dendrogram_data = hierarchical_heatmap(
        all_data,
        "cell_type",
        layer=None,
        z_score="annotation",
        swap_axes=True,
        cluster_feature=False,
        cluster_annotations=True
    )

    # To display a standalone dendrogram using the returned linkage matrix:
    import scipy.cluster.hierarchy as sch
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert the linkage data to type double
    dendro_col_data = np.array(dendrogram_data['col_linkage'], dtype=np.double)

    # Ensure the linkage matrix has at least two dimensions and
    more than one linkage
    if dendro_col_data.ndim == 2 and dendro_col_data.shape[0] > 1:
        fig, ax = plt.subplots(figsize=(10, 7))
        sch.dendrogram(dendro_col_data, ax=ax)
        plt.title('Standalone Col Dendrogram')
        plt.show()
    else:
        print("Insufficient data to plot a dendrogram.")
    """

    # Use utility functions to check inputs
    check_annotation(adata, annotations=annotation)
    if features:
        check_feature(adata, features=features)
    if layer:
        check_table(adata, tables=layer)

    # Raise an error if there are any NaN values in the annotation column
    if adata.obs[annotation].isna().any():
        raise ValueError("NaN values found in annotation column.")

    # Convert the observation column to categorical if it's not already
    if not pd.api.types.is_categorical_dtype(adata.obs[annotation]):
        adata.obs[annotation] = adata.obs[annotation].astype('category')

    # Calculate mean intensity
    if layer:
        intensities = pd.DataFrame(
            adata.layers[layer],
            index=adata.obs_names,
            columns=adata.var_names
        )
    else:
        intensities = adata.to_df()

    labels = adata.obs[annotation]
    grouped = pd.concat([intensities, labels], axis=1).groupby(annotation)
    mean_intensity = grouped.mean()

    # If swap_axes is True, transpose the mean_intensity
    if swap_axes:
        mean_intensity = mean_intensity.T

    # Map z_score based on user's input and the state of swap_axes
    if z_score == "annotation":
        z_score = 0 if not swap_axes else 1
    elif z_score == "feature":
        z_score = 1 if not swap_axes else 0

    # Subset the mean_intensity DataFrame based on selected features
    if features is not None and len(features) > 0:
        mean_intensity = mean_intensity.loc[features]

    # Determine clustering behavior based on swap_axes
    if swap_axes:
        row_cluster = cluster_feature  # Rows are features
        col_cluster = cluster_annotations  # Columns are annotations
    else:
        row_cluster = cluster_annotations  # Rows are annotations
        col_cluster = cluster_feature  # Columns are features

    # Use seaborn's clustermap for hierarchical clustering and
    # heatmap visualization.
    clustergrid = sns.clustermap(
        mean_intensity,
        standard_scale=standard_scale,
        z_score=z_score,
        method='centroid',
        metric='euclidean',
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cmap="viridis",
        **kwargs
    )

    # Rotate x-axis tick labels if rotate_label is True
    if rotate_label:
        plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=45)

    # Extract the dendrogram data for return
    dendro_row_data = None
    dendro_col_data = None

    if clustergrid.dendrogram_row:
        dendro_row_data = clustergrid.dendrogram_row.linkage

    if clustergrid.dendrogram_col:
        dendro_col_data = clustergrid.dendrogram_col.linkage

    # Define the dendrogram_data dictionary
    dendrogram_data = {
        'row_linkage': dendro_row_data,
        'col_linkage': dendro_col_data
    }

    return mean_intensity, clustergrid, dendrogram_data


def threshold_heatmap(
    adata, feature_cutoffs, annotation, layer=None, **kwargs
):
    """
    Creates a heatmap for each feature, categorizing intensities into low,
    medium, and high based on provided cutoffs.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the feature intensities in .X attribute
        or specified layer.
    feature_cutoffs : dict
        Dictionary with feature names as keys and tuples with two intensity
        cutoffs as values.
    annotation : str
        Column name in .obs DataFrame that contains the annotation
        used for grouping.
    layer : str, optional
        Layer name in adata.layers to use for intensities.
        If None, uses .X attribute.
    **kwargs : keyword arguments
        Additional keyword arguments to pass to scanpy's heatmap function.

    Returns
    -------
    Dictionary of :class:`~matplotlib.axes.Axes`
        A dictionary contains the axes of figures generated in the scanpy
        heatmap function.
        Consistent Key: 'heatmap_ax'
        Potential Keys includes: 'groupby_ax', 'dendrogram_ax', and
        'gene_groups_ax'.
    """

    # Use utility functions for input validation
    check_table(adata, tables=layer)
    if annotation:
        check_annotation(adata, annotations=annotation)
    if feature_cutoffs:
        check_feature(adata, features=list(feature_cutoffs.keys()))

    # Assert annotation is a string
    if not isinstance(annotation, str):
        err_type = type(annotation).__name__
        err_msg = (f'Annotation should be string. Got {err_type}.')
        raise TypeError(err_msg)

    # Assert annotation is a column in adata.obs DataFrame
    if annotation not in adata.obs.columns:
        err_msg = f"'{annotation}' not found in adata.obs DataFrame."
        raise ValueError(err_msg)

    if not isinstance(feature_cutoffs, dict):
        raise TypeError("feature_cutoffs should be a dictionary.")

    for key, value in feature_cutoffs.items():
        if not (isinstance(value, tuple) and len(value) == 2):
            raise ValueError(
                "Each value in feature_cutoffs should be a "
                "tuple of two elements."
            )
        if math.isnan(value[0]):
            raise ValueError(f"Low cutoff for {key} should not be NaN.")
        if math.isnan(value[1]):
            raise ValueError(f"High cutoff for {key} should not be NaN.")

    adata.uns['feature_cutoffs'] = feature_cutoffs

    intensity_df = pd.DataFrame(
        index=adata.obs_names, columns=feature_cutoffs.keys()
    )

    for feature, cutoffs in feature_cutoffs.items():
        low_cutoff, high_cutoff = cutoffs
        feature_values = (
            adata[:, feature].layers[layer]
            if layer else adata[:, feature].X
        ).flatten()
        intensity_df.loc[feature_values <= low_cutoff, feature] = 0
        intensity_df.loc[(feature_values > low_cutoff) &
                         (feature_values <= high_cutoff), feature] = 1
        intensity_df.loc[feature_values > high_cutoff, feature] = 2

    intensity_df = intensity_df.astype(int)
    adata.layers["intensity"] = intensity_df.to_numpy()
    adata.obs[annotation] = adata.obs[annotation].astype('category')

    color_map = {0: (0/255, 0/255, 139/255), 1: 'green', 2: 'yellow'}
    colors = [color_map[i] for i in range(3)]
    cmap = ListedColormap(colors)

    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    heatmap_plot = sc.pl.heatmap(
        adata,
        var_names=intensity_df.columns,
        groupby=annotation,
        use_raw=False,
        layer='intensity',
        cmap=cmap,
        norm=norm,
        **kwargs
    )

    colorbar = plt.gcf().axes[-1]
    colorbar.set_yticks([0, 1, 2])
    colorbar.set_yticklabels(['low', 'medium', 'high'])

    return heatmap_plot


def spatial_plot(
        adata,
        spot_size,
        alpha,
        vmin=-999,
        vmax=-999,
        annotation=None,
        feature=None,
        layer=None,
        ax=None,
        pin_color_rules=None,
        **kwargs
):

    """
    Generate the spatial plot of selected features
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing target feature and spatial coordinates.

    spot_size : int
        The size of spot on the spatial plot.
    alpha : float
        The transparency of spots, range from 0 (invisible) to 1 (solid)
    vmin : float or int
        The lower limit of the feature value for visualization
    vmax : float or int
        The upper limit of the feature value for visualization
    feature : str
        The feature to visualize on the spatial plot.
        Default None.
    annotation : str
        The annotation to visualize in the spatial plot.
        Can't be set with feature, default None.
    layer : str
        Name of the AnnData object layer that wants to be plotted.
        By default adata.raw.X is plotted.
    ax : matplotlib.axes.Axes
        The matplotlib Axes containing the analysis plots.
        The returned ax is the passed ax or new ax created.
        Only works if plotting a single component.
    pin_color_rules : str, optional
        Dictionary name in `adata.uns` for custom colors.
    **kwargs
        Arguments to pass to matplotlib.pyplot.scatter()
    Returns
    -------
        Single or a list of class:`~matplotlib.axes.Axes`.
    """

    err_msg_layer = "The 'layer' parameter must be a string, " + \
        f"got {str(type(layer))}"
    err_msg_feature = "The 'feature' parameter must be a string, " + \
        f"got {str(type(feature))}"
    err_msg_annotation = "The 'annotation' parameter must be a string, " + \
        f"got {str(type(annotation))}"
    err_msg_feat_annotation_coe = "Both annotation and feature are passed, " +\
        "please provide sinle input."
    err_msg_feat_annotation_non = "Both annotation and feature are None, " + \
        "please provide single input."
    err_msg_spot_size = "The 'spot_size' parameter must be an integer, " + \
        f"got {str(type(spot_size))}"
    err_msg_alpha_type = "The 'alpha' parameter must be a float," + \
        f"got {str(type(alpha))}"
    err_msg_alpha_value = "The 'alpha' parameter must be between " + \
        f"0 and 1 (inclusive), got {str(alpha)}"
    err_msg_vmin = "The 'vmin' parameter must be a float or an int, " + \
        f"got {str(type(vmin))}"
    err_msg_vmax = "The 'vmax' parameter must be a float or an int, " + \
        f"got {str(type(vmax))}"
    err_msg_ax = "The 'ax' parameter must be an instance " + \
        f"of matplotlib.axes.Axes, got {str(type(ax))}"

    if adata is None:
        raise ValueError("The input dataset must not be None.")

    if not isinstance(adata, anndata.AnnData):
        err_msg_adata = "The 'adata' parameter must be an " + \
            f"instance of anndata.AnnData, got {str(type(adata))}."
        raise ValueError(err_msg_adata)

    if layer is not None and not isinstance(layer, str):
        raise ValueError(err_msg_layer)

    if layer is not None and layer not in adata.layers.keys():
        err_msg_layer_exist = f"Layer {layer} does not exists, " + \
            f"available layers are {str(adata.layers.keys())}"
        raise ValueError(err_msg_layer_exist)

    if feature is not None and not isinstance(feature, str):
        raise ValueError(err_msg_feature)

    if annotation is not None and not isinstance(annotation, str):
        raise ValueError(err_msg_annotation)

    if annotation is not None and feature is not None:
        raise ValueError(err_msg_feat_annotation_coe)

    if annotation is None and feature is None:
        raise ValueError(err_msg_feat_annotation_non)

    if 'spatial' not in adata.obsm_keys():
        err_msg = "Spatial coordinates not found in the 'obsm' attribute."
        raise ValueError(err_msg)

    # Extract annotation name
    annotation_names = adata.obs.columns.tolist()
    annotation_names_str = ", ".join(annotation_names)

    if annotation is not None and annotation not in annotation_names:
        error_text = f'The annotation "{annotation}"' + \
            'not found in the dataset.' + \
            f" Existing annotations are: {annotation_names_str}"
        raise ValueError(error_text)

    # Extract feature name
    if layer is None:
        layer = adata.X
    else:
        layer = adata.layers[layer]

    feature_names = adata.var_names.tolist()

    if feature is not None and feature not in feature_names:
        error_text = f"Feature {feature} not found," + \
            " please check the sample metadata."
        raise ValueError(error_text)

    if not isinstance(spot_size, int):
        raise ValueError(err_msg_spot_size)

    if not isinstance(alpha, float):
        raise ValueError(err_msg_alpha_type)

    if not (0 <= alpha <= 1):
        raise ValueError(err_msg_alpha_value)

    if vmin != -999 and not (
        isinstance(vmin, float) or isinstance(vmin, int)
    ):
        raise ValueError(err_msg_vmin)

    if vmax != -999 and not (
        isinstance(vmax, float) or isinstance(vmax, int)
    ):
        raise ValueError(err_msg_vmax)

    if ax is not None and not isinstance(ax, plt.Axes):
        raise ValueError(err_msg_ax)

    feature_names = adata.var_names.tolist()
    annotation_names = adata.obs.columns.tolist()

    if feature is not None:
        feature_index = feature_names.index(feature)
        feature_values = adata.layers[layer][:, feature_index] if layer else adata.X[:, feature_index]
        feature_annotation = feature + "_spatial_plot"
        if vmin == -999:
            vmin = np.min(feature_values)
        if vmax == -999:
            vmax = np.max(feature_values)
        adata.obs[feature_annotation] = feature_values.flatten()
        color_region = feature_annotation
    else:
        color_region = annotation

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    spatial_coords = adata.obsm['spatial']
    x_coords, y_coords = spatial_coords[:, 0], spatial_coords[:, 1]

    # Color handling
    if pin_color_rules:
        color_map = adata.uns.get(pin_color_rules, {})
    elif '_spac_colors' in adata.uns:
        color_map = adata.uns['_spac_colors']
    else:
        color_map = {}
    
    colors = [color_map.get(label, 'gray') for label in adata.obs[annotation]]
    # Create scatter plot
    scatter = ax.scatter(
        x=x_coords,
        y=y_coords,
        c=colors, 
        s=spot_size,
        alpha=alpha,
        **kwargs
    )

    # Set color limits for feature-based visualization
    if feature is not None:
        scatter.set_clim(vmin, vmax)

    return [ax]


def boxplot(adata, annotation=None, second_annotation=None, layer=None,
            ax=None, features=None, log_scale=False, **kwargs):
    """
    Create a boxplot visualization of the features in the passed adata object.
    This function offers flexibility in how the boxplots are displayed,
    based on the arguments provided.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.

    annotation : str, optional
        Annotation to determine if separate plots are needed for every label.

    second_annotation : str, optional
        Second annotation to further divide the data.

    layer : str, optional
        The name of the matrix layer to use. If not provided,
        uses the main data matrix adata.X.

    ax : matplotlib.axes.Axes, optional
        An existing Axes object to draw the plot onto, optional.

    features : list, optional
        List of feature names to be plotted.
        If not provided, all features will be plotted.

    log_scale : bool, optional
        If True, the Y-axis will be in log scale. Default is False.

    **kwargs
        Additional arguments to pass to seaborn.boxplot.
        Key arguments include:
        - `orient`: Determines the orientation of the plot.
        * "v": Vertical orientation (default). In this case, categorical data
           will be plotted on the x-axis, and the boxplots will be vertical.
        * "h": Horizontal orientation. Categorical data will be plotted on the
           y-axis, and the boxplots will be horizontal.
    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axes for the plot.

    Examples
    --------
    - Multiple features boxplot: boxplot(adata, features=['GeneA','GeneB'])
    - Boxplot grouped by a single annotation:
      boxplot(adata, features=['GeneA'], annotation='cell_type')
    - Boxplot for multiple features grouped by a single annotation:
      boxplot(adata, features=['GeneA', 'GeneB'], annotation='cell_type')
    - Nested grouping by two annotations: boxplot(adata, features=['GeneA'],
      annotation='cell_type', second_annotation='treatment')
    """

    # Use utility functions to check inputs
    if layer:
        check_table(adata, tables=layer)
    if annotation:
        check_annotation(adata, annotations=annotation)
    if second_annotation:
        check_annotation(adata, annotations=second_annotation)
    if features:
        check_feature(adata, features=features)

    if 'orient' not in kwargs:
        kwargs['orient'] = 'v'

    if kwargs['orient'] != 'v':
        v_orient = False
    else:
        v_orient = True

    # Validate ax instance
    if ax and not isinstance(ax, plt.Axes):
        raise TypeError("Input 'ax' must be a matplotlib.axes.Axes object.")

    # Use the specified layer if provided
    if layer:
        data_matrix = adata.layers[layer]
    else:
        data_matrix = adata.X

    # Create a DataFrame from the data matrix with features as columns
    df = pd.DataFrame(data_matrix, columns=adata.var_names)

    # Add annotations to the DataFrame if provided
    if annotation:
        df[annotation] = adata.obs[annotation].values
    if second_annotation:
        df[second_annotation] = adata.obs[second_annotation].values

    # If features is None, set it to all available features
    if features is None:
        features = adata.var_names.tolist()

    df = df[
        features +
        ([annotation] if annotation else []) +
        ([second_annotation] if second_annotation else [])
    ]

    # Create the plot
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Plotting logic based on provided annotations
    if annotation and second_annotation:
        if v_orient:
            sns.boxplot(data=df, y=features[0], x=annotation,
                        hue=second_annotation, ax=ax, **kwargs)

        else:
            sns.boxplot(data=df, y=annotation, x=features[0],
                        hue=second_annotation, ax=ax, **kwargs)

        title_str = f"Nested Grouping by {annotation} and {second_annotation}"

        ax.set_title(title_str)

    elif annotation:
        if len(features) > 1:
            # Reshape the dataframe to long format for visualization
            melted_data = df.melt(id_vars=annotation)
            if v_orient:
                sns.boxplot(data=melted_data, x="variable", y="value",
                            hue=annotation,  ax=ax, **kwargs)
            else:
                sns.boxplot(data=melted_data, x="value", y="variable",
                            hue=annotation,  ax=ax, **kwargs)
            ax.set_title(f"Multiple Features Grouped by {annotation}")
        else:
            if v_orient:
                sns.boxplot(data=df, y=features[0], x=annotation,
                            ax=ax, **kwargs)
            else:
                sns.boxplot(data=df, x=features[0], y=annotation,
                            ax=ax, **kwargs)
            ax.set_title(f"Grouped by {annotation}")

    else:
        if len(features) > 1:
            sns.boxplot(data=df[features], ax=ax, **kwargs)
            ax.set_title("Multiple Features")
        else:
            if v_orient:
                sns.boxplot(x=df[features[0]], ax=ax, **kwargs)
            else:
                sns.boxplot(y=df[features[0]], ax=ax, **kwargs)
            ax.set_title("Single Boxplot")

    # Check if all data points are positive and non-zero
    all_positive = (df[features] > 0).all().all()

    # If log_scale is True and all data points are positive and non-zero
    if log_scale and all_positive:
        plt.yscale('log')

    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()

    return fig, ax


def interative_spatial_plot(
    adata,
    annotations,
    dot_size=1.5,
    dot_transparancy=0.75,
    colorscale='Viridis',
    figure_width=12,
    figure_height=8,
    figure_dpi=200,
    font_size=12

):

    """
    Create an interactive scatter plot for
    spatial data using provided annotations.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix object,
        must have a .obsm attribute with 'spatial' key.
    annotations : list of str or str
        Column(s) in `adata.obs` that contain the annotations to plot.
        If a single string is provided, it will be converted to a list.
        The interactive plot will show all the labels in the annotation
        columns passed.
    dot_size : float, optional
        Size of the scatter dots in the plot. Default is 1.5.
    dot_transparancy : float, optional
        Transparancy level of the scatter dots. Default is 0.75.
    colorscale : str, optional
        Name of the color scale to use for the dots. Default is 'Viridis'.
    figure_width : int, optional
        Width of the figure in inches. Default is 12.
    figure_height : int, optional
        Height of the figure in inches. Default is 8.
    figure_dpi : int, optional
        DPI (dots per inch) for the figure. Default is 200.
    font_size : int, optional
        Font size for text in the plot. Default is 12.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        A plotly figure object containing the spatial plot.

    Notes
    -----
    This function is specifically tailored for
    spatial single-cell data and expects the input AnnData object
    to have spatial coordinates stored in its .obsm attribute
    under the 'spatial' key.
    """

    if not isinstance(annotations, list):
        annotations = [annotations]

    for annotation in annotations:
        check_annotation(
            adata,
            annotations=annotation
        )

    if not hasattr(adata, 'obsm'):
        error_msg = ".obsm attribute (Spatial Coordinate) does not exist " + \
            "in the input AnnData object. Please check."
        raise ValueError(error_msg)

    if 'spatial' not in adata.obsm:
        error_msg = 'The key "spatial" is missing from .obsm field, hence ' + \
            "missing spatial coordniates. Please check."
        raise ValueError(error_msg)

    spatial_coords = adata.obsm['spatial']

    extract_columns_raw = []

    for item in annotations:
        extract_columns_raw.append(adata.obs[item])

    extract_columns = []

    for i, item in enumerate(extract_columns_raw):
        extract_columns.append(
            [annotations[i] + "_" + str(value) for value in item]
        )

    xcoord = [coord[0] for coord in spatial_coords]
    ycoord = [coord[1] for coord in spatial_coords]

    data = {'X': xcoord, 'Y': ycoord}

    # Add the extract_columns data as columns in the dictionary
    for i, column in enumerate(extract_columns):
        column_name = annotations[i]
        data[column_name] = column

    # Create the DataFrame
    df = pd.DataFrame(data)

    max_x_range = max(xcoord) * 1.1
    min_x_range = min(xcoord) * 0.9
    max_y_range = max(ycoord) * 1.1
    min_y_range = min(ycoord) * 0.9

    width_px = int(figure_width * figure_dpi)
    height_px = int(figure_height * figure_dpi)

    main_fig = px.scatter(
        df,
        x='X',
        y='Y',
        color=annotations[0],
        hover_data=[annotations[0]]
    )

    # If annotation is more than 1, we would first call px.scatter
    # to create plotly object, than append the data to main figure
    # with add_trace for a centralized view.
    if len(annotations) > 1:
        for obs in annotations[1:]:
            scatter_fig = px.scatter(
                                df,
                                x='X',
                                y='Y',
                                color=obs,
                                hover_data=[obs]
                            )

            main_fig.add_traces(scatter_fig.data)

    # Reset the color attribute of the traces in combined_fig
    for trace in main_fig.data:
        trace.marker.color = None

    main_fig.update_traces(
        mode='markers',
        marker=dict(
            size=dot_size,
            colorscale=colorscale,
            opacity=dot_transparancy
        ),
        hovertemplate="%{customdata[0]}<extra></extra>"
    )

    main_fig.update_layout(
        width=width_px,
        height=height_px,
        plot_bgcolor='white',
        font=dict(size=font_size),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='right',
            x=1.15,
            title='',
            itemwidth=30,
            bgcolor="rgba(0, 0, 0, 0)",
            traceorder='normal',
            entrywidth=50
        ),
        xaxis=dict(
                    range=[min_x_range, max_x_range],
                    showgrid=False,
                    showticklabels=False,
                    title_standoff=5,
                    constrain="domain"
                ),
        yaxis=dict(
                    range=[max_y_range, min_y_range],
                    showgrid=False,
                    scaleanchor="x",
                    scaleratio=1,
                    showticklabels=False,
                    title_standoff=5,
                    constrain="domain"
                ),
        shapes=[
            go.layout.Shape(
                type="rect",
                xref="x",
                yref="y",
                x0=min_x_range,
                y0=min_y_range,
                x1=max_x_range,
                y1=max_y_range,
                line=dict(color="black", width=1),
                fillcolor="rgba(0,0,0,0)",
            )
        ]
    )
    return main_fig


def sankey_plot(
        adata: anndata.AnnData,
        source_annotation: str,
        target_annotation: str,
        source_color_map: str = "tab20",
        target_color_map: str = "tab20c",
        sankey_font: float = 12.0,
        prefix: bool = True
):
    """
    Generates a Sankey plot from the given AnnData object.
    The color map refers to matplotlib color maps, default is tab20 for
    source annotation, and tab20c for target annotation.
    For more information on colormaps, see:
    https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    source_annotation : str
        The source annotation to use for the Sankey plot.
    target_annotation : str
        The target annotation to use for the Sankey plot.
    source_color_map : str
        The color map to use for the source nodes. Default is tab20.
    target_color_map : str
        The color map to use for the target nodes. Default is tab20c.
    sankey_font : float, optional
        The font size to use for the Sankey plot. Defaults to 12.0.
    prefix : bool, optional
        Whether to prefix the target labels with
        the source labels. Defaults to True.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        The generated Sankey plot.
    """

    label_relations = annotation_category_relations(
        adata=adata,
        source_annotation=source_annotation,
        target_annotation=target_annotation,
        prefix=prefix
    )
    # Extract and prepare source and target labels
    source_labels = label_relations["source"].unique().tolist()
    target_labels = label_relations["target"].unique().tolist()
    all_labels = source_labels + target_labels

    source_label_colors = color_mapping(source_labels, source_color_map)
    target_label_colors = color_mapping(target_labels, target_color_map)
    label_colors = source_label_colors + target_label_colors

    # Create a dictionary to map labels to indices
    label_to_index = {
        label: index for index, label in enumerate(all_labels)}
    color_to_map = {
        label: color
        for label, color in zip(source_labels, source_label_colors)
    }
    # Initialize lists to store the source indices, target indices, and values
    source_indices = []
    target_indices = []
    values = []
    link_colors = []

    # For each row in label_relations, add the source index, target index,
    # and count to the respective lists
    for _, row in label_relations.iterrows():
        source_indices.append(label_to_index[row['source']])
        target_indices.append(label_to_index[row['target']])
        values.append(row['count'])
        link_colors.append(color_to_map[row['source']])

    # Generate Sankey diagram
    # Calculate the x-coordinate for each label
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=sankey_font * 1.05,
            thickness=sankey_font * 1.05,
            line=dict(color=None, width=0.1),
            label=all_labels,
            color=label_colors
        ),
        link=dict(
            arrowlen=15,
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors
        ),
        arrangement="snap",
        textfont=dict(
            color='black',
            size=sankey_font
        )
    ))

    fig.data[0].link.customdata = label_relations[
        ['percentage_source', 'percentage_target']
    ]
    hovertemplate = (
        '%{source.label} to %{target.label}<br>'
        '%{customdata[0]}% to %{customdata[1]}%<br>'
        'Count: %{value}<extra></extra>'
    )
    fig.data[0].link.hovertemplate = hovertemplate

    # Customize the Sankey diagram layout
    fig.update_layout(
        title_text=(
            f'"{source_annotation}" to "{target_annotation}"<br>Sankey Diagram'
        ),
        title_x=0.5,
        title_font=dict(
            family='Arial, bold',
            size=sankey_font,  # Set the title font size
            color="black"  # Set the title font color
        )
    )

    fig.update_layout(margin=dict(
        l=10,
        r=10,
        t=sankey_font * 3,
        b=sankey_font))

    return fig
