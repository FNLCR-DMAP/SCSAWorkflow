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
import plotly.figure_factory as ff
import plotly.colors as pc
from matplotlib.colors import ListedColormap, BoundaryNorm
from spac.utils import check_table, check_annotation
from spac.utils import check_feature, annotation_category_relations
from spac.utils import check_label
from functools import partial
from spac.utils import color_mapping, spell_out_special_characters
from spac.data_utils import select_values
import logging
import warnings
import re
import copy
import datashader as ds
import datashader.transfer_functions as tf

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def heatmap_datashader(x, y, labels=None, theme=None,
                        x_axis_title="Component 1", y_axis_title="Component 2",
                        plot_title=None, **kwargs):
    """
    Generates a heatmap using Datashader for large-scale scatter data.
    
    Parameters:
    x (iterable): X-axis coordinates.
    y (iterable): Y-axis coordinates.
    labels (iterable, optional): Categorical labels for subgrouping data.
    theme (str, optional): Colormap theme for visualization. Defaults to 'viridis'.
    x_axis_title (str, optional): Label for the x-axis. Default is "Component 1".
    y_axis_title (str, optional): Label for the y-axis. Default is "Component 2".
    plot_title (str, optional): Title of the plot.
    **kwargs: Additional keyword arguments (e.g., 'fig_width', 'fig_height').
    
    Returns:
    matplotlib.figure.Figure: A Matplotlib figure containing the heatmap visualization.
    """
    
    # Ensure x and y are iterable
    if not hasattr(x, "__iter__") or not hasattr(y, "__iter__"):
        raise ValueError("x and y must be array-like.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if labels is not None and len(labels) != len(x):
        raise ValueError("Labels length should match x and y length.")

    # Define available color themes
    themes = {
        'Elevation': ds.colors.Elevation,
        'viridis': ds.colors.viridis,
        'Hot': ds.colors.Hot,
        'Set1': ds.colors.Set1,
        'Set2': ds.colors.Set2,
        'Set3': ds.colors.Set3,
        'Sets1to3': ds.colors.Sets1to3,
        'inferno': ds.colors.inferno,
        'color_lookup': ds.colors.color_lookup,
    }
    cmap = themes.get(theme, ds.colors.viridis)  # Default to 'viridis' if theme is not specified

    # Create a DataFrame for processing
    coords = pd.DataFrame({"x": x, "y": y})
    if labels is not None:
        coords["labels"] = labels

    # Determine plot boundaries
    x_min, x_max = coords["x"].min(), coords["x"].max()
    y_min, y_max = coords["y"].min(), coords["y"].max()

    # Set figure dimensions
    plot_width = kwargs.get('fig_width', 600)
    plot_height = kwargs.get('fig_height', 400)

    if labels is not None:
        # Create multiple subplots for each category
        categories = pd.Categorical(coords["labels"]).categories
        num_categories = len(categories)

        # Arrange subplots in rows of 3
        rows = (num_categories // 3) + (1 if num_categories % 3 != 0 else 0)
        fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
        axes = axes.flatten()

        for i, cat in enumerate(categories):
            subset = coords[coords["labels"] == cat]
            canvas = ds.Canvas(
                plot_width=plot_width,
                plot_height=plot_height,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max)
            )
            agg = canvas.points(subset, x="x", y="y", agg=ds.count())
            img = tf.shade(agg, cmap=cmap).to_pil()

            ax = axes[i]
            ax.imshow(img, origin='lower', extent=(x_min, x_max, y_min, y_max))
            ax.set_title(f"{plot_title} - {cat}" if plot_title else str(cat))
            ax.set_xlabel(x_axis_title)
            ax.set_ylabel(y_axis_title)

        # Remove unused subplot axes if they exist
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    else:
        # Generate single heatmap if no labels are provided
        canvas = ds.Canvas(
            plot_width=plot_width,
            plot_height=plot_height,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max)
        )
        agg = canvas.points(coords, x="x", y="y", agg=ds.count())
        img = tf.shade(agg, cmap=cmap).to_pil()

        fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100))
        ax.imshow(img, origin='lower', extent=(x_min, x_max, y_min, y_max))
        ax.set_title(plot_title if plot_title else "Density Plot")
        ax.set_xlabel(x_axis_title)
        ax.set_ylabel(y_axis_title)

    plt.tight_layout()
    return fig




def visualize_2D_scatter(
    x, y, labels=None, point_size=None, theme=None,
    ax=None, annotate_centers=False,
    x_axis_title='Component 1', y_axis_title='Component 2', plot_title=None,
    color_representation=None, **kwargs
):
    """
    Visualize 2D data using plt.scatter.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the data.
    labels : array-like, optional
        Array of labels for the data points. Can be numerical or categorical.
    point_size : float, optional
        Size of the points. If None, it will be automatically determined.
    theme : str, optional
        Color theme for the plot. Defaults to 'viridis' if theme not
        recognized. For a list of supported themes, refer to Matplotlib's
        colormap documentation:
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
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
    if labels is not None and len(labels) != len(x):
        raise ValueError("Labels length should match x and y length.")

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
        error_msg = (
            f"Theme '{theme}' not recognized. Please use a valid theme."
        )
        raise ValueError(error_msg)
    cmap = themes.get(theme, plt.get_cmap('viridis'))

    # Determine point size
    num_points = len(x)
    if point_size is None:
        point_size = 5000 / num_points

    # Get figure size and fontsize from kwargs or set defaults
    fig_width = kwargs.get('fig_width', 10)
    fig_height = kwargs.get('fig_height', 8)
    fontsize = kwargs.get('fontsize', 12)

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure

    # Plotting logic
    if labels is not None:
        # Check if labels are categorical
        if pd.api.types.is_categorical_dtype(labels):

            # Determine how to access the categories based on
            # the type of 'labels'
            if isinstance(labels, pd.Series):
                unique_clusters = labels.cat.categories
            elif isinstance(labels, pd.Categorical):
                unique_clusters = labels.categories
            else:
                raise TypeError(
                    "Expected labels to be of type Series[Categorical] or "
                    "Categorical."
                )

            # Combine colors from multiple colormaps
            cmap1 = plt.get_cmap('tab20')
            cmap2 = plt.get_cmap('tab20b')
            cmap3 = plt.get_cmap('tab20c')
            colors = cmap1.colors + cmap2.colors + cmap3.colors

            # Use the number of unique clusters to set the colormap length
            cmap = ListedColormap(colors[:len(unique_clusters)])

            for idx, cluster in enumerate(unique_clusters):
                mask = np.array(labels) == cluster
                ax.scatter(
                    x[mask], y[mask],
                    color=cmap(idx),
                    label=cluster,
                    s=point_size
                )
                print(f"Cluster: {cluster}, Points: {np.sum(mask)}")

                if annotate_centers:
                    center_x = np.mean(x[mask])
                    center_y = np.mean(y[mask])
                    ax.text(
                        center_x, center_y, cluster,
                        fontsize=fontsize, ha='center', va='center'
                    )
            # Create a custom legend with color representation
            ax.legend(
                loc='best',
                bbox_to_anchor=(1.25, 1),  # Adjusting position
                title=f"Color represents: {color_representation}"
            )

        else:
            # If labels are continuous
            scatter = ax.scatter(
                x, y, c=labels, cmap=cmap, s=point_size, **kwargs
            )
            plt.colorbar(scatter, ax=ax)
            if color_representation is not None:
                ax.set_title(
                    f"{plot_title}\nColor represents: {color_representation}"
                )
    else:
        scatter = ax.scatter(x, y, c='gray', s=point_size, **kwargs)

    # Equal aspect ratio for the axes
    ax.set_aspect('equal', 'datalim')

    # Set axis labels
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)

    # Set plot title
    if plot_title is not None:
        ax.set_title(plot_title)

    return fig, ax


def dimensionality_reduction_plot(
        adata,
        method=None,
        annotation=None,
        feature=None,
        layer=None,
        ax=None,
        associated_table=None,
        **kwargs):
    """
    Visualize scatter plot in PCA, t-SNE, UMAP, or associated table.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with coordinates precomputed by the 'tsne' or 'UMAP'
        function and stored in 'adata.obsm["X_tsne"]' or 'adata.obsm["X_umap"]'
    method : str, optional (default: None)
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
    associated_table : str, optional (default: None)
        Name of the key in `obsm` that contains the numpy array. Takes
        precedence over `method`
    **kwargs
        Parameters passed to visualize_2D_scatter function,
        including point_size.

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
    if associated_table is None:
        valid_methods = ['tsne', 'umap', 'pca']
        if method not in valid_methods:
            raise ValueError("Method should be one of {'tsne', 'umap', 'pca'}"
                             f'. Got:"{method}"')

        key = f'X_{method}'
        if key not in adata.obsm.keys():
            raise ValueError(
                f"{key} coordinates not found in adata.obsm. "
                f"Please run {method.upper()} before calling this function."
            )

    else:
        check_table(
            adata=adata,
            tables=associated_table,
            should_exist=True,
            associated_table=True
        )

        associated_table_shape = adata.obsm[associated_table].shape
        if associated_table_shape[1] != 2:
            raise ValueError(
                f'The associated table:"{associated_table}" does not have'
                f' two dimensions. It shape is:"{associated_table_shape}"'
            )
        key = associated_table

    print(f'Running visualization using the coordinates: "{key}"')

    # Extract the 2D coordinates
    x, y = adata.obsm[key].T

    # Determine coloring scheme
    if annotation:
        color_values = adata.obs[annotation].astype('category').values
        color_representation = annotation
    elif feature:
        data_source = adata.layers[layer] if layer else adata.X
        color_values = data_source[:, adata.var_names == feature].squeeze()
        color_representation = feature
    else:
        color_values = None
        color_representation = None

    # Set axis titles based on method and color representation
    if method == 'tsne':
        x_axis_title = 't-SNE 1'
        y_axis_title = 't-SNE 2'
        plot_title = f'TSNE-{color_representation}'
    elif method == 'pca':
        x_axis_title = 'PCA 1'
        y_axis_title = 'PCA 2'
        plot_title = f'PCA-{color_representation}'
    elif method == 'umap':
        x_axis_title = 'UMAP 1'
        y_axis_title = 'UMAP 2'
        plot_title = f'UMAP-{color_representation}'
    else:
        x_axis_title = f'{associated_table} 1'
        y_axis_title = f'{associated_table} 2'
        plot_title = f'{associated_table}-{color_representation}'

    # Remove conflicting keys from kwargs
    kwargs.pop('x_axis_title', None)
    kwargs.pop('y_axis_title', None)
    kwargs.pop('plot_title', None)
    kwargs.pop('color_representation', None)

    fig, ax = visualize_2D_scatter(
        x=x,
        y=y,
        ax=ax,
        labels=color_values,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        plot_title=plot_title,
        color_representation=color_representation,
        **kwargs
    )

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
              group_by=None, together=False, ax=None,
              x_log_scale=False, y_log_scale=False, **kwargs):
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

    x_log_scale : bool, default False
        If True, the data will be transformed using np.log1p before plotting,
        and the x-axis label will be adjusted accordingly.

    y_log_scale : bool, default False
        If True, the y-axis will be set to log scale.

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

    axs : matplotlib.axes.Axes or list of Axes
        The Axes object(s) of the histogram plot(s). Returns a single Axes
        if only one plot is created, otherwise returns a list of Axes.

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

    # Check for negative values and apply log1p transformation if x_log_scale is True
    if x_log_scale:
        if (df[data_column] < 0).any():
            print(
                "There are negative values in the data, disabling x_log_scale."
            )
            x_log_scale = False
        else:
            df[data_column] = np.log1p(df[data_column])

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    axs = []

    # Prepare the data for plotting
    plot_data = df.dropna(subset=[data_column])

    # Bin calculation section
    # The default bin calculation used by sns.histo take quite
    # some time to compute for large number of points,
    # DMAP implemented the Rice rule for bin computation

    def cal_bin_num(
        num_rows
    ):
        bins = max(int(2*(num_rows ** (1/3))), 1)
        print(f'Automatically calculated number of bins is: {bins}')
        return(bins)

    num_rows = plot_data.shape[0]

    # Check if bins is being passed
    # If not, the in house algorithm will compute the number of bins 
    if 'bins' not in kwargs:
        kwargs['bins'] = cal_bin_num(num_rows)

    # Plotting with or without grouping
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
            else:
                ax_array = ax_array.flatten()

            for i, ax_i in enumerate(ax_array):
                group_data = plot_data[plot_data[group_by] == groups[i]]

                sns.histplot(data=group_data, x=data_column, ax=ax_i, **kwargs)
                ax_i.set_title(groups[i])

                # Set axis scales if y_log_scale is True
                if y_log_scale:
                    ax_i.set_yscale('log')

                # Adjust x-axis label if x_log_scale is True
                if x_log_scale:
                    xlabel = f'log({data_column})'
                else:
                    xlabel = data_column
                ax_i.set_xlabel(xlabel)

                # Adjust y-axis label based on 'stat' parameter
                stat = kwargs.get('stat', 'count')
                ylabel_map = {
                    'count': 'Count',
                    'frequency': 'Frequency',
                    'density': 'Density',
                    'probability': 'Probability'
                }
                ylabel = ylabel_map.get(stat, 'Count')
                if y_log_scale:
                    ylabel = f'log({ylabel})'
                ax_i.set_ylabel(ylabel)

                axs.append(ax_i)
    else:
        sns.histplot(data=plot_data, x=data_column, ax=ax, **kwargs)
        axs.append(ax)

    # Set axis scales if y_log_scale is True
    if y_log_scale:
        ax.set_yscale('log')

    # Adjust x-axis label if x_log_scale is True
    if x_log_scale:
        xlabel = f'log({data_column})'
    else:
        xlabel = data_column
    ax.set_xlabel(xlabel)

    # Adjust y-axis label based on 'stat' parameter
    stat = kwargs.get('stat', 'count')
    ylabel_map = {
        'count': 'Count',
        'frequency': 'Frequency',
        'density': 'Density',
        'probability': 'Probability'
    }
    ylabel = ylabel_map.get(stat, 'Count')
    if y_log_scale:
        ylabel = f'log({ylabel})'
    ax.set_ylabel(ylabel)

    if len(axs) == 1:
        return fig, axs[0]
    else:
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
    adata, feature_cutoffs, annotation, layer=None, swap_axes=False, **kwargs
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
    swap_axes : bool, optional
        If True, swaps the axes of the heatmap.
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
    check_annotation(adata, annotations=annotation)
    if feature_cutoffs:
        check_feature(adata, features=list(feature_cutoffs.keys()))

    # Assert annotation is a string
    if not isinstance(annotation, str):
        err_type = type(annotation).__name__
        err_msg = (f'Annotation should be string. Got {err_type}.')
        raise TypeError(err_msg)

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
        show=False,   # Ensure the plot is not displayed but returned
        swap_axes=swap_axes,
        **kwargs
    )

    # Print the keys of the heatmap_plot dictionary
    print("Keys of heatmap_plot:", heatmap_plot.keys())

    # Get the main heatmap axis from the available keys
    heatmap_ax = heatmap_plot.get('heatmap_ax')

    # If 'heatmap_ax' key does not exist, access the first axis available
    if heatmap_ax is None:
        heatmap_ax = next(iter(heatmap_plot.values()))
    print("Heatmap Axes:", heatmap_ax)

    # Find the colorbar associated with the heatmap
    cbar = None
    for child in heatmap_ax.get_children():
        if hasattr(child, 'colorbar'):
            cbar = child.colorbar
            break
    if cbar is None:
        print("No colorbar found in the plot.")
        return
    print("Colorbar:", cbar)

    new_ticks = [0, 1, 2]
    new_labels = ['Low', 'Medium', 'High']
    cbar.set_ticks(new_ticks)
    cbar.set_ticklabels(new_labels)
    pos_heatmap = heatmap_ax.get_position()
    cbar.ax.set_position(
        [pos_heatmap.x1 + 0.02, pos_heatmap.y0, 0.02, pos_heatmap.height]
    )

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
        layer_process = adata.X
    else:
        layer_process = adata.layers[layer]

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

    if feature is not None:

        feature_index = feature_names.index(feature)
        feature_annotation = feature + "spatial_plot"
        if vmin == -999:
            vmin = np.min(layer_process[:, feature_index])
        if vmax == -999:
            vmax = np.max(layer_process[:, feature_index])
        adata.obs[feature_annotation] = layer_process[:, feature_index]
        color_region = feature_annotation
    else:
        color_region = annotation
        vmin = None
        vmax = None

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax = sc.pl.spatial(
        adata=adata,
        layer=layer,
        color=color_region,
        spot_size=spot_size,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        show=False,
        **kwargs)

    return ax


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
    print("Calculating Box Plot...")
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

    # Check for negative values
    if log_scale and (df[features] < 0).any().any():
        print(
            "There are negative values in this data, disabling the log scale."
        )
        log_scale = False

    # Apply log1p transformation if log_scale is True
    if log_scale:
        df[features] = np.log1p(df[features])

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
            if v_orient:
                sns.boxplot(data=df[features], ax=ax, **kwargs)
            else:
                melted_data = df.melf()
                sns.boxplot(data=melted_data, x="value", y="variable",
                            hue=annotation,  ax=ax, **kwargs)
            ax.set_title("Multiple Features")
        else:
            if v_orient:
                sns.boxplot(y=df[features[0]], ax=ax, **kwargs)
                ax.set_xticks([0])  # Set a single tick for the single feature
                ax.set_xticklabels([features[0]])  # Set the label for the tick
            else:
                sns.boxplot(x=df[features[0]], ax=ax, **kwargs)
                ax.set_yticks([0])  # Set a single tick for the single feature
                ax.set_yticklabels([features[0]])  # Set the label for the tick
            ax.set_title("Single Boxplot")

    # Set x and y-axis labels
    if v_orient:
        xlabel = annotation if annotation else 'Feature'
        ylabel = 'log(Intensity)' if log_scale else 'Intensity'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        xlabel = 'log(Intensity)' if log_scale else 'Intensity'
        ylabel = annotation if annotation else 'Feature'
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig, ax, df


def interative_spatial_plot(
    adata,
    annotations,
    dot_size=1.5,
    dot_transparancy=0.75,
    colorscale='rainbow',
    figure_width=6,
    figure_height=4,
    figure_dpi=200,
    font_size=12,
    stratify_by=None,
    defined_color_map=None,
    **kwargs
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
    stratify_by : str, optional
        Column in `adata.obs` to stratify the plot. Default is None.
    defined_color_map : str, optional
        Predefined color mapping stored in adata.uns for specific labels.
        Default is None, which will generate the color mapping automatically.
    **kwargs
        Additional keyword arguments for customization.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing the following keys:
        - "image_name": str, the name of the generated image.
        - "image_object": Plotly Figure object.

    Notes
    -----
    This function is tailored for spatial single-cell data and expects the
    AnnData object to have spatial coordinates in its `.obsm` attribute under
    the 'spatial' key.
    """

    if not isinstance(annotations, list):
        annotations = [annotations]

    for annotation in annotations:
        check_annotation(
            adata,
            annotations=annotation
        )

    check_table(
        adata,
        tables='spatial',
        associated_table=True
    )

    if defined_color_map is not None:
        if not isinstance(defined_color_map, str):
            raise TypeError(
                'The "degfined_color_map" should be a string ' + \
                f'getting {type(defined_color_map)}.'
            )
        uns_keys = list(adata.uns.keys())
        if len(uns_keys) == 0:
            raise ValueError(
                "No existing color map found, please" + \
                " make sure the Append Pin Color Rules " + \
                "template had been ran prior to the "+ \
                "current visualization node.")

        if defined_color_map not in uns_keys:
            raise ValueError(
                f'The given color map name: {defined_color_map} ' + \
                "is not found in current analysis, " + \
                f'available items are: {uns_keys}'
            )
        defined_color_map_dict = adata.uns[defined_color_map]
        print(
            f'Selected color mapping "{defined_color_map}":\n' + \
            f'{defined_color_map_dict}'
        )


    def main_figure_generation(
        adata,
        annotations=annotations,
        dot_size=dot_size,
        dot_transparancy=dot_transparancy,
        colorscale=colorscale,
        figure_width=figure_width,
        figure_height=figure_height,
        figure_dpi=figure_dpi,
        font_size=font_size,
        **kwargs
    ):
        """
        Create the core interactive plot for downstream processing.
        This function generates the main interactive plot using Plotly
        that contains the spatial scatter plot with annotations and
        image configuration.

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

        """

        spatial_coords = adata.obsm['spatial']

        extract_columns_raw = []

        for item in annotations:
            extract_columns_raw.append(adata.obs[item])

        extract_columns = []

        # The `extract_columns` list is needed for generating Plotly images
        # because it stores transformed annotation data. These annotations
        # are added as columns in the DataFrame (`df`) and are used as inputs
        # for the `color` and `hover_data` parameters in the Plotly scatter
        # plot. This enables the plot to visually encode annotations, providing
        # better insights into the spatial data. Without `extract_columns`, the
        # plot would lack essential annotation-based differentiation and
        # interactivity.

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
        # This is necessary to ensure that the color attribute
        # does not interfere with subsequent plots
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

    def generate_and_update_image(
        adata,
        title,
        color_mapping=None,
        **kwargs
    ):
        """
        This function generates the main figure with annotations and
        optional stratifications or color mappings, providing flexibility
        for detailed visualizations. It processes data, groups it by
        annotations, and enables advanced legend handling and styling.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix containing either the full dataset
            or a subset of the data.
        title : str
            Title for the plot.
        stratify_by : str, optional
            Column to stratify the plot. Default is None.
        color_mapping : dict, optional
            Color mapping for specific labels. Default is None.

        Returns
        -------
        dict
            A dictionary with "image_name" and "image_object" keys.
        """
        main_fig_parent = main_figure_generation(
            adata,
            annotations=annotations,
            dot_size=dot_size,
            dot_transparancy=dot_transparancy,
            colorscale=colorscale,
            figure_width=figure_width,
            figure_height=figure_height,
            figure_dpi=figure_dpi,
            font_size=font_size,
            **kwargs
        )

        # Create a copy of the figure for non-destructive updates
        main_fig_copy = copy.copy(main_fig_parent)
        data = main_fig_copy.data
        main_fig_parent.data = []

        # Prepare to track updates and manage grouped annotations
        updated_index = []
        legend_list = [
            f"legend{i+1}" if i > 0 else "legend"
            for i in range(len(annotations))
        ]
        previous_group = None

        # Process each trace in the figure for grouping and legends
        indices = list(range(len(data)))
        for item in indices:
            cat_label = data[item]['customdata'][0][0]
            cat_dataset = pd.DataFrame(
                {'X': data[item]['x'], 'Y': data[item]['y']}
            )

            # Assign the label to the appropriate legend group
            for i, legend_group in enumerate(annotations):
                if cat_label.startswith(legend_group):
                    cat_leg_group = f"<b>{legend_group}</b>"
                    cat_label = cat_label[len(legend_group) + 1:]
                    cat_group = legend_list[i]

            # Add a new legend entry if this group hasn't been encountered
            if previous_group is None or cat_group != previous_group:
                main_fig_parent.add_trace(go.Scattergl(
                    x=[data[item]['x'][0]],
                    y=[data[item]['y'][0]],
                    name=cat_leg_group,
                    mode="markers",
                    showlegend=True,
                    marker=dict(
                        color="white",
                        colorscale=None,
                        size=0,
                        opacity=0
                    )
                ))
                previous_group = cat_group

            # Add the category label to the dataset for grouping
            cat_dataset['label'] = cat_label

            main_fig_parent.add_trace(go.Scattergl(
                x=cat_dataset['X'],
                y=cat_dataset['Y'],
                name=cat_label,
                mode="markers",
                showlegend=True,
                marker=dict(
                    colorscale=colorscale,
                    size=dot_size,
                    opacity=dot_transparancy
                )
            ))

            updated_index.append(cat_label)

        if color_mapping is not None:
            main_fig_copy = copy.copy(main_fig_parent)
            data = main_fig_copy.data
            main_fig_parent.data = []

            for trace in data:
                trace_name = trace["name"]
                if color_mapping is not None:
                    if trace_name in color_mapping.keys():
                        trace['marker']['color'] = color_mapping[trace_name]
                main_fig_parent.add_trace(trace)

        main_fig_parent.update_layout(
            title={
                'text': title,
                'font': {'size': font_size},
                'xanchor': 'center',
                'yanchor': 'top',
                'x': 0.5,
                'y': 0.99
            },
            legend={
                'x': 1.05,
                'y': 0.5,
                'xanchor': 'left',
                'yanchor': 'middle'
            },
            margin=dict(l=5, r=5, t=font_size*2, b=5)
        )

        return {
            "image_name": f"{spell_out_special_characters(title)}.html",
            "image_object": main_fig_parent
        }


    #####################
    ## Main Code Block ##
    #####################

    results = []
    if defined_color_map:
        color_dict = adata.uns[defined_color_map]
    else:
        unique_ann_labels = np.unique(adata.obs[annotations].values)
        color_dict = color_mapping(
            unique_ann_labels,
            color_map=colorscale,
            rgba_mode=False,
            return_dict=True
        )

    if stratify_by is not None:
        unique_stratification_values = adata.obs[stratify_by].unique()

        for strat_value in unique_stratification_values:
            condition = adata.obs[stratify_by] == strat_value
            title = f"Highlighting {stratify_by}: {strat_value}"
            indices = np.where(condition)[0]
            selected_spatial = adata.obsm['spatial'][indices]
            print(f"number of cells in the region: {len(selected_spatial)}")

            adata_subset = select_values(
                data=adata,
                annotation=stratify_by,
                values=strat_value
            )

            result = generate_and_update_image(
                adata=adata_subset,
                title=title,
                stratify_by=stratify_by,
                color_mapping=color_dict,
                **kwargs
            )
            results.append(result)
    else:
        title = "Interactive Spatial Plot"
        result = generate_and_update_image(
            adata=adata,
            title=title,
            stratify_by=None,
            color_mapping=color_dict,
            **kwargs
        )
        results.append(result)

    return results


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


def relational_heatmap(
        adata: anndata.AnnData,
        source_annotation: str,
        target_annotation: str,
        color_map: str = "mint",
        **kwargs
):
    """
    Generates a relational heatmap from the given AnnData object.
    The color map refers to matplotlib color maps, default is mint.
    For more information on colormaps, see:
    https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix.
    source_annotation : str
        The source annotation to use for the relational heatmap.
    target_annotation : str
        The target annotation to use for the relational heatmap.
    color_map : str
        The color map to use for the relational heatmap. Default is mint.
    **kwargs : dict, optional
        Additional keyword arguments. For example, you can pass font_size=12.0.

    Returns
    -------
    dict
        A dictionary containing:
        - "figure" (plotly.graph_objs._figure.Figure):
            The generated relational heatmap as a Plotly figure.
        - "file_name" (str):
            The name of the file where the relational matrix can be saved.
        - "data" (pandas.DataFrame):
            A relational matrix DataFrame with percentage values.
            Rows represent source annotations,
            columns represent target annotations,
            and an additional "total" column sums
            the percentages for each source.
    """
    # Default font size
    font_size = kwargs.get('font_size', 12.0)
    prefix = kwargs.get('prefix', True)

    # Get the relationship between source and target annotations

    label_relations = annotation_category_relations(
            adata=adata,
            source_annotation=source_annotation,
            target_annotation=target_annotation,
            prefix=prefix
        )

    # Pivot the data to create a matrix for the heatmap
    heatmap_matrix = label_relations.pivot(
        index='source',
        columns='target',
        values='percentage_source'
    )

    heatmap_matrix = heatmap_matrix.fillna(0)

    x = list(heatmap_matrix.columns)
    y = list(heatmap_matrix.index)

    # Create text labels for the heatmap
    label_relations['text_label'] = [
        '{}%'.format(val) for val in label_relations["percentage_source"]
    ]

    heatmap_matrix2 = label_relations.pivot(
        index='source',
        columns='target',
        values='percentage_source'
        )

    heatmap_matrix2 = heatmap_matrix2.fillna(0)

    hover_template = 'Source: %{z}%<br>Target: %{customdata}%<extra></extra>'
    # Ensure alignment of the text data with the heatmap matrix
    z = list()
    iter_list = list()
    for y_item in y:
        iter_list.clear()
        for x_item in x:
            z_data_point = label_relations[
                (
                    label_relations['target'] == x_item
                ) & (
                    label_relations['source'] == y_item
                )
            ]['percentage_source']
            iter_list.append(
                0 if len(z_data_point) == 0 else z_data_point.iloc[0]
            )
        z.append([_ for _ in iter_list])

    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=z,
        colorscale=color_map,
        customdata=heatmap_matrix2.values,
        hovertemplate=hover_template
    )

    fig.update_layout(
        overwrite=True,
        xaxis=dict(
            title=source_annotation,
            ticks="",
            dtick=1,
            side="top",
            gridcolor="rgb(0, 0, 0)",
            tickvals=list(range(len(x))),
            ticktext=x
        ),
        yaxis=dict(
            title=target_annotation,
            ticks="",
            dtick=1,
            ticksuffix="   ",
            tickvals=list(range(len(y))),
            ticktext=y
        ),
        margin=dict(
            l=5,
            r=5,
            t=font_size * 2,
            b=font_size * 2
        )
    )

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = font_size

    fig.update_xaxes(
        side="bottom",
        tickangle=90
    )

    # Data output section
    data = fig.data[0]
    layout = fig.layout
    # Create a DataFrame
    matrix = pd.DataFrame(data['customdata'])
    matrix.index=layout['yaxis']['ticktext']
    matrix.columns=layout['xaxis']['ticktext']
    matrix["total"] = matrix.sum(axis=1)
    matrix = matrix.fillna(0)

    # Display the DataFrame
    file_name = f"{source_annotation}_to_{target_annotation}" + \
                "_relation_matrix.csv"

    return {"figure": fig, "file_name": file_name, "data": matrix}


def plot_ripley_l(
        adata,
        phenotypes,
        annotation=None,
        regions=None,
        sims=False,
        return_df=False,
        **kwargs):
    """
    Plot Ripley's L statistic for multiple bins and different regions
    for a given pair of phenotypes.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing Ripley's L results in `adata.uns['ripley_l']`.
    phenotypes : tuple of str
        A tuple of two phenotypes: (center_phenotype, neighbor_phenotype).
    regions : list of str, optional
        A list of region labels to plot. If None, plot all available regions.
        Default is None.
    sims : bool, optional
        Whether to plot the simulation results. Default is False.
    return_df : bool, optional
        Whether to return the DataFrame containing the Ripley's L results.
    kwargs : dict, optional
        Additional keyword arguments to pass to `seaborn.lineplot`.

    Raises
    ------
    ValueError
        If the Ripley L results are not found in `adata.uns['ripley_l']`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object containing the plot, which can be further modified.
    df : pandas.DataFrame, optional
        The DataFrame containing the Ripley's L results, if `return_df` is True.

    Example
    -------
    >>> ax = plot_ripley_l(
    ...     adata,
    ...     phenotypes=('Phenotype1', 'Phenotype2'),
    ...     regions=['region1', 'region2'])
    >>> plt.show()

    This returns the `Axes` object for further customization and displays the plot.
    """

    # Retrieve the results from adata.uns['ripley_l']
    ripley_results = adata.uns.get('ripley_l')

    if ripley_results is None:
        raise ValueError(
            "Ripley L results not found in the analsyis."
        )

    # Filter the results for the specific pair of phenotypes
    filtered_results = ripley_results[
        (ripley_results['center_phenotype'] == phenotypes[0]) &
        (ripley_results['neighbor_phenotype'] == phenotypes[1])
    ]

    if filtered_results.empty:
        # Generate all unique combinations of phenotype pairs
        unique_pairs = ripley_results[
            ['center_phenotype', 'neighbor_phenotype']].drop_duplicates()
        raise ValueError(
            "No Ripley L results found for the specified pair of phenotypes."
            f'\nCenter Phenotype: "{phenotypes[0]}"'
            f'\nNeighbor Phenotype: "{phenotypes[1]}"'
            f"\nExisiting unique pairs: {unique_pairs}"
        )

    # If specific regions are provided, filter them, otherwise plot all regions
    if regions is not None:
        filtered_results = filtered_results[
            filtered_results['region'].isin(regions)]

    # Check if the results are emply after subsetting the regions
    if filtered_results.empty:
        available_regions = ripley_results['region'].unique()
        raise ValueError(
            f"No data available for the specified regions: {regions}. "
            f"Available regions: {available_regions}."
        )

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    plot_data = []

    # Plot Ripley's L for each region
    for _, row in filtered_results.iterrows():
        region = row['region']  # Region label

        if row['ripley_l'] is None:
            message = (
               f"Ripley L results not found for region: {region}"
               f"\n Message: {row['message']}"
            )
            logging.warning(
              message
            )
            print(message)
            continue
        n_center = row['ripley_l']['n_center']
        n_neighbors = row['ripley_l']['n_neighbor']
        n_cells = f"({n_center}, {n_neighbors})"
        area = row['ripley_l']['area']
        # Plot the Ripley L statistic for the region
        sns.lineplot(
            data=row['ripley_l']['L_stat'],
            x='bins',
            y='stats',
            label=f'{region}: {n_cells}, {int(area)}',
            ax=ax,
            **kwargs)

        # Prepare plotted data to return if return_df is True
        l_stat_data = row['ripley_l']['L_stat']
        for _, stat_row in l_stat_data.iterrows():
            plot_data.append({
                'region': region,
                'radius': stat_row['bins'],
                'ripley(radius)': stat_row['stats'],
                'region_area': area,
                'n_center': n_center,
                'n_neighbor': n_neighbors,
            })

        if sims:
            confidence_level = 95
            errorbar = ("pi", confidence_level)
            n_sims = row["n_simulations"]
            sns.lineplot(
                x="bins",
                y="stats",
                data=row["ripley_l"]["sims_stat"],
                errorbar=errorbar,
                label=f"Simulations({region}):{n_sims} runs",
                **kwargs
            )

    # Set labels, title, and grid
    ax.set_title(
        "Ripley's L Statistic for phenotypes:"
        f"({phenotypes[0]}, {phenotypes[1]})\n"
    )
    ax.legend(title='Regions:(center, neighbor), area', loc='upper left')
    ax.grid(True)

    # Set the horizontal axis lable
    ax.set_xlabel("Radii (pixels)")
    ax.set_ylabel("Ripley's L Statistic")

    if return_df:
        df = pd.DataFrame(plot_data)
        return fig, df

    return fig


def _prepare_spatial_distance_data(
    adata,
    annotation,
    stratify_by=None,
    spatial_distance='spatial_distance',
    distance_from=None,
    distance_to=None,
    log=False
):
    """
    Prepares a tidy DataFrame for nearest-neighbor (spatial distance) plotting.

    This function:
      1) Validates required parameters (annotation, distance_from).
      2) Retrieves the spatial distance matrix from
         `adata.obsm[spatial_distance]`.
      3) Merges annotation (and optional stratify column).
      4) Filters rows to the reference phenotype (`distance_from`).
      5) Subsets columns if `distance_to` is given;
         otherwise keeps all distances.
      6) Reshapes (melts) into long-form data:
         columns -> [cellid, group, distance].
      7) Applies optional log1p transform.

    The resulting DataFrame is suitable for plotting with tool like Seaborn.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix, containing distances in
        `adata.obsm[spatial_distance]`.
    annotation : str
        Column in `adata.obs` indicating cell phenotype or annotation.
    stratify_by : str, optional
        Column in `adata.obs` used to group/stratify data
        (e.g., image or sample).
    spatial_distance : str, optional
        Key in `adata.obsm` storing the distance DataFrame.
        Default 'spatial_distance'.
    distance_from : str
        Reference phenotype from which distances are measured. Required.
    distance_to : str or list of str, optional
        Target phenotype(s). If None, use all available phenotype distances.
    log : bool, optional
        If True, applies np.log1p transform to the 'distance' column, which is
        renamed to 'log_distance'.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
            - 'cellid': index of the cell from 'adata.obs'.
            - 'group': the target phenotype (column names of 'distance_map'.
            - 'distance': the numeric distance value.
            - 'phenotype': the reference phenotype ('distance_from').
            - 'stratify_by': optional grouping column, if provided.

    Raises
    ------
    ValueError
        If required parameters are missing, if phenotypes are not found in
        `adata.obs`, or if the spatial distance matrix is not available in
        `adata.obsm`.

    Examples
    --------
    >>> df_long = _prepare_spatial_distance_data(
    ...     adata=my_adata,
    ...     annotation='cell_type',
    ...     stratify_by='sample_id',
    ...     spatial_distance='spatial_distance',
    ...     distance_from='Tumor',
    ...     distance_to=['Stroma', 'Immune'],
    ...     log=True
    ... )
    >>> df_long.head()
    """

    # Validate required parameters
    if distance_from is None:
        raise ValueError(
            "Please specify the 'distance_from' phenotype. This indicates "
            "the reference group from which distances are measured."
        )
    check_annotation(adata, annotations=annotation)

    # Convert distance_to to list if needed
    if distance_to is not None and isinstance(distance_to, str):
        distance_to = [distance_to]

    phenotypes_to_check = [distance_from] + (
        distance_to if distance_to else []
    )

    # Ensure distance_from and distance_to exist in adata.obs[annotation]
    check_label(
        adata,
        annotation=annotation,
        labels=phenotypes_to_check,
        should_exist=True
    )

    # Retrieve the spatial distance matrix from adata.obsm
    if spatial_distance not in adata.obsm:
        raise ValueError(
            f"'{spatial_distance}' does not exist in the provided dataset. "
            "Please run 'calculate_nearest_neighbor' first to compute and "
            "store spatial distance. "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    distance_map = adata.obsm[spatial_distance].copy()

    # Verify requested phenotypes exist in the distance_map columns
    missing_cols = [
        p for p in phenotypes_to_check if p not in distance_map.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Phenotypes {missing_cols} not found in columns of "
            f"'{spatial_distance}'. Columns present: "
            f"{list(distance_map.columns)}"
        )

    # Validate 'stratify_by' column if provided
    if stratify_by is not None:
        check_annotation(adata, annotations=stratify_by)

    # Build a meta DataFrame with phenotype & optional stratify column
    meta_data = pd.DataFrame({'phenotype': adata.obs[annotation]},
                             index=adata.obs.index)
    if stratify_by:
        meta_data[stratify_by] = adata.obs[stratify_by]

    # Merge metadata with distance_map and filter for 'distance_from'
    df_merged = meta_data.join(distance_map, how='left')
    df_merged = df_merged[df_merged['phenotype'] == distance_from]
    if df_merged.empty:
        raise ValueError(
            f"No cells found with phenotype == '{distance_from}'."
        )

    # Reset index to ensure cell names are in a column called 'cellid'
    df_merged = df_merged.reset_index().rename(columns={'index': 'cellid'})

    # Prepare the list of metadata columns
    meta_cols = ['phenotype']
    if stratify_by:
        meta_cols.append(stratify_by)

    # Determine distance columns
    if distance_to:
        keep_cols = ['cellid'] + meta_cols + distance_to
    else:
        non_distance_cols = ['cellid', 'phenotype']
        if stratify_by:
            non_distance_cols.append(stratify_by)
        distance_columns = [
            c for c in df_merged.columns if c not in non_distance_cols
        ]
        keep_cols = ['cellid'] + meta_cols + distance_columns

    df_merged = df_merged[keep_cols]

    # Melt the DataFrame from wide to long format
    df_long = df_merged.melt(
        id_vars=['cellid'] + meta_cols,
        var_name='group',
        value_name='distance'
    )

    # Convert columns to categorical for consistency
    for col in ['group', 'phenotype', stratify_by]:
        if col and col in df_long.columns:
            df_long[col] = df_long[col].astype(str).astype('category')

    # Reorder categories for 'group' if 'distance_to' is provided
    if distance_to:
        df_long['group'] = df_long['group'].cat.reorder_categories(distance_to)
        df_long.sort_values('group', inplace=True)

    # Ensure 'distance' is numeric and apply log transform if requested
    df_long['distance'] = pd.to_numeric(df_long['distance'], errors='coerce')
    if log:
        df_long['distance'] = np.log1p(df_long['distance'])
        df_long.rename(columns={'distance': 'log_distance'}, inplace=True)

    # Reorder columns dynamically based on the presence of 'log'
    distance_col = 'log_distance' if log else 'distance'
    final_cols = ['cellid', 'group', distance_col, 'phenotype']
    if stratify_by is not None:
        final_cols.append(stratify_by)
    df_long = df_long[final_cols]

    return df_long


def _plot_spatial_distance_dispatch(
    df_long,
    method,
    plot_type,
    stratify_by=None,
    facet_plot=False,
    **kwargs
):
    """
    Decides the figure layout based on 'stratify_by' and 'facet_plot'
    and dispatches actual plotting calls.

    Logic:
      1) If stratify_by and facet_plot => single figure with subplots (faceted)
      2) If stratify_by and not facet_plot => multiple figures, one per group
      3) If stratify_by is None => single figure (no subplots)

    This function calls seaborn figure-level functions (catplot or displot).

    Parameters
    ----------
    df_long : pd.DataFrame
        Tidy DataFrame with columns ['cellid', 'group', 'distance',
        'phenotype', 'stratify_by'].
    method : {'numeric', 'distribution'}
        Determines which seaborn function is used (catplot or displot).
    plot_type : str
        For method='numeric': 'box', 'violin', 'boxen', etc.
        For method='distribution': 'hist', 'kde', 'ecdf', etc.
    stratify_by : str or None
        Column name for grouping. If None, no grouping is done.
    facet_plot : bool
        If True, subplots in a single figure (faceted).
        If False, separate figures (one per group) or a single figure.
    **kwargs
        Additional seaborn plotting arguments (e.g., col_wrap=2).

    Returns
    -------
    dict
        Dictionary with two keys:
            - "data": the DataFrame (df_long)
            - "fig": a Matplotlib Figure or a list of Figures

    Raises
    ------
    ValueError
        If 'method' is invalid (not 'numeric' or 'distribution').

    Examples
    --------
    Called internally by 'visualize_nearest_neighbor'. Typically not used
    directly by end users.
    """

    distance_col = kwargs.pop('distance_col', 'distance')
    hue_axis = kwargs.pop('hue_axis', None)

    if method not in ['numeric', 'distribution']:
        raise ValueError("`method` must be 'numeric' or 'distribution'.")

    # Set up the plotting function using partial
    if method == 'numeric':
        plot_func = partial(
            sns.catplot,
            data=None,
            x=distance_col,
            y='group',
            kind=plot_type
        )
    else:  # distribution
        plot_func = partial(
            sns.displot,
            data=None,
            x=distance_col,
            hue=hue_axis if hue_axis else None,
            kind=plot_type
        )

    # Helper to plot a single figure or faceted figure
    def _make_figure(data, **kws):
        g = plot_func(data=data, **kws)
        if distance_col == 'log_distance':
            x_label = "Log(Nearest Neighbor Distance)"
        else:
            x_label = "Nearest Neighbor Distance"

        # Set axis label based on whether log transform was applied
        if hasattr(g, 'set_axis_labels'):
            g.set_axis_labels(x_label, None)
        else:
            # Fallback if 'set_axis_labels' is unavailable
            plt.xlabel(x_label)

        return g.fig

    figures = []

    # Branching logic for figure creation
    if stratify_by and facet_plot:
        # Single figure with faceted subplots (col=stratify_by)
        fig = _make_figure(df_long, col=stratify_by, **kwargs)
        figures.append(fig)

    elif stratify_by and not facet_plot:
        # Multiple separate figures, one per unique value in stratify_by
        categories = df_long[stratify_by].unique()
        for cat in categories:
            subset = df_long[df_long[stratify_by] == cat]
            fig = _make_figure(subset, **kwargs)
            figures.append(fig)
    else:
        # Single figure (no subplots)
        fig = _make_figure(df_long, **kwargs)
        figures.append(fig)

    # Return dictionary: { 'data': DataFrame, 'fig': Figure(s) }
    result = {"data": df_long}
    if len(figures) == 1:
        result["fig"] = figures[0]
    else:
        result["fig"] = figures
    return result


def visualize_nearest_neighbor(
    adata,
    annotation,
    stratify_by=None,
    spatial_distance='spatial_distance',
    distance_from=None,
    distance_to=None,
    facet_plot=False,
    plot_type=None,
    log=False,
    method=None,
    **kwargs
):
    """
    Visualize nearest-neighbor (spatial distance) data between groups of cells
    as numeric or distribution plots.

    This user-facing function assembles the data by calling
    `_prepare_spatial_distance_data` and then creates plots through
    `_plot_spatial_distance_dispatch`.

    Plot arrangement logic:
      1) If stratify_by is not None and facet_plot=True => single figure
         with subplots (faceted).
      2) If stratify_by is not None and facet_plot=False => multiple separate
         figures, one per group.
      3) If stratify_by is None => a single figure with one plot.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with distances in `adata.obsm[spatial_distance]`.
    annotation : str
        Column in `adata.obs` containing cell phenotypes or annotations.
    stratify_by : str, optional
        Column in `adata.obs` used to group or stratify data (e.g. imageid).
    spatial_distance : str, optional
        Key in `adata.obsm` storing the distance DataFrame. Default is
        'spatial_distance'.
    distance_from : str
        Reference phenotype from which distances are measured. Required.
    distance_to : str or list of str, optional
        Target phenotype(s) to measure distance to. If None, uses all
        available phenotypes.
    facet_plot : bool, optional
        If True (and stratify_by is not None), subplots in a single figure.
        Else, multiple or single figure(s).
    plot_type : str, optional
        For method='numeric': 'box', 'violin', 'boxen', etc.
        For method='distribution': 'hist', 'kde', 'ecdf', etc.
    log : bool, optional
        If True, applies np.log1p transform to the distance values.
    method : {'numeric', 'distribution'}
        Determines the plotting style (catplot vs displot).
    **kwargs : dict
        Additional arguments for seaborn figure-level functions.

    Returns
    -------
    dict
        {
            "data": pd.DataFrame,  # Tidy DataFrame used for plotting
            "fig": Figure or list[Figure]  # Single or multiple figures
        }

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.

    Examples
    --------
    >>> # Numeric box plot comparing Tumor distances to multiple targets
    >>> res = visualize_nearest_neighbor(
    ...     adata=my_adata,
    ...     annotation='cell_type',
    ...     stratify_by='sample_id',
    ...     spatial_distance='spatial_distance',
    ...     distance_from='Tumor',
    ...     distance_to=['Stroma', 'Immune'],
    ...     facet_plot=True,
    ...     plot_type='box',
    ...     method='numeric'
    ... )
    >>> df_long, fig = res["data"], res["fig"]

    >>> # Distribution plot (kde) for a single target, single figure
    >>> res2 = visualize_nearest_neighbor(
    ...     adata=my_adata,
    ...     annotation='cell_type',
    ...     distance_from='Tumor',
    ...     distance_to='Stroma',
    ...     method='distribution',
    ...     plot_type='kde'
    ... )
    >>> df_dist, fig2 = res2["data"], res2["fig"]
    """

    if distance_from is None:
        raise ValueError(
            "Please specify the 'distance_from' phenotype. It indicates "
            "the reference group from which distances are measured."
        )
    if method not in ['numeric', 'distribution']:
        raise ValueError(
            "Invalid 'method'. Please choose 'numeric' or 'distribution'."
        )

    df_long = _prepare_spatial_distance_data(
        adata=adata,
        annotation=annotation,
        stratify_by=stratify_by,
        spatial_distance=spatial_distance,
        distance_from=distance_from,
        distance_to=distance_to,
        log=log
    )

    # Determine plot_type if not provided
    if plot_type is None:
        plot_type = 'boxen' if method == 'numeric' else 'kde'

    # If log=True, the column name is 'log_distance', else 'distance'
    distance_col = 'log_distance' if log else 'distance'

    # Dispatch to the plot logic
    result_dict = _plot_spatial_distance_dispatch(
        df_long=df_long,
        method=method,
        plot_type=plot_type,
        stratify_by=stratify_by,
        facet_plot=facet_plot,
        distance_col=distance_col,
        **kwargs
    )

    return result_dict
