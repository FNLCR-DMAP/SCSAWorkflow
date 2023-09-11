import seaborn as sns
import seaborn
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from spac.utils import check_table, check_annotation, check_feature


def dimensionality_reduction_plot(adata, method, annotation=None, feature=None,
                                  layer=None, ax=None, **kwargs):
    """
    Visualize scatter plot in t-SNE or UMAP basis.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object with coordinates precomputed by the 'tsne' or 'UMAP'
        function and stored in 'adata.obsm["X_tsne"]' or 'adata.obsm["X_umap"]'
    method : str
        Dimensionality reduction method to visualize.
        Choose from {'tsne', 'umap'}.
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
        Parameters passed to scanpy.pl.tsne or scanpy.pl.umap function.

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
    check_table(adata, tables=layer)
    if annotation:
        check_annotation(adata, annotations=annotation)
    if feature:
        check_feature(adata, features=[feature])

    # Validate the method and check if the necessary data exists in adata.obsm
    if method == 'umap':
        key = 'X_umap'
    elif method == 'tsne':
        key = 'X_tsne'
    else:
        raise ValueError("Method should be one of {'tsne', 'umap'}.")

    if key not in adata.obsm.keys():
        error_msg = (
            f"{key} coordinates not found in adata.obsm."
            f"Please run {method.upper()} before calling this function."
        )
        raise ValueError(error_msg)

    # Determine coloring scheme
    color = None
    if annotation:
        color = annotation
    elif feature:
        color = feature

    # If a layer is provided, use it for visualization
    if layer:
        adata.X = adata.layers[layer]

    # Add color column to the kwargs for the scanpy plot
    kwargs['color'] = color

    # Plot the chosen method
    if method == 'tsne':
        sc.pl.tsne(adata, ax=ax, **kwargs)
    else:
        sc.pl.umap(adata, ax=ax, **kwargs)

    fig = plt.gcf()  # Get the current figure
    if ax is None:  # If no ax was provided, get the current ax
        ax = plt.gca()

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


def hierarchical_heatmap(adata, annotation, layer=None, dendrogram=True,
                         standard_scale=None, ax=None, **kwargs):
    """
    Generates a hierarchical clustering heatmap.
    Cells are stratified by `annotation`,
    then mean intensities are calculated for each feature across all cells
    to plot the heatmap using scanpy.tl.dendrogram and sc.pl.matrixplot.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    annotation : str
        Name of the annotation in adata.obs to group by and calculate mean
        intensity.
    layer : str, optional
        The name of the `adata` layer to use to calculate the mean intensity.
        Default is None.
    dendrogram : bool, optional
        If True, a dendrogram based on the hierarchical clustering between
        the `annotation` categories is computed and plotted. Default is True.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new figure and axes
        object will be created. Default is None.
    **kwargs:
        Additional parameters passed to sc.pl.matrixplot function.

    Returns
    ----------
    mean_intensity : pandas.DataFrame
        A DataFrame containing the mean intensity of cells for each
        annotation.
    matrixplot : scanpy.pl.matrixplot
        A Scanpy matrixplot object.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from spac.visualization import hierarchical_heatmap
    >>> import anndata

    >>> X = pd.DataFrame([[1, 2], [3, 4]], columns=['gene1', 'gene2'])
    >>> annotation = pd.DataFrame(['type1', 'type2'], columns=['cell_type'])
    >>> all_data = anndata.AnnData(X=X, obs=annotation)

    >>> fig, ax = plt.subplots()  # Create a new figure and axes object
    >>> mean_intensity, matrixplot = hierarchical_heatmap(all_data,
    ...                                                   "cell_type",
    ...                                                   layer=None,
    ...                                                   standard_scale='var',
    ...                                                   ax=None)
    # Display the figure
    # matrixplot.show()
    """

    # Check if annotation exists in adata
    if annotation not in adata.obs.columns:
        msg = (f"The annotation '{annotation}' does not exist in the "
               f"provided AnnData object. Available annotations are: "
               f"{list(adata.obs.columns)}")
        raise KeyError(msg)

    # Check if the layer exists in adata
    if layer and layer not in adata.layers.keys():
        msg = (f"The layer '{layer}' does not exist in the "
               f"provided AnnData object. Available layers are: "
               f"{list(adata.layers.keys())}")
        raise KeyError(msg)

    # Raise an error if there are any NaN values in the annotation column
    if adata.obs[annotation].isna().any():
        raise ValueError("NaN values found in annotation column.")

    # Calculate mean intensity
    intensities = adata.to_df(layer=layer)
    labels = adata.obs[annotation]
    grouped = pd.concat([intensities, labels], axis=1).groupby(annotation)
    mean_intensity = grouped.mean()

    # Reset the index of mean_feature
    mean_intensity = mean_intensity.reset_index()

    # Convert mean_intensity to AnnData
    mean_intensity_adata = sc.AnnData(
        X=mean_intensity.iloc[:, 1:].values,
        obs=pd.DataFrame(
            index=mean_intensity.index,
            data={
                annotation: mean_intensity.iloc[:, 0]
                .astype('category').values
            }
        ),
        var=pd.DataFrame(index=mean_intensity.columns[1:])
    )

    # Compute dendrogram if needed
    if dendrogram:
        sc.tl.dendrogram(
            mean_intensity_adata,
            groupby=annotation,
            var_names=mean_intensity_adata.var_names,
            n_pcs=None
        )

    # Create the matrix plot
    matrixplot = sc.pl.matrixplot(
        mean_intensity_adata,
        var_names=mean_intensity_adata.var_names,
        groupby=annotation, use_raw=False,
        dendrogram=dendrogram,
        standard_scale=standard_scale, cmap="viridis",
        return_fig=True, ax=ax, show=False, **kwargs
    )
    return mean_intensity, matrixplot


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

    if feature is not None:

        feature_index = feature_names.index(feature)
        feature_annotation = feature + "spatial_plot"
        if vmin == -999:
            vmin = np.min(layer[:, feature_index])
        if vmax == -999:
            vmax = np.max(layer[:, feature_index])
        adata.obs[feature_annotation] = layer[:, feature_index]
        color_region = feature_annotation
    else:
        color_region = annotation
        vmin = None
        vmax = None

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    print(feature)
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
    plt.show()

    return fig, ax
