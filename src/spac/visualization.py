import seaborn as sns
import seaborn
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def boxplot(adata, annotation=None, layer=None, ax=None,
            features=None, **kwargs):
    """
    Plot boxplots for all features available in the passed adata object or
    a subset if features are provided.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.

    annotation : str, optional
        Annotation to determine if separate plots are needed for every label.

    layer : str, optional
        The name of the matrix layer to use. If not provided,
        uses the main data matrix adata.X.

    ax : matplotlib.axes.Axes, optional
        An existing Axes object to draw the plot onto, optional.

    features : list, optional
        List of feature names to be plotted.
        If not provided, all features will be plotted.

    **kwargs
        Additional arguments to pass to seaborn.boxplot.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The created figure and axes for the plot.
    """

    # Validate inputs
    if not isinstance(adata, anndata.AnnData):
        raise TypeError("Input 'adata' must be an instance "
              "of anndata.AnnData.")

    if ax and not isinstance(ax, plt.Axes):
        raise TypeError("Input 'ax' must be a matplotlib.axes.Axes object.")

    if annotation and annotation not in adata.obs.columns:
        raise ValueError(f"Specified annotation '{annotation}' not found in "
                         "the provided AnnData object's .obs.")

    if layer and layer not in adata.layers:
        raise ValueError(f"Specified layer '{layer}' not found in the "
                         "provided AnnData object.")

    # Use the specified layer if provided
    if layer:
        data_matrix = adata.layers[layer]
    else:
        data_matrix = adata.X

    # Create a DataFrame from the data matrix with features as columns
    df = pd.DataFrame(data_matrix, columns=adata.var_names)

    # If annotation is provided, add it to the DataFrame
    if annotation:
        df[annotation] = adata.obs[annotation].values

    # Filter the DataFrame based on provided features
    if features:
        if not all(feature in df.columns for feature in features):
            raise ValueError("One or more provided features are not "
                             "found in the data.")
        df = df[features + ([annotation] if annotation else [])]

    # If ax is provided, get the associated figure
    if ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figsize as needed

    # If annotation is provided, loop through unique labels and plot separately
    if annotation:
        unique_labels = df[annotation].unique()
        for label in unique_labels:
            data_subset = df[df[annotation] == label]
            melted_data = data_subset.melt(id_vars=annotation)
            sns.boxplot(data=melted_data, x="variable", y="value", ax=ax,
                        **kwargs)
            ax.set_title(f"Features for Annotation: {label}")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()
    else:
        melted_data = df.melt()
        sns.boxplot(data=melted_data, x="variable", y="value", ax=ax, **kwargs)
        ax.set_title("Features Boxplot")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

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


def histogram(adata, feature_name=None, annotation_name=None, layer=None,
              group_by=None, together=False, ax=None, **kwargs):
    """
    Plot the histogram of cells based on a specific feature from adata.X
    or annotation from adata.obs.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.

    feature_name : str, optional
        Name of continuous feature from adata.X to plot its histogram.

    annotation_name : str, optional
        Name of the annotation from adata.obs to plot its histogram.

    group_by : str, default None
        Choose either to group the histogram by another column.

    together : bool, default False
        If True, and if group_by !=None create one plot for all groups.
        Otherwise, divide every histogram by the number of elements.

    ax : matplotlib.axes.Axes, optional
        An existing Axes object to draw the plot onto, optional.

    **kwargs
        Parameters passed to seaborn histplot function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes of the histogram plot.

    fig : matplotlib.figure.Figure
        The created figure for the plot.

    """
    # Validate inputs
    err = "adata must be an instance of anndata.AnnData, not {type(adata)}."
    if not isinstance(adata, anndata.AnnData):
        raise TypeError(err)

    df = adata.to_df()
    df = pd.concat([df, adata.obs], axis=1)

    if feature_name and annotation_name:
        raise ValueError("Cannot pass both feature_name and annotation_name,"
                         " choose one.")

    if feature_name:
        if feature_name not in df.columns:
            raise ValueError("feature_name not found in adata.")
        x = feature_name

    if annotation_name:
        if annotation_name not in df.columns:
            raise ValueError("annotation_name not found in adata.")
        x = annotation_name

    if group_by and group_by not in df.columns:
        raise ValueError("group_by not found in adata.")

    if ax is not None:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()

    if group_by:
        groups = df[group_by].dropna().unique().tolist()
        n_groups = len(groups)
        if n_groups == 0:
            raise ValueError("There must be at least one group to create a"
                             " histogram.")
        if together:
            colors = sns.color_palette("hsv", n_groups)
            sns.histplot(data=df.dropna(), x=x, hue=group_by, multiple="stack",
                         palette=colors, ax=ax, **kwargs)
            return fig, ax
        else:
            fig, axs = plt.subplots(n_groups, 1, figsize=(5, 5*n_groups))
            if n_groups == 1:
                axs = [axs]
            for i, ax_i in enumerate(axs):
                sns.histplot(data=df[df[group_by] == groups[i]].dropna(),
                             x=x, ax=ax_i, **kwargs)
                ax_i.set_title(groups[i])
            return fig, axs

    sns.histplot(data=df, x=x, ax=ax, **kwargs)
    return fig, ax


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


def threshold_heatmap(adata, feature_cutoffs, annotation):
    """
    Creates a heatmap for each feature, categorizing intensities into low,
    medium, and high based on provided cutoffs.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing the feature intensities in .X attribute.
    feature_cutoffs : dict
        Dictionary with feature names as keys and tuples with two intensity
        cutoffs as values.
    annotation : str Column name in .obs DataFrame
        that contains the annotation used for grouping.

    Returns
    -------
    Dictionary of :class:`~matplotlib.axes.Axes`
        A dictionary contains the axes of figures generated in the scanpy
        heatmap function.
        Consistent Key: 'heatmap_ax'
        Potential Keys includes: 'groupby_ax', 'dendrogram_ax', and
        'gene_groups_ax'.

    """

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
        swap_axes=True,
        show=False
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
