import seaborn as sns
import scanpy as sc
import seaborn
import pandas as pd
import numpy as np
import anndata
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def histogram(
    adata, observation, group_by=None,
    together=False, ax=None, **kwargs
):
    """
    Plot histogram of cells based on specific observation using seaborn.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    observation : str
        Name of member of adata.obs to plot the histogram.

    group_by : str, default None
        Choose either to group the histogram by another column.

    together : bool, default False
        If True, and if group_by !=None  create one plot for all groups.
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
    if not isinstance(adata, anndata.AnnData):
        raise TypeError(
            f"adata must be an instance of anndata.AnnData,"
            f" not {type(adata)}."
        )

    if observation not in adata.obs.columns:
        raise ValueError(
            f"observation '{observation}'"
            " not found in adata.obs."
        )

    if not pd.api.types.is_numeric_dtype(adata.obs[observation]):
        raise TypeError(
            f"observation '{observation}'"
            " must be a numeric data type."
        )

    if pd.isnull(adata.obs[observation]).any():
        raise ValueError(
            f"observation '{observation}' contains NaN values."
        )

    if group_by is not None:
        if group_by not in adata.obs.columns:
            raise ValueError(
                f"group_by '{group_by}'"
                " not found in adata.obs."
            )

        if not pd.api.types.is_categorical_dtype(
                adata.obs[group_by]) and not pd.api.types.is_object_dtype(
                adata.obs[group_by]):
            raise TypeError(
                f"group_by '{group_by}' must be a categorical"
                " or object data type."
            )

        if adata.obs[group_by].isna().any():
            raise ValueError(
                f"group_by '{group_by}' contains None values."
            )

    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError(
            f"ax must be an instance of matplotlib.axes.Axes,"
            f" not {type(ax)}."
        )

    if ax is not None:
        fig = ax.get_figure()  # Get the figure associated with the Axes
    else:
        fig, ax = plt.subplots()

    df = adata.obs.copy()

    if group_by is not None and pd.api.types.is_categorical_dtype(
            adata.obs[group_by]):
        df[group_by] = df[group_by].astype('string')

    if group_by is not None:
        groups = df[group_by].unique().tolist()

        if together:
            if ax is None:
                fig, ax = plt.subplots()
            sns.histplot(
                data=df, x=observation, hue=group_by,
                multiple="stack", ax=ax, **kwargs
            )
            ax.set_title(observation)
            return ax, fig

        else:
            n_groups = len(groups)
            if ax is None:
                fig, axs = plt.subplots(n_groups, 1)
            else:
                fig, axs = plt.subplots(n_groups, 1, num=fig.number)
            fig.tight_layout(pad=1)
            fig.set_figwidth(5)
            fig.set_figheight(5*n_groups)

            for i, ax in enumerate(axs):
                sns.histplot(
                    data=df[df[group_by] == groups[i]],
                    x=observation, ax=ax, **kwargs
                )
                ax.set_title(groups[i])
            return axs, fig

    else:
        if ax is None:
            fig, ax = plt.subplots()
        sns.histplot(data=df, x=observation, ax=ax, **kwargs)
        ax.set_title(observation)
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


def hierarchical_heatmap(adata, observation, layer=None, dendrogram=True,
                         standard_scale=None, ax=None, **kwargs):
    """
    Generates a hierarchical clustering heatmap.
    Cells are stratified by `observation`,
    then mean intensities are calculated for each feature across all cells
    to plot the heatmap using scanpy.tl.dendrogram and sc.pl.matrixplot.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    observation : str
        Name of the observation in adata.obs to group by and calculate mean
        intensity.
    layer : str, optional
        The name of the `adata` layer to use to calculate the mean intensity.
        Default is None.
    dendrogram : bool, optional
        If True, a dendrogram based on the hierarchical clustering between
        the `observation` categories is computed and plotted. Default is True.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new figure and axes
        object will be created. Default is None.
    **kwargs:
        Additional parameters passed to sc.pl.matrixplot function.

    Returns
    ----------
    mean_intensity : pandas.DataFrame
        A DataFrame containing the mean intensity of cells for each
        observation.
    matrixplot : scanpy.pl.matrixplot
        A Scanpy matrixplot object.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from spac.visualization import hierarchical_heatmap
    >>> import anndata

    >>> X = pd.DataFrame([[1, 2], [3, 4]], columns=['gene1', 'gene2'])
    >>> obs = pd.DataFrame(['type1', 'type2'], columns=['cell_type'])
    >>> all_data = anndata.AnnData(X=X, obs=obs)

    >>> fig, ax = plt.subplots()  # Create a new figure and axes object
    >>> mean_intensity, matrixplot = hierarchical_heatmap(all_data,
    ...                                                   "cell_type",
    ...                                                   layer=None,
    ...                                                   standard_scale='var',
    ...                                                   ax=None)
    # Display the figure
    # matrixplot.show()
    """

    # Check if observation exists in adata
    if observation not in adata.obs.columns:
        msg = (f"The observation '{observation}' does not exist in the "
               f"provided AnnData object. Available observations are: "
               f"{list(adata.obs.columns)}")
        raise KeyError(msg)

    # Check if the layer exists in adata
    if layer and layer not in adata.layers.keys():
        msg = (f"The layer '{layer}' does not exist in the "
               f"provided AnnData object. Available layers are: "
               f"{list(adata.layers.keys())}")
        raise KeyError(msg)

    # Raise an error if there are any NaN values in the observation column
    if adata.obs[observation].isna().any():
        raise ValueError("NaN values found in observation column.")

    # Calculate mean intensity
    intensities = adata.to_df(layer=layer)
    labels = adata.obs[observation]
    grouped = pd.concat([intensities, labels], axis=1).groupby(observation)
    mean_intensity = grouped.mean()

    # Reset the index of mean_feature
    mean_intensity = mean_intensity.reset_index()

    # Convert mean_intensity to AnnData
    mean_intensity_adata = sc.AnnData(
        X=mean_intensity.iloc[:, 1:].values,
        obs=pd.DataFrame(
            index=mean_intensity.index,
            data={
                observation: mean_intensity.iloc[:, 0]
                .astype('category').values
            }
        ),
        var=pd.DataFrame(index=mean_intensity.columns[1:])
    )

    # Compute dendrogram if needed
    if dendrogram:
        sc.tl.dendrogram(
            mean_intensity_adata,
            groupby=observation,
            var_names=mean_intensity_adata.var_names,
            n_pcs=None
        )

    # Create the matrix plot
    matrixplot = sc.pl.matrixplot(
        mean_intensity_adata,
        var_names=mean_intensity_adata.var_names,
        groupby=observation, use_raw=False,
        dendrogram=dendrogram,
        standard_scale=standard_scale, cmap="viridis",
        return_fig=True, ax=ax, show=False, **kwargs
    )
    return mean_intensity, matrixplot


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
        cutoffs as values.
    observation : str Column name in .obs DataFrame
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

    # Assert observation is a string
    if not isinstance(observation, str):
        err_type = type(observation).__name__
        err_msg = (f'Observation should be string. Got {err_type}.')
        raise TypeError(err_msg)

    # Assert observation is a column in adata.obs DataFrame
    if observation not in adata.obs.columns:
        err_msg = f"'{observation}' not found in adata.obs DataFrame."
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


def spatial_plot(
        adata,
        spot_size,
        alpha,
        vmin=-999,
        vmax=-999,
        observation=None,
        feature=None,
        table=None,
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
    observation : str
        The observation to visualize in the spatial plot.
        Can't be set with feature, default None.
    table : str
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

    err_msg_table = "The 'table' parameter must be a string, " + \
        f"got {str(type(table))}"
    err_msg_feature = "The 'feature' parameter must be a string, " + \
        f"got {str(type(feature))}"
    err_msg_observation = "The 'observation' parameter must be a string, " + \
        f"got {str(type(observation))}"
    err_msg_feat_obs_coe = "Both observation and feature are passed, " + \
        "please provide sinle input."
    err_msg_feat_obs_non = "Both observation and feature are None, " + \
        "please provide sinle input."
    err_msg_spot_size = "The 'spot_size' parameter must be an integerm, " + \
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

    if table is not None and not isinstance(table, str):
        raise ValueError(err_msg_table)

    if table is not None and table not in adata.layers.keys():
        err_msg_table_exist = f"Table {table} does not exists, " + \
            f"available tables are {str(adata.layers.keys())}"
        raise ValueError(err_msg_table_exist)

    if feature is not None and not isinstance(feature, str):
        raise ValueError(err_msg_feature)

    if observation is not None and not isinstance(observation, str):
        raise ValueError(err_msg_observation)

    if observation is not None and feature is not None:
        raise ValueError(err_msg_feat_obs_coe)

    if observation is None and feature is None:
        raise ValueError(err_msg_feat_obs_non)

    if 'spatial' not in adata.obsm_keys():
        err_msg = "Spatial coordinates not found in the 'obsm' attribute."
        raise ValueError(err_msg)

    # Extract obs name
    obs_names = adata.obs.columns.tolist()
    obs_names_str = ",".join(obs_names)

    if observation is not None and observation not in obs_names:
        error_text = f"Observation {observation} not found in the dataset." + \
            f"existing observations are: {obs_names_str}"
        raise ValueError(error_text)

    # Extract feature name
    if table is None:
        layer = adata.X
    else:
        layer = adata.layers[table]

    feature_names = adata.var_names.tolist()
    feature_names_str = ",".join(feature_names)

    if feature is not None and feature not in feature_names:
        error_text = f"Feature {feature} not found," + \
            f"existing features are: {feature_names_str}"
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
        color_region = feature
        feature_index = feature_names.index(feature)
        if vmin == -999:
            vmin = np.min(layer[:, feature_index])
        if vmax == -999:
            vmax = np.max(layer[:, feature_index])
    else:
        color_region = observation
        vmin = None
        vmax = None

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax = sc.pl.spatial(
        adata,
        layer=table,
        color=color_region,
        spot_size=spot_size,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        show=False,
        **kwargs)

    return ax
