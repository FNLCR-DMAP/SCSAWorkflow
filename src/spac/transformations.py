import numpy as np
from numpy import arcsinh
import scanpy as sc
import pandas as pd
import anndata
import warnings
import logging
import scanpy.external as sce
from spac.utils import check_table, check_annotation, check_feature
from scipy import stats
from scipy.sparse import issparse


def phenograph_clustering(adata, features, layer=None, k=50, seed=None):
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

    layer : str, optional
        The layer to be used in calculating the phengraph clusters.

    k : int, optional
        The number of nearest neighbor to be used in creating the graph.

    seed : int, optional
        Random seed for reproducibility.
    """

    # Use utility functions for input validation
    check_table(adata, tables=layer)
    check_feature(adata, features=features)

    if not isinstance(k, int) or k <= 0:
        raise ValueError("`k` must be a positive integer")

    if seed is not None:
        np.random.seed(seed)

    if layer is not None:
        phenograph_df = adata.to_df(layer=layer)[features]
    else:
        phenograph_df = adata.to_df()[features]

    phenograph_out = sce.tl.phenograph(phenograph_df,
                                       clustering_algo="leiden",
                                       k=k,
                                       seed=seed)

    adata.obs["phenograph"] = pd.Categorical(phenograph_out[0])
    adata.uns["phenograph_features"] = features


def get_cluster_info(adata, annotation="phenograph", features=None):
    """
    Retrieve information about clusters based on specific annotation.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    annotation : str, optional
        Annotation/column in adata.obs for cluster info.
    features : list of str, optional
        Features (e.g., genes) for cluster metrics.
        Defaults to all features in adata.var_names.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each cluster.
    """

    # Use utility functions for input validation
    check_annotation(adata, annotations=annotation)

    if features is None:
        features = list(adata.var_names)
    else:
        check_feature(adata, features=features)

    # Count cells in each cluster
    cluster_counts = adata.obs[annotation].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Number of Cells"]

    # Initialize DataFrame for cluster metrics
    cluster_metrics = pd.DataFrame({"Cluster": cluster_counts["Cluster"]})

    # Convert adata.X to DataFrame
    adata_df = pd.DataFrame(adata.X, columns=adata.var_names)

    # Add cluster annotation
    adata_df[annotation] = adata.obs[annotation].values

    # Calculate statistics for each feature in each cluster
    for feature in features:
        grouped = adata_df.groupby(annotation)[feature].agg(
            ["mean", "std", "median",
             lambda x: x.quantile(0),
             lambda x: x.quantile(0.995)
             ]).reset_index()
        grouped.columns = [
            f"{col}_{feature}" if col != annotation else "Cluster"
            for col in grouped.columns
        ]
        cluster_metrics = cluster_metrics.merge(
            grouped, on="Cluster", how="left"
            )

    # Merge cluster counts
    cluster_metrics = pd.merge(
        cluster_metrics, cluster_counts, on="Cluster", how="left"
    )

    return cluster_metrics


def tsne(adata, layer=None, **kwargs):
    """
    Perform t-SNE transformation on specific layer information.

    Parameters
    ----------
    adata : anndata.AnnData
       The AnnData object.
    layer : str
        Layer for phenograph cluster calculation.
    **kwargs
        Parameters for scanpy.tl.tsne function.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with t-SNE coordinates.
    """
    # If a layer is provided, it's transferred to 'obsm' for t-SNE computation
    # in scanpy.tl.tsne, which defaults to using the 'X' data matrix if not.

    if not isinstance(adata, anndata.AnnData):
        raise ValueError("adata must be an AnnData object.")

    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")

        tsne_obsm_name = layer + "_tsne"
        X_tsne = adata.to_df(layer=layer)
        adata.obsm[tsne_obsm_name] = X_tsne
    else:
        tsne_obsm_name = None

    sc.tl.tsne(adata, use_rep=tsne_obsm_name, random_state=7)

    return adata


def UMAP(
        adata,
        n_neighbors=15,
        n_pcs=30,
        min_dist=0.1,
        spread=1.0,
        n_components=2,
        random_state=42,
        layer=None):
    """
    Perform UMAP analysis on specific layer information.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_neighbors : int, default=15
        Number of neighbors for neighborhood graph.
    n_pcs : int, default=30
        Number of principal components.
    min_dist : float, default=0.1
        Minimum distance between points in UMAP.
    spread : float, default=1.0
        Spread of UMAP embedding.
    n_components : int, default=2
        Number of components in UMAP embedding.
    random_state : int, default=42
        Seed for random number generation.
    layer : str, optional
        Layer of the AnnData object to perform UMAP on.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with UMAP coordinates.
    """

    # Use utility function to check if the layer exists in adata.layers
    check_table(adata, tables=layer)

    if layer is not None:
        use_rep = layer + "_umap"
        X_umap = adata.layers[layer]
        adata.obsm[use_rep] = X_umap
    else:
        use_rep = 'X'

    # Compute the neighborhood graph
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        random_state=random_state
    )

    # Embed the neighborhood graph using UMAP
    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state
    )

    return adata


def batch_normalize(adata, annotation, layer, method="median", log=False):
    """
    Adjust the features of every marker using a normalization method.

    The normalization methods are summarized here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8723144/
    Adds the normalized values in
    `.layers[`layer`]`

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    annotation: str
        The name of the annotation in `adata` to define batches.

    layer : str
        The name of the new layer to add to the anndata object.

    method : {"median", "Q50", "Q75}
        The normlalization method to use.

    log : bool, default False
        If True, take the log2 of features before normalization.

    """
    allowed_methods = ["median", "Q50", "Q75"]
    regions = adata.obs[annotation].unique().tolist()
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
        region_cells = original[adata.obs[annotation] == region]

        if method == "median":
            region_median = region_cells.quantile(q=0.5)
            new_features = region_cells + \
                (all_regions_quantile - region_median)

        if method == "Q50":
            region_median = region_cells.quantile(q=0.5)
            new_features = (region_cells
                            * all_regions_quantile
                            / region_median)

        if method == "Q75":
            region_75quantile = region_cells.quantile(q=0.75)
            new_features = (region_cells
                            * all_regions_quantile
                            / region_75quantile)

        new_df_list.append(new_features)

    new_df = pd.concat(new_df_list)
    adata.layers[layer] = new_df


def rename_annotations(
        adata, src_annotation, dest_annotation, mappings,
        layer=None
):
    """
    Rename labels in a given annotation in an AnnData object based on a
    provided dictionary. This function modifies the adata object in-place
    and creates a new annotation column.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    src_annotation : str
        Name of the column in adata.obs containing the original
        labels of the source annotation.
    dest_annotation : str
        The name of the new column to be created in the AnnData object
        containing the renamed labels.
    mappings : dict
        A dictionary mapping the original annotation labels to
        the new labels.
    layer : str, optional
        The name of the layer in the AnnData object to check.

    Examples
    --------
    >>> adata = your_anndata_object
    >>> src_annotation = "phenograph"
    >>> mappings = {
    ...     "0": "group_8",
    ...     "1": "group_2",
    ...     "2": "group_6",
    ...     # ...
    ...     "37": "group_5",
    ... }
    >>> dest_annotation = "renamed_annotations"
    >>> adata = rename_annotations(
    ...     adata, src_annotation, dest_annotation, mappings)
    """

    # Use utility functions for input validation
    if layer:
        check_table(adata, tables=layer)
    check_annotation(adata, annotations=src_annotation)

    # Inform the user about the data type of the original column
    original_dtype = adata.obs[src_annotation].dtype
    print(f"The data type of the original column '{src_annotation}' is "
          f"{original_dtype}.")

    # Convert the keys in mappings to the same data type as the unique values
    unique_values = adata.obs[src_annotation].unique()
    mappings = {
        type(unique_values[0])(key): value for key, value in mappings.items()
    }

    # Identify and handle unmapped labels
    missing_mappings = [
        key for key in unique_values if key not in mappings.keys()
    ]

    if missing_mappings:
        warnings.warn(
            f"Missing mappings for the following labels: {missing_mappings}. "
            f"They will be set to NaN in the '{dest_annotation}' column."
        )
        for missing in missing_mappings:
            mappings[missing] = np.nan  # Assign NaN for missing mappings

    # Create a new column based on the mappings
    adata.obs[dest_annotation] = (
        adata.obs[src_annotation].map(mappings).astype("category")
    )


def normalize_features(
    adata: anndata,
    low_quantile: float = 0.02,
    high_quantile: float = 0.98,
    interpolation: str = "nearest",
    input_layer: str = None,
    new_layer_name: str = "normalized_feature",
    overwrite: bool = True
):

    """
    Normalize the features stored in an AnnData object.
    Any entry lower than the value corresponding to low_quantile of the column
    will be assigned a value of 0, and entry that are greater than
    high_quantile value will be assigned as 1. Other entries will be normalized
    with (values - quantile min)/(quantile max - quantile min).
    Resulting column will have value ranged between [0, 1].

    Parameters
    ----------
    adata : anndata.AnnData
        An AnnData object containing the data to be normalized.

    low_quantile : float, optional (default: 0.02)
        The lower quantile to use for normalization. Determines the
        minimum value after normalization.
        Must be a positive float between [0,1).

    high_quantile : float, optional (default: 0.02)
        The higher quantile to use for normalization. Determines the
        maximum value after normalization.
        Must be a positive float between (0,1].

    interpolation : str, optional (default: "nearest")
        The interpolation method to use when selecting the value for
        low and high quantile. Values can be "nearest" or "linear"

    input_layer : str, optional (default: None)
        The name of the layer in the AnnData object to be normalized.
        If None, the function will use the default data layer.

    new_layer_name : str, optional (default: "normalized_feature")
        The name of the new layer where the normalized features
        will be stored in the AnnData object.

    overwrite: bool, optional (default: True)
        If the new layer name exists in the anndata object,
        the function will defaultly overwrite the existing table unless
        'overwrite' is False

    Returns
    -------
    quantiles : pandas.DataFrame
        A DataFrame containing the quantile values calculated for every
        feature. The DataFrame has columns representing the features and rows
        representing the quantile values.
 """

    # Perform error checks for anndata object:
    check_table(adata, input_layer, should_exist=True)

    if not overwrite:
        check_table(adata, new_layer_name, should_exist=False)

    if not isinstance(high_quantile, (int, float)):
        raise TypeError("The high quantile should a numeric values, "
                        f"currently get {str(type(high_quantile))}")

    if not isinstance(low_quantile, (int, float)):
        raise TypeError("The low quantile should a numeric values, "
                        f"currently get {str(type(low_quantile))}")

    if low_quantile < high_quantile:
        if high_quantile <= 0 or high_quantile > 1:
            raise ValueError("The high quantile value should be within"
                             f"(0, 1], current value: {high_quantile}")
        if low_quantile < 0 or low_quantile >= 1:
            raise ValueError("The low quantile value should be within"
                             f"[0, 1), current value: {low_quantile}")
    else:
        raise ValueError("The low quantile shoud be smaller than"
                         "the high quantile, currently value is:\n"
                         f"low quantile: {low_quantile}\n"
                         f"high quantile: {high_quantile}")

    if interpolation not in ["nearest", "linear"]:
        raise ValueError("interpolation must be either 'nearest' or 'linear'"
                         f"passed value is:{interpolation}")

    dataframe = adata.to_df(layer=input_layer)

    # Calculate low and high quantiles
    quantiles = dataframe.quantile([low_quantile, high_quantile],
                                   interpolation=interpolation)

    for column in dataframe.columns:
        # low quantile value
        qmin = quantiles.loc[low_quantile, column]

        # high quantile value
        qmax = quantiles.loc[high_quantile, column]

        # Scale column values
        if qmax != 0:
            dataframe[column] = dataframe[column].apply(
                lambda x: 0 if x < qmin else (
                    1 if x > qmax else (x - qmin) / (qmax - qmin)
                )
            )

    # Append normalized feature to the anndata object
    adata.layers[new_layer_name] = dataframe

    return quantiles


def arcsinh_transformation(adata, input_layer=None, co_factor=None,
                           percentile=20, output_layer="arcsinh"):
    """
    Apply arcsinh transformation using a co-factor.

    The co-factor is determined either by the given percentile of each
    biomarker (feature-wise) or a provided fixed number. The function computes
    the co-factor for each biomarker individually, considering its unique
    range of expression levels. This ensures that each biomarker is scaled
    based on its inherent distribution, which is particularly important when
    dealing with datasets where features have a wide range of values.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to transform.
    input_layer : str, optional
        The name of the layer in the AnnData object to transform.
        If None, the main data matrix .X is used.
    co_factor : float, optional
        A fixed positive number to use as a co-factor for the transformation.
        If provided, it takes precedence over the percentile argument.
    percentile : int, default=20
        The percentile to determine the co-factor if co_factor is not provided.
        The percentile is computed for each feature (column) individually.
    output_layer : str, default="arcsinh"
        Name of the layer to put the transformed results. If it already exists,
        it will be overwritten with a warning.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object with the transformed data stored in the specified
        output_layer.
    """

    # Check if the provided input_layer exists in the AnnData object
    if input_layer:
        check_table(adata, tables=input_layer)
        data_to_transform = adata.layers[input_layer]
    else:
        data_to_transform = adata.X

    # Validate input parameters
    if co_factor and co_factor <= 0:
        raise ValueError("Co_factor should be a positive value.")

    if not (0 <= percentile <= 100):
        raise ValueError("Percentile should be between 0 and 100.")

    # Determine the co-factor
    if co_factor:
        factor = co_factor
    else:
        # Handle sparse matrix
        if issparse(data_to_transform):
            data_to_transform = data_to_transform.toarray()
        # Compute the percentiles per column (feature-wise)
        factor = np.percentile(data_to_transform, percentile, axis=0)
        # Check for zero values in factor and replace them to avoid division
        # by zero
        factor[factor == 0] = 1e-10

    # Apply the arcsinh transformation using the co-factor
    transformed_data = np.arcsinh(data_to_transform / factor)

    # Check if output_layer already exists and issue a warning if it does
    if output_layer in adata.layers:
        logging.warning(
            f"Layer '{output_layer}' already exists. It will be overwritten "
            "with the new transformed data."
        )

    # Store the transformed data in the specified output_layer
    adata.layers[output_layer] = transformed_data

    return adata


def z_score_normalization(adata, layer=None):
    """
    Compute z-scores for the provided AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the data to normalize.
    layer : str, optional
        The name of the layer in the AnnData object to normalize.
        If None, the main data matrix .X is used.

    Returns
    -------
    adata : anndata.AnnData
        The AnnData object with a new layer 'z_scores' containing the computed
        z-scores.
    """

    # Check if the provided layer exists in the AnnData object
    if layer:
        check_table(adata, tables=layer)
        data_to_normalize = adata.layers[layer]
    else:
        data_to_normalize = adata.X

    # Compute z-scores using scipy.stats.zscore
    z_scores = stats.zscore(data_to_normalize, axis=0, nan_policy='omit')

    # Store the computed z-scores in the 'z_scores' layer
    adata.layers['z_scores'] = z_scores

    return adata
