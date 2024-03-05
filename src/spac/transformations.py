import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import warnings
import logging
import scanpy.external as sce
from spac.utils import check_table, check_annotation, check_feature
from scipy import stats
import umap as umap_lib
from scipy.sparse import issparse, isspmatrix


def phenograph_clustering(adata, features, layer=None,
                          k=50, seed=None, **kwargs):
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
                                       seed=seed,
                                       **kwargs)

    adata.obs["phenograph"] = pd.Categorical(phenograph_out[0])
    adata.uns["phenograph_features"] = features


def get_cluster_info(
    adata, annotation="phenograph", features=None, layer=None
):
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
    layer : str, optional
        Specific layer from which to retrieve the features.
        If None, uses adata.X.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each cluster including the percentage of
        each cluster to the whole sample.
    """

    # Use utility functions for input validation
    check_annotation(adata, annotations=annotation)
    if features is None:
        features = list(adata.var_names)
    else:
        check_feature(adata, features=features)

    # Check if the layer is specified and validate it
    if layer:
        check_table(adata, tables=layer)
        data_matrix = adata.layers[layer]
    else:
        data_matrix = adata.X

    # Convert adata.X to DataFrame
    if isspmatrix(data_matrix):
        data_array = data_matrix.toarray()
    else:
        data_array = data_matrix
    data_df = pd.DataFrame(data_array, columns=adata.var_names)

    # Count cells in each cluster
    cluster_counts = adata.obs[annotation].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Number of Cells"]

    # Calculate the percentage of cells in each cluster
    total_cells = adata.obs.shape[0]
    cluster_counts['Percentage'] = (
        cluster_counts['Number of Cells'] / total_cells
    ) * 100

    # Initialize DataFrame for cluster metrics
    cluster_metrics = pd.DataFrame({"Cluster": cluster_counts["Cluster"]})

    # Add cluster annotation
    data_df[annotation] = adata.obs[annotation].values

    # Calculate statistics for each feature in each cluster
    for feature in features:
        grouped = data_df.groupby(annotation)[feature]\
                            .agg(["mean", "median"])\
                            .reset_index()
        grouped.columns = [
            f"{col}_{feature}" if col != annotation else "Cluster"
            for col in grouped.columns
        ]
        cluster_metrics = cluster_metrics.merge(
            grouped, on="Cluster", how="left"
        )

    # Merge cluster counts and percentage
    cluster_metrics = pd.merge(
        cluster_metrics, cluster_counts,
        on="Cluster", how="left"
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


def run_umap(
        adata,
        n_neighbors=75,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=0,
        transform_seed=42,
        layer=None,
        **kwargs
):
    """
    Perform UMAP analysis on the specific layer of the AnnData object
    or the default data.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    n_neighbors : int, default=75
        Number of neighbors to consider when constructing the UMAP. This
        influences the balance between preserving local and global structures
        in the data.
    min_dist : float, default=0.1
        Minimum distance between points in the UMAP space. Controls how
        tightly the embedding is allowed to compress points together.
    n_components : int, default=2
        Number of dimensions for embedding.
    metric : str, optional
        Metric to compute distances in high dimensional space.
        Check `https://umap-learn.readthedocs.io/en/latest/api.html` for
        options. The default is 'euclidean'.
    random_state : int, default=0
        Seed used by the random number generator(RNG) during UMAP fitting.
    transform_seed : int, default=42
        RNG seed during UMAP transformation.
    layer : str, optional
        Layer of AnnData object for UMAP. Defaults to `adata.X`.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object with UMAP coordinates stored in the `obsm`
        attribute. The key for the UMAP embedding in `obsm` is "X_umap".
    """

    # Use utility function to check if the layer exists in adata.layers
    if layer:
        check_table(adata, tables=layer)

    # Extract the data from the specified layer or the default data
    if layer is not None:
        data = adata.layers[layer]
    else:
        data = adata.X

    # Convert data to pandas DataFrame for better memory handling
    data = pd.DataFrame(data.astype(np.float32))

    # Create and configure the UMAP model
    umap_model = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        low_memory=True,
        random_state=random_state,
        transform_seed=transform_seed,
        **kwargs
    )

    # Fit and transform the data with the UMAP model
    embedding = umap_model.fit_transform(data)

    # Store the UMAP coordinates back into the AnnData object under the
    # 'X_umap' key, always
    adata.obsm['X_umap'] = embedding

    return adata


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
    batches = adata.obs[annotation].unique().tolist()
    original = adata.to_df()

    if log:
        original = np.log2(1+original)
        logging.info("Data transformed with log2")

    if method == "median" or method == "Q50":
        all_batch_quantile = original.quantile(q=0.5)
        logging.info("Median for al cells: %s", all_batch_quantile)
    elif method == "Q75":
        all_batch_quantile = original.quantile(q=0.75)
        logging.info("Q75 for all cells: %s", all_batch_quantile)
    else:
        raise Exception(
            "Unsupported normalization {0}, allowed methods = {1]",
            method, allowed_methods)

    # Place holder for normalized dataframes per batch
    for batch in batches:
        batch_cells = original[adata.obs[annotation] == batch]
        logging.info(f"Processing batch: {batch}, "
                     f"original values:\n{batch_cells}")

        if method == "median":
            batch_median = batch_cells.quantile(q=0.5)
            logging.info(f"Median for {batch}: %s", batch_median)
            original.loc[
                (adata.obs[annotation] == batch)
            ] = batch_cells + (all_batch_quantile - batch_median)

        elif method == "Q50":
            batch_50quantile = batch_cells.quantile(q=0.5)
            logging.info(f"Q50 for {batch}: %s", batch_50quantile)
            original.loc[adata.obs[annotation] == batch] = (
                batch_cells * all_batch_quantile / batch_50quantile
            )

        elif method == "Q75":
            batch_75quantile = batch_cells.quantile(q=0.75)
            logging.info(f"Q75 for {batch}: %s", batch_75quantile)
            original.loc[adata.obs[annotation] == batch] = (
                batch_cells * all_batch_quantile / batch_75quantile
            )

    adata.layers[layer] = original


def rename_annotations(adata, src_annotation, dest_annotation, mappings):
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
