import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import scanpy.external as sce
from spac.utils import check_table


def phenograph_clustering(adata, features, layer=None, k=30):
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

    if not isinstance(adata, sc.AnnData):
        raise TypeError("`adata` must be of type anndata.AnnData")

    if (not isinstance(features, list) or
            not all(isinstance(feature, str) for feature in features)):
        raise TypeError("`features` must be a list of strings")

    if layer is not None and layer not in adata.layers.keys():
        raise ValueError(f"`{layer}` not found in `adata.layers`. "
                         f"Available layers are {list(adata.layers.keys())}")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("`k` must be a positive integer")

    if not all(feature in adata.var_names for feature in features):
        raise ValueError("One or more of the `features` are not in "
                         "`adata.var_names`")

    if layer is not None:
        phenograph_df = adata.to_df(layer=layer)[features]
    else:
        phenograph_df = adata.to_df()[features]

    phenograph_out = sce.tl.phenograph(phenograph_df,
                                       clustering_algo="louvain",
                                       k=k)

    adata.obs["phenograph"] = pd.Categorical(phenograph_out[0])
    adata.uns["phenograph_features"] = features


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


def rename_labels(adata, src_annotation, dest_annotation, mappings):
    """
    Rename labels in a given annotation in an AnnData object based on a 
    provided dictionary. This function creates a new annotation column.

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

    Returns
    -------
    adata : anndata.AnnData
        The updated Anndata object with the new column containing the
        renamed labels.

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

    # Check if the source annotation exists in the AnnData object
    if src_annotation not in adata.obs.columns:
        raise ValueError(
            f"Source annotation '{src_annotation}' not found in the "
            "AnnData object."
        )

    # Get the unique values of the source annotation
    unique_values = adata.obs[src_annotation].unique()

    # Convert the keys in mappings to the same data type as the unique values
    mappings = {
        type(unique_values[0])(key): value
        for key, value in mappings.items()
    }

    # Check if all keys in mappings match the unique values in the
    # source annotation
    if not all(key in unique_values for key in mappings.keys()):
        raise ValueError(
            "All keys in the mappings dictionary should match the unique "
            "values in the source annotation."
        )

    # Check if the destination annotation already exists in the AnnData object
    if dest_annotation in adata.obs.columns:
        raise ValueError(
            f"Destination annotation '{dest_annotation}' already exists "
            "in the AnnData object."
        )

    # Create a new column in adata.obs with the updated annotation labels
    adata.obs[dest_annotation] = (
        adata.obs[src_annotation]
        .map(mappings)
        .astype("category")
    )

    # Ensure that all categories are covered
    if adata.obs[dest_annotation].isna().any():
        raise ValueError(
            "Not all unique values in the source annotation are "
            "covered by the mappings. "
            "Please ensure that the mappings cover all unique values."
        )

    return adata


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
