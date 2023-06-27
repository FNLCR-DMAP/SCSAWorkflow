import numpy as np
import scanpy as sc
import pandas as pd
import anndata
import scanpy.external as sce


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


def batch_normalize(adata, obs, layer, method="median", log=False):
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

    obs: str
        The name of the observation in `adata` to define batches.

    layer : str
        The name of the new layer to add to the anndata object.

    method : {"median", "Q50", "Q75}
        The normlalization method to use.

    log : bool, default False
        If True, take the log2 of features before normalization.

    """
    allowed_methods = ["median", "Q50", "Q75"]
    regions = adata.obs[obs].unique().tolist()
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
        region_cells = original[adata.obs[obs] == region]

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


def rename_observations(adata, src_observation, dest_observation, mappings):
    """
    Rename observations in an AnnData object based on a provided dictionary.
    This function creates a new observation column.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    src_observation : str
        Name of the column in adata.obs containing the original
        observation labels.
    dest_observation : str
        The name of the new column to be created in the AnnData object
        containing the renamed observation labels.
    mappings : dict
        A dictionary mapping the original observation labels to
        the new labels.

    Returns
    -------
    adata : anndata.AnnData
        The updated Anndata object with the new column containing the
        renamed observation labels.

    Examples
    --------
    >>> adata = your_anndata_object
    >>> src_observation = "phenograph"
    >>> mappings = {
    ...     "0": "group_8",
    ...     "1": "group_2",
    ...     "2": "group_6",
    ...     # ...
    ...     "37": "group_5",
    ... }
    >>> dest_observation = "renamed_observations"
    >>> adata = rename_observations(
    ...     adata, src_observation, dest_observation, mappings)
    """

    # Check if the source observation exists in the AnnData object
    if src_observation not in adata.obs.columns:
        raise ValueError(
            f"Source observation '{src_observation}' not found in the "
            "AnnData object."
        )

    # Get the unique values of the source observation
    unique_values = adata.obs[src_observation].unique()

    # Convert the keys in mappings to the same data type as the unique values
    mappings = {
        type(unique_values[0])(key): value
        for key, value in mappings.items()
    }

    # Check if all keys in mappings match the unique values in the
    # source observation
    if not all(key in unique_values for key in mappings.keys()):
        raise ValueError(
            "All keys in the mappings dictionary should match the unique "
            "values in the source observation."
        )

    # Check if the destination observation already exists in the AnnData object
    if dest_observation in adata.obs.columns:
        raise ValueError(
            f"Destination observation '{dest_observation}' already exists "
            "in the AnnData object."
        )

    # Create a new column in adata.obs with the updated observation labels
    adata.obs[dest_observation] = (
        adata.obs[src_observation]
        .map(mappings)
        .fillna(adata.obs[src_observation])
        .astype("category")
    )

    return adata
