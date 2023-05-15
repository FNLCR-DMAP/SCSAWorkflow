import numpy as np
import scanpy as sc
import pandas as pd
import scanpy.external as sce


def phenograph_clustering(adata, features, layer, k=30):
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

    adata.obs["phenograph"] = pd.Categorical(phenograph_out[0])
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


def batch_normalize(adata, obs, layer, method="median", log=False):
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

    obs: str
        The name of the observation in `adata` to define batches.

    layer : str
        The name of the new layer to add to the anndata object.

    method : {"median", "Q50", "Q75}
        The normlalization method to use.

    log : bool, default False
        If True, take the log2 of intensities before normalization.

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


def rename_clustering(adata, column,
                      new_phenotypes,
                      new_column_name="renamed_clusters"):
    """
    Rename and group clusters in an AnnData object based on the
    provideddictionary, keeping the original observation column.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    column : str
        Name of the column in adata.obs containing
        the original cluster labels.
    new_phenotypes : dict
        A dictionary mapping the original cluster names
        to the new phenotype names.
    new_column_name : str, optional, default: "renamed_clusters"
        The name of the new column to be created in the AnnData object
        containing the renamed cluster labels.

    Returns
    -------
    adata : anndata.AnnData
        The updated Anndata object with the new column containing the renamed
        cluster labels.
    """

    """
    # An example to call the function:
    adata = your_anndata_object
    column = "phenograph"
    new_phenotypes = {
        "0": "group_8",
        "1": "group_2",
        "2": "group_6",
        # ...
        "37": "group_5",
    }
    new_column_name = "renamed_clusters"

    adata = rename_clustering(adata, column, new_phenotypes, new_column_name)
    """

    # Get the unique values of the observation column
    unique_values = adata.obs[column].unique()

    # Convert the keys in new_phenotypes to the same data type as the unique
    # values in the observation column
    new_phenotypes = {
        type(unique_values[0])(key): value
        for key, value in new_phenotypes.items()
    }

    # Check if all keys in new_phenotypes are present in the unique values of
    # the observation column
    if not all(key in unique_values for key in new_phenotypes.keys()):
        error_message = "All keys in the new_phenotypes dictionary " + \
                        "should match the unique values in " + \
                        "the observation column."

        raise ValueError(error_message)

    # Create a new column in adata.obs with the updated cluster names
    adata.obs[new_column_name] = adata.obs[column].map(new_phenotypes)\
        .fillna(adata.obs[column]).astype("category")

    return adata
