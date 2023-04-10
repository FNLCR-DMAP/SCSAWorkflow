import anndata

def rename_clustering(adata, column, new_phenotypes, new_column_name="renamed_clusters"):
    """
    Rename and group clusters in an AnnData object based on the provided
    dictionary, keeping the original observation column.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    column : str
        Name of the column in adata.obs containing the original cluster labels.
    new_phenotypes : dict
        A dictionary mapping the original cluster names to the new phenotype names.
    new_column_name : str, optional, default: "renamed_clusters"
        The name of the new column to be created in the AnnData object
        containing the renamed cluster labels.

    Returns
    -------
    adata : anndata.AnnData
        The updated Anndata object with the new column containing the renamed
        cluster labels.
    """
    
    # Get the unique values of the observation column
    unique_values = adata.obs[column].unique()

    # Convert the keys in new_phenotypes to the same data type as the unique
    # values in the observation column
    new_phenotypes = {
        type(unique_values[0])(key): value for key, value in new_phenotypes.items()
    }

    # Check if all keys in new_phenotypes are present in the unique values of
    # the observation column
    if not all(key in unique_values for key in new_phenotypes.keys()):
        raise ValueError(
            "All keys in the new_phenotypes dictionary should match the unique "
            "values in the observation column."
        )
    
    # Create a new column in adata.obs with the updated cluster names
    adata.obs[new_column_name] = adata.obs[column].map(new_phenotypes)\
        .fillna(adata.obs[column]).astype("category")

    return adata



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

