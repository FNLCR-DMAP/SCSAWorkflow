
import pandas as pd
import anndata
import scanpy as sc
import matplotlib.pyplot as plt

def hierarchical_heatmap(adata, column, layer=None, dendrogram=True, standard_scale=None, **kwargs):
    """
    Plot a hierarchical clustering heatmap of the mean intensity of cells that belong to a `column' using scanpy.tl.dendrogram and sc.pl.matrixplot.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.
    column : str
        Name of the column in adata.obs to group by and calculate mean intensity.
    layer : str, optional, default: None
        The name of the `adata` layer to use to calculate the mean intensity.
    dendrogram : bool, optional, default: True
        If True, a dendrogram based on the hierarchical clustering between the `column` categories is computed and plotted.
    **kwargs:
        Additional parameters passed to sc.pl.matrixplot function.

    Returns
    ----------
    mean_intensity, matrixplot, ax_dendrogram
    
    """


    # Calculate mean intensity
    intensities = adata.to_df(layer=layer)
    labels = adata.obs[column]
    grouped = pd.concat([intensities, labels], axis=1).groupby(column)
    mean_intensity = grouped.mean()

    # Reset the index of mean_intensity
    mean_intensity = mean_intensity.reset_index()

    # Convert mean_intensity to AnnData
    mean_intensity_adata = sc.AnnData(X=mean_intensity.iloc[:, 1:].values, 
                                      obs=pd.DataFrame(index=mean_intensity.index,
                                      data={column: mean_intensity.iloc[:, 0].astype('category').values}), 
                                      var=pd.DataFrame(index=mean_intensity.columns[1:]))

    # Compute dendrogram if needed
    if dendrogram:
        sc.tl.dendrogram(mean_intensity_adata,
                         groupby=column,
                         var_names=mean_intensity_adata.var_names,
                         n_pcs=None)

    
    # Create the matrix plot
    matrixplot = sc.pl.matrixplot(mean_intensity_adata,
                                  var_names=mean_intensity_adata.var_names,
                                  groupby=column, use_raw=False, dendrogram=dendrogram,
                                  standard_scale=standard_scale, cmap="viridis", return_fig=True, **kwargs)



    # Create the dendrogram plot
    fig, ax_dendrogram = plt.subplots()
    sc.pl.dendrogram(mean_intensity_adata, groupby=column, ax=ax_dendrogram, show=None)

    return mean_intensity, matrixplot, ax_dendrogram


"""
# An example to call this function:
mean_intensity, matrixplot, ax_dendrogram = hierarchical_heatmap(all_data, "phenograph", layer=None, standard_scale='var')

# Display the matrix plot
matrixplot.show()

# Display the dendrogram
ax_dendrogram.figure.show()
"""

