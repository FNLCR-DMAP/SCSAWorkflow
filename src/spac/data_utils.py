import re
import seaborn
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import scanpy.external as sce
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



def ingest_cells(dataframe, regex_str, x_col=None, y_col=None, obs=None):

    """
    Read the csv file into an anndata object.

    The function will also intialize intensities and spatial coordiantes.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data frame that contains cells as rows, and cells informations as
        columns. 

    regex_str : str or list of str
        A string or a list of strings representing python regular expression
        for the intensities columns in the data frame.  x_col : str The column
        name for the x coordinate of the cell.

    y_col : str
        The column name for the y coordinate of the cell.

    obs : str or list of str
        The column name for the re gion that the cells. If a list is passed,
        multiple observations will be created in the returned AnnData object.


    Returns
    -------
    anndata.AnnData
        The generated AnnData object
    """

    if not isinstance(regex_str, list):
        regex_list = [regex_str]
    else:
        regex_list = regex_str

    all_intensities = []
    all_columns = list(dataframe.columns)
    for regex in regex_list:
        intensities_regex = re.compile(regex)
        intensities = list(
            filter(intensities_regex.match, all_columns))
        all_intensities.extend(intensities)

    intensities_df = dataframe[all_intensities]
    adata = ad.AnnData(
        intensities_df,
        dtype=intensities_df[all_intensities[0]].dtype)

    if obs is not None:
        if isinstance(obs, str):
            list_of_obs = [obs]
        else:
            list_of_obs = obs 

        for observation in list_of_obs:

            # As selecting one column of the dataframe returns a series which
            # AnnData converts to NaN, then I convert it to a list before
            # assignment.
            adata.obs[observation] = dataframe[observation].tolist()

    if x_col is not None and y_col is not None:
        numpy_array = dataframe[[x_col, y_col]].to_numpy().astype('float32')
        adata.obsm["spatial"] = numpy_array
    return adata


def concatinate_regions(regions):
    """
    Concatinate data from multiple regions and create new indexes.

    Parameters
    ----------
    regions : list of anndata.AnnData
        AnnData objects to be concatinated.

    Returns
    -------
    anndata.AnnData
        New AnddData object with the concatinated values in AnnData.X

    """
    all_adata = ad.concat(regions)
    all_adata.obs_names_make_unique()
    return all_adata


def rescale_intensities(intensities, min_quantile=0.01, max_quantile=0.99):
    """
    Clip and rescale intensities outside the minimum and maximum quantile.

    The rescaled intensities will be between 0 and 1.

    Parameters
    ----------
    intensities : pandas.Dataframe
        The DataRrame of intensities.

    min_quantile : float
        The minimum quantile to be consider zero.

    max_quantile: float
        The maximum quantile to be considerd 1.

    Returns
    -------
    pandas.DataFrame
        The created DataFrame with normalized intensities.
    """
    markers_max_quantile = intensities.quantile(max_quantile)
    markers_min_quantile = intensities.quantile(min_quantile)

    intensities_clipped = intensities.clip(
        markers_min_quantile,
        markers_max_quantile,
        axis=1)

    scaler = MinMaxScaler()
    np_intensities_scaled = scaler.fit_transform(
        intensities_clipped.to_numpy())

    intensities_scaled = pd.DataFrame(
        np_intensities_scaled,
        columns=intensities_clipped.columns)

    return intensities_scaled


def add_rescaled_intensity(adata, min_quantile, max_quantile, layer):
    """
    Clip and rescale the intensities matrix.

    The results will be added into a new layer in the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    min_quantile : float
        The minimum quantile to rescale to zero.

    max_quantile : float
        The maximum quantile to rescale to one.

    layer : str
        The name of the new layer to add to the anndata object.
    """

    original = adata.to_df()
    rescaled = rescale_intensities(original, min_quantile, max_quantile)
    adata.layers[layer] = rescaled

def subtract_min_per_region(adata, obs, layer, min_quantile=0.01):
    """
    Substract the minimum quantile of every marker per region.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    obs: str
        The name of the observation in `adata` to define batches.

    min_quantile : float
        The minimum quantile to rescale to zero.

    layer : str
        The name of the new layer to add to the AnnData object.
    """
    regions = adata.obs[obs].unique().tolist()
    original = adata.to_df()

    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs[obs] == region]
        new_intensities = subtract_min_quantile(region_cells, min_quantile)
        new_df_list.append(new_intensities)

    new_df = pd.concat(new_df_list)
    adata.layers[layer] = new_df

def subtract_min_quantile(intensities, min_quantile=.01):
    """
    Subtract the intensity defined by the minimum quantile from all columns.

    Parameters
    ----------

    intensities : pandas.DataFrame
        The dataframe of intensities.

    min_quantile: float
        The minimum quantile to be consider zero.

    Returns
    -------
    pandas.DataFrame
        dataframe with rescaled intensities.
    """
    columns_min_quantile = intensities.quantile(min_quantile)

    subtracted_min = intensities - columns_min_quantile

    # Clip negative values to zero
    subtracted_min.clip(lower=0, axis=1, inplace=True)

    return subtracted_min

