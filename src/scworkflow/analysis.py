
import anndata as ad
import scanpy.external as sce
import pandas as pd
from scworkflow.normalization import subtract_min_quantile
import re

def ingest_cells(dataframe, regex_str, x_col=None, y_col=None, region=None):
    """
    Read the csv file into an anndata object. Intializes intensities and spatial coordiantes
    
    @param dataframe <pandas.DataFrame>
        The data frame that contains cells as rows, and cells informations as columns
        
    @param regex_str <str>
        A python regular expression for the intensities columns in the data frame
        
    @param x_col <str>
        The column name for the x coordinate of the cell
        
    @param y_col <str>
        The column name for the y coordinate of the cell
        
    @param region <str>
        The column name for the region that the cells
        
    @return adata <AnnData>
        The generated AnnData object
    """
    
    intensities_regex = re.compile(regex_str)
    all_intensities = list(filter(intensities_regex.match, list(dataframe.columns))) 
   
    intensities_df = dataframe[all_intensities]
    adata = ad.AnnData(intensities_df, dtype=intensities_df[all_intensities[0]].dtype)
   
    if region != None:
        #As selecting one column of the dataframe returns a series which AnnData 
        #converts to NaN, then I convert it to list before assignment
        adata.obs["region"] = dataframe[region].tolist()
    
   
    if x_col != None and y_col != None:
        adata.obsm["spatial"] = dataframe[[x_col, y_col]].to_numpy().astype('float32')
    
    return adata

def concatinate_regions(regions):
    """
    Concatinate data from multiple regions and create new indexes 

    @param regions list<anndata>
        list of multiple anndata objects

    returns:
        anndata with that includes all regions
    """
   
    all_adata = ad.concat(regions)
    all_adata.obs_names_make_unique()
    return all_adata

def rescale_intensities(intensities, min_quantile=0.01, max_quantile=0.99):
    """
    Clip intensities outside the minimum and maximum quantile and rescale them between 0 and 1
    
    @param intensities dataframe
        The dataframe of intensities
        
    
    min_quantile: float
        The minimum quantile to be consider zero
    
    max_qunatile: float
        The maximum quantile to be considerd 1
        
    returns: 
        dataframe with normalized intensities
    """
    
    markers_max_quantile = intensities.quantile(max_quantile)
    markers_min_quantile = intensities.quantile(min_quantile)
    
    intensities_clipped = intensities.clip(markers_min_quantile, markers_max_quantile, axis=1)
    
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    np_intensities_scaled = scaler.fit_transform(intensities_clipped.to_numpy())
    intensities_scaled = pd.DataFrame(np_intensities_scaled, columns=intensities_clipped.columns)
    return intensities_scaled

def add_rescaled_intensity(adata, min_quantile, max_quantile, layer_name):
    """
    Clips and rescales the intensities matrix and add the results into a new layer in the AnnData object
    @param adata <AnnData>
         The AnnData object
    
    @param min_quantile <float>
        The minimum quantile to rescale to zero
        
    @param max_quantime <float>
        The maximum quantile to rescale to one
        
    @param layer_name <str>
        The name of the new layer to add to the anndata object
    """
    
    original = adata.to_df()
    rescaled = rescale_intensities(original, min_quantile, max_quantile)
    adata.layers[layer_name] = rescaled 


def pheongraph_clustering(adata, features, layer, k=30):
    """
    Calculates automatic phenotypes using phenograph
    @param adata <AnnData>
       The AnnData object
    
    @param features <list[str]>
        The variables that would be included in creating the phenograph clusters
        
    @param layer <str>
        The layer to be used in calculating the phengraph clusters
    
    @param k <int>
        The number of nearest neighbor to be used in creating the graph.
    """
    phenograph_df = adata.to_df(layer=layer)[features]
    phenograph_out = sce.tl.phenograph(phenograph_df, clustering_algo="louvain", k=k)
    adata.obs["phenograph"] = phenograph_out[0]
    adata.uns["phenograph_features"] = features


def subtract_min_per_region(adata, layer_name, min_quantile=0.01):
    """
    Substract the minimum quantile of every marker per region

    @param adata <AnnData>
         The AnnData object
    
    @param min_quantile <float>
        The minimum quantile to rescale to zero
        
    @param layer_name <str>
        The name of the new layer to add to the anndata object
    """

    regions = adata.obs['region'].unique().tolist()
    original = adata.to_df()

    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs['region'] == region] 
        new_intensities = subtract_min_quantile(region_cells, min_quantile)
        new_df_list.append(new_intensities)



    new_df = pd.concat(new_df_list)
    adata.layers[layer_name] = new_df 
    #print(adata.to_df(layer=layer_name))

def normalize(adata, layer_name, method="median"):
    """
    Adjust the intensity of every marker using some normalization method defined here:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8723144/

    @param adata <AnnData>
         The AnnData object
    
    @param layer_name <str>
        The name of the new layer to add to the anndata object

    @param method <str> 
        The normlalization method to use: "median", "Q50", "Q75" 
    """

    allowed_methods = ["median", "Q50", "Q75"]
    regions = adata.obs['region'].unique().tolist()
    original = adata.to_df()

    if method == "median" or method == "Q50" :
        all_regions_quantile = original.quantile(q=0.5)
    elif method == "Q75":
        all_regions_quantile = original.quantile(q=0.75)
    else: 
        raise Exception("Unsupported normalization {0}, allowed methods = {1]",
            method, allowed_methods)
    
    #Place holder for normalized dataframes per region
    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs['region'] == region] 
        
        if method == "median":
            region_median = region_cells.quantile(q=0.5)
            new_intensities = region_cells + \
                (all_regions_quantile - region_median)

        if method == "Q50":
            region_median = region_cells.quantile(q=0.5)
            new_intensities = region_cells  * all_regions_quantile / region_median

        if method == "Q75":
            region_75quantile = region_cells.quantile(q=0.75)
            new_intensities = region_cells  * all_regions_quantile / region_75quantile

        new_df_list.append(new_intensities)

    new_df = pd.concat(new_df_list)
    adata.layers[layer_name] = new_df 
    #print(adata.to_df(layer=layer_name))
