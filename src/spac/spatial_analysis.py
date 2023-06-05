import squidpy as sq
import matplotlib.pyplot as plt
import pandas as pd
import anndata


def spatial_interaction(
        adata,
        observation,
        analysis_method,
        ax=None,
        **kwargs):
    """
    Perform spatial analysis on the selected observation in the dataset.
    Current analysis methods are provided in squidpy:
        Neighborhood Enrichment,
        Cluster Interaction Matrix
    Parameters:
    -----------
        adata : anndata.AnnData
            The AnnData object.

        observation : str
            The column name of the observation to analysis in the dataset.

        analysis_method : str
            The analysis method to use, currently available:
            "Neighborhood Enrichment" and "Cluster Interaction Matrix".

        ax: matplotlib.axes.Axes, default None
            The matplotlib Axes to display the image.

        **kwargs
            Keyword arguments for matplotlib.pyplot.text()
    Returns:
    -------
        ax : matplotlib.axes.Axes
            The matplotlib Axes containing the analysis plots.
            The returned ax is the passed ax or new ax created.
    """

    # List all available methods
    available_methods = [
        "Neighborhood Enrichment",
        "Cluster Interaction Matrix"
    ]
    available_methods_str = ",".join(available_methods)

    # pacakge each methods into a function to allow
    # centralized control and improve flexibility
    def Neighborhood_Enrichment_Analysis(
                adata,
                new_observation_name,
                ax):

        # Calculate Neighborhood_Enrichment
        sq.gr.nhood_enrichment(
                    adata,
                    cluster_key=new_observation_name)

        # Plot Neighborhood_Enrichment
        sq.pl.nhood_enrichment(
                    adata,
                    cluster_key=new_observation_name,
                    ax=ax,
                    **kwargs)

        return ax

    def Cluster_Interaction_Matrix_Analysis(
                adata,
                new_observation_name,
                ax):

        # Calculate Cluster_Interaction_Matrix
        sq.gr.interaction_matrix(
                    adata,
                    cluster_key=new_observation_name)

        # Plot Cluster_Interaction_Matrix
        sq.pl.interaction_matrix(
                    adata,
                    cluster_key=new_observation_name,
                    ax=ax,
                    **kwargs)

        return ax

    if not isinstance(adata, anndata.AnnData):
        error_text = "Input data is not an Anndata object." + \
            f"Got {str(type(ax))}"
        raise ValueError(error_text)

    # Extract column name
    column_names = adata.obs.columns.tolist()
    column_names_str = ",".join(column_names)

    if observation not in column_names:
        error_text = f"Feature {observation} not found in the dataset." + \
            f"existing observations are: {column_names_str}"
        raise ValueError(error_text)

    if not isinstance(analysis_method, str):
        error_text = "The analysis methods must be a string."
        raise ValueError(error_text)
    else:
        if analysis_method not in available_methods:
            error_text = f"Method {analysis_method}" + \
                " is not supported currently." + \
                f"Available methods are: {available_methods_str}"
            raise ValueError(error_text)

    if ax is not None:
        if not isinstance(ax, plt.Axes):
            error_text = "Invalid 'ax' argument. " + \
                "Expected an instance of matplotlib.axes.Axes." + \
                f"Got {str(type(ax))}"
            raise ValueError(error_text)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Create a categorical column data for plotting
    new_observation_name = observation + "_plot"

    adata.obs[new_observation_name] = pd.Categorical(
        adata.obs[observation])

    # Compute a connectivity matrix from spatial coordinates
    sq.gr.spatial_neighbors(adata)

    if analysis_method == "Neighborhood Enrichment":
        ax = Neighborhood_Enrichment_Analysis(
                adata,
                new_observation_name,
                ax)

    elif analysis_method == "Cluster Interaction Matrix":
        ax = Cluster_Interaction_Matrix_Analysis(
                adata,
                new_observation_name,
                ax)

    return ax
