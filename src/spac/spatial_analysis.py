import squidpy as sq
import matplotlib.pyplot as plt
import pandas as pd
import anndata
import pickle
import io
from spac.utils import check_annotation


def spatial_interaction(
        adata,
        annotation,
        analysis_method,
        stratify_by=None,
        ax=None,
        **kwargs):
    """
    Perform spatial analysis on the selected annotation in the dataset.
    Current analysis methods are provided in squidpy:
        Neighborhood Enrichment,
        Cluster Interaction Matrix
    Parameters:
    -----------
        adata : anndata.AnnData
            The AnnData object.

        annotation : str
            The column name of the annotation to analysis in the dataset.

        analysis_method : str
            The analysis method to use, currently available:
            "Neighborhood Enrichment" and "Cluster Interaction Matrix".

        stratify_by : str or list of strs
            The annotation[s] to stratify the dataset. If single annotation is
            passed, the dataset will stratify by the unique values in
            the annotation column. If n (n>=2) annotations are passed,
            the function will stratify the dataset basing on existing
            label combinations.

        ax: matplotlib.axes.Axes, default None
            The matplotlib Axes to display the image.

        **kwargs
            Keyword arguments for matplotlib.pyplot.text()
    Returns:
    -------
        ax_dictionary : dictionary of matplotlib.axes.Axes
            A dictionary of the matplotlib Axes containing the analysis plots.
            If not stratify, the key for analysis will be "Full",
            otherwise the plot will be stored with key <stratify combination>.
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
                new_annotation_name,
                ax):

        # Calculate Neighborhood_Enrichment
        sq.gr.nhood_enrichment(
                    adata,
                    cluster_key=new_annotation_name)

        # Plot Neighborhood_Enrichment
        sq.pl.nhood_enrichment(
                    adata,
                    cluster_key=new_annotation_name,
                    ax=ax,
                    **kwargs)

        return ax

    def Cluster_Interaction_Matrix_Analysis(
                adata,
                new_annotation_name,
                ax):

        # Calculate Cluster_Interaction_Matrix
        sq.gr.interaction_matrix(
                    adata,
                    cluster_key=new_annotation_name)

        # Plot Cluster_Interaction_Matrix
        sq.pl.interaction_matrix(
                    adata,
                    cluster_key=new_annotation_name,
                    ax=ax,
                    **kwargs)

        return ax

    # Error Check Section
    # -----------------------------------------------
    if not isinstance(adata, anndata.AnnData):
        error_text = "Input data is not an AnnData object. " + \
            f"Got {str(type(adata))}"
        raise ValueError(error_text)

    # Check if annotation is in the dataset
    check_annotation(
        adata,
        [annotation]
    )

    # Check if stratify_by is list or list of str
    if stratify_by:
        if not isinstance(stratify_by, str):
            if isinstance(stratify_by, list):
                for item in stratify_by:
                    if not isinstance(item, str):
                        error_text = "Item in the stratify_by " + \
                            "list should be " + \
                            f"strings, getting {type(item)} for {item}."
                        raise ValueError(error_text)

                    check_annotation(
                        adata,
                        item
                    )

            else:
                error_text = "The stratify_by variable should be " + \
                    "single string or a list of string, currently is" + \
                    f"{type(stratify_by)}"
                raise ValueError(error_text)
        else:
            check_annotation(
                adata,
                stratify_by
            )

    if not isinstance(analysis_method, str):
        error_text = "The analysis methods must be a string."
        raise ValueError(error_text)
    else:
        if analysis_method not in available_methods:
            error_text = f"Method {analysis_method}" + \
                " is not supported currently. " + \
                f"Available methods are: {available_methods_str}"
            raise ValueError(error_text)

    if ax is not None:
        if not isinstance(ax, plt.Axes):
            error_text = "Invalid 'ax' argument. " + \
                "Expected an instance of matplotlib.axes.Axes. " + \
                f"Got {str(type(ax))}"
            raise ValueError(error_text)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Operational Section
    # -----------------------------------------------

    # Create a categorical column data for plotting
    new_annotation_name = annotation + "_plot"

    adata.obs[new_annotation_name] = pd.Categorical(
        adata.obs[annotation])

    if stratify_by:
        if isinstance(stratify_by, list):
            adata.obs[
                'concatenated_obs'
            ] = adata.obs[stratify_by].astype(str).agg('_'.join, axis=1)
        else:
            adata.obs[
                'concatenated_obs'
            ] = adata.obs[stratify_by]

    # Compute a connectivity matrix from spatial coordinates
    if stratify_by:
        ax_dictionary = {}
        unique_values = adata.obs['concatenated_obs'].unique()
        buffer = io.BytesIO()
        pickle.dump(ax, buffer)
        for subset_key in unique_values:
            # Subset the original AnnData object based on the unique value
            subset_adata = adata[
                adata.obs[
                    'concatenated_obs'
                ] == subset_key
            ].copy()

            sq.gr.spatial_neighbors(subset_adata)

            buffer.seek(0)

            ax_copy = pickle.load(buffer)

            if analysis_method == "Neighborhood Enrichment":
                ax_copy = Neighborhood_Enrichment_Analysis(
                        subset_adata,
                        new_annotation_name,
                        ax_copy)

            elif analysis_method == "Cluster Interaction Matrix":
                ax_copy = Cluster_Interaction_Matrix_Analysis(
                        subset_adata,
                        new_annotation_name,
                        ax_copy)

            ax_dictionary[subset_key] = ax_copy

            del subset_adata

    else:
        sq.gr.spatial_neighbors(adata)

        if analysis_method == "Neighborhood Enrichment":
            ax = Neighborhood_Enrichment_Analysis(
                    adata,
                    new_annotation_name,
                    ax)

        elif analysis_method == "Cluster Interaction Matrix":
            ax = Cluster_Interaction_Matrix_Analysis(
                    adata,
                    new_annotation_name,
                    ax)
        ax_dictionary = {"Full": ax}

    return ax_dictionary
