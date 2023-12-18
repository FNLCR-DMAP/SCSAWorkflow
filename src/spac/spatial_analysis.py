import squidpy as sq
import matplotlib.pyplot as plt
import pandas as pd
import anndata
from spac.utils import check_annotation


def spatial_interaction(
        adata,
        annotation,
        analysis_method,
        stratify_by=None,
        ax=None,
        return_matrix=False,
        seed=None,
        coord_type=None,
        n_rings=1,
        n_neighs=6,
        radius=None,
        **kwargs):
    """
    Perform spatial analysis on the selected annotation in the dataset.
    Current analysis methods are provided in squidpy:
        Neighborhood Enrichment,
        Cluster Interaction Matrix

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object.

    annotation : str
        The column name of the annotation to analysis in the dataset.

    analysis_method : str
        The analysis method to use, currently available:
        "Neighborhood Enrichment" and "Cluster Interaction Matrix".

    stratify_by : str or list of strs
        The annotation[s] to stratify the dataset when generating
        interaction plots. If single annotation is passed, the dataset
        will be stratified by the unique labels in the annotation column.
        If n (n>=2) annotations are passed, the function will be stratified
        based on existing combination of labels in the passed annotations.

    ax: matplotlib.axes.Axes, default None
        The matplotlib Axes to display the image. This option is only
        available when stratify is None.

    return_matrix: boolean, default False
        If true, the fucntion will return a list of two dictionaries,
        the first contains axes and the second containing computed matrix.
        Note that for Neighborhood Encrichment, the matrix will be a tuple
        with the z-score and the enrichment count.
        For Cluster Interaction Matrix, it will returns the
        interaction matrix.
        If False, the function will return only the axes dictaionary.

    seed: int, default None
        Random seed for reproducibility, used in Neighborhood Enrichment
        Analysis.

    coord_type : str, optional
        Type of coordinate system used in sq.gr.spatial_neighbors.
        Should be either 'grid' (Visium Data) or 'generic' (Others).
        Default is None, decided by the squidy pacakge. If spatial_key
        is in anndata.uns the coord_type would be 'grid', otherwise
        general.

    n_rings : int, default 1
        Number of rings of neighbors for grid data. 
        Only used when coord_type = 'grid' (Visium)
    
    n_neights : int, optional
        Default is 6.
        Depending on the ``coord_type``:
        - 'grid' (Visium) - number of neighboring tiles.
        - 'generic' - number of neighborhoods for non-grid data.
        
    radius : float, optional
        Default is None.
        Only available when coord_type = 'generic'. Depending on the type:
        - :class:`float` - compute the graph based on neighborhood radius.
        - :class:`tuple` - prune the final graph to only contain
            edges in interval `[min(radius), max(radius)]`.

    **kwargs
        Keyword arguments for matplotlib.pyplot.text()
    Returns:
    -------
    ax_dictionary : dictionary
        The returned dictionary containse matplotlib.axes.Axes
        under 'Ax' key and optional matrix under 'Matrix' key.
        If stratify is used, the function will return dictionary
        of Axes and optional matrix with keys representing
        the stratification groups.
        For example: if stratify is not used and matrix is called,
        the matrix can be acquired by result['Matrix'], and axes
        from result['Ax']. If stratify is used and has two levels,
        "A" and "B", the axes for A can be extracted by
        result['Ax']['A'] and matrix through result['Matrix']['A'].

           
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
                ax,
                return_matrix=False,
                title=None,
                seed=None,
                **kwargs
            ):

        # Calculate Neighborhood_Enrichment
        if return_matrix:
            matrix = sq.gr.nhood_enrichment(
                        adata,
                        copy=True,
                        seed=seed,
                        cluster_key=new_annotation_name
                )

            sq.gr.nhood_enrichment(
                        adata,
                        seed=seed,
                        cluster_key=new_annotation_name
                )
        else:
            sq.gr.nhood_enrichment(
                        adata,
                        seed=seed,
                        cluster_key=new_annotation_name
                )

        # Plot Neighborhood_Enrichment
        sq.pl.nhood_enrichment(
                    adata,
                    cluster_key=new_annotation_name,
                    title=title,
                    ax=ax,
                    **kwargs
            )


        if return_matrix:
            return [ax, matrix]
        else:
            return ax

    def Cluster_Interaction_Matrix_Analysis(
                adata,
                new_annotation_name,
                ax,
                return_matrix=False,
                title=None,
                **kwargs
            ):

        # Calculate Cluster_Interaction_Matrix

        if return_matrix:
            matrix = sq.gr.interaction_matrix(
                    adata,
                    cluster_key=new_annotation_name,
                    copy=True
            )

            sq.gr.interaction_matrix(
                    adata,
                    cluster_key=new_annotation_name
            )

        else:
            sq.gr.interaction_matrix(
                    adata,
                    cluster_key=new_annotation_name
            )

        sq.pl.interaction_matrix(
                    adata,
                    title=title,
                    cluster_key=new_annotation_name,
                    ax=ax,
                    **kwargs
            )

        if return_matrix:
            return [ax, matrix]
        else:
            return ax

    # Perfrom the actual analysis, first call sq.gr.spatial_neighbors
    # to calculate neighboring graph, then do different analysis.
    def perform_analysis(
            adata,
            analysis_method,
            new_annotation_name,
            ax,
            coord_type,
            n_rings,
            n_neighs,
            radius,
            return_matrix=False,
            title=None,
            seed=None,
            **kwargs
    ):

        sq.gr.spatial_neighbors(
            adata,
            coord_type=coord_type,
            n_rings=n_rings,
            n_neighs=n_neighs,
            radius=radius
        )

        if analysis_method == "Neighborhood Enrichment":
            ax = Neighborhood_Enrichment_Analysis(
                    adata,
                    new_annotation_name,
                    ax,
                    return_matrix,
                    title,
                    seed,
                    **kwargs)

        elif analysis_method == "Cluster Interaction Matrix":
            ax = Cluster_Interaction_Matrix_Analysis(
                    adata,
                    new_annotation_name,
                    ax,
                    return_matrix,
                    title,
                    **kwargs)

        return ax

    # Error Check Section
    # -----------------------------------------------
    if not isinstance(adata, anndata.AnnData):
        error_text = "Input data is not an AnnData object. " + \
            f"Got {str(type(adata))}"
        raise ValueError(error_text)
    
    check_annotation(
        adata,
        annotations=annotation,
        parameter_name="annotation",
        should_exist=True)

    # Check if stratify_by is list or list of str
    check_annotation(
        adata,
        annotations=stratify_by,
        parameter_name="stratify_by",
        should_exist=True)

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
        fig, ax = plt.subplots()

    # Operational Section
    # -----------------------------------------------

    # Create a categorical column data for plotting
    new_annotation_name = annotation + "_plot"

    adata.obs[new_annotation_name] = pd.Categorical(
        adata.obs[annotation])

    if stratify_by:
        if isinstance(stratify_by, list):
            adata.obs['concatenated_obs'] = \
                adata.obs[stratify_by].astype(str).agg('_'.join, axis=1)
        else:
            adata.obs['concatenated_obs'] = \
                adata.obs[stratify_by]

    # Compute a connectivity matrix from spatial coordinates
    if stratify_by:
        ax_dictionary = {}
        matrix_dictionary = {}
        unique_values = adata.obs['concatenated_obs'].unique()
        
        for subset_key in unique_values:
            # Subset the original AnnData object based on the unique value
            subset_adata = adata[
                adata.obs['concatenated_obs'] == subset_key
            ].copy()

            fig, ax = plt.subplots()

            image_title = f"Group: {subset_key}"

            ax = perform_analysis(
                            subset_adata,
                            analysis_method,
                            new_annotation_name,
                            ax,
                            coord_type,
                            n_rings,
                            n_neighs,
                            radius,
                            return_matrix,
                            image_title,
                            seed,
                            **kwargs
                        )

            if return_matrix:
                ax_dictionary[subset_key] = ax[0]
                matrix_dictionary[subset_key] = ax[1]
            else:
                ax_dictionary[subset_key] = ax

            del subset_adata

        if return_matrix:
            results = {
                "Ax": ax_dictionary,
                "Matrix": matrix_dictionary
            }

        else:
            results = {"Ax": ax_dictionary}

    else:
        ax = perform_analysis(
                adata,
                analysis_method,
                new_annotation_name,
                ax,
                coord_type,
                n_rings,
                n_neighs,
                radius,
                return_matrix,
                seed=seed,
                **kwargs
            )

        if return_matrix:
            results = {
                "Ax": ax[0],
                "Matrix": ax[1]
            }
        else:
            results = {"Ax": ax}

    return results
