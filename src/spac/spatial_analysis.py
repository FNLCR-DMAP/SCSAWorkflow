import squidpy as sq
import matplotlib.pyplot as plt
import pandas as pd
import anndata
from spac.utils import (
    check_annotation,
    check_table,
    check_distances,
    check_label,
)
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import distance_matrix
from sklearn.preprocessing import LabelEncoder
from functools import partial
import logging


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
        cmap="seismic",
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
        The column name of the annotation (e.g., phenotypes) to analyze in the
        provided dataset.

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
        If False, the function will return only the axes dictionary.

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

    cmap : str, default 'seismic'
        The colormap to use for the plot. The 'seismic' color map
        consist of three color regions: red for positive, blue for negative,
        and the white at the center. This color map effectively represents
        the nature of the spatial interaction analysis results,
        where positive values indicate clustering and
        negative values indicate seperation. For more color maps, please visit
        https://matplotlib.org/stable/tutorials/colors/colormaps.html

    **kwargs
        Keyword arguments for matplotlib.pyplot.text()

    Returns
    -------
    dict
        A dictionary containing the results of the spatial interaction
        analysis. The keys of the dictionary depend on the parameters
        passed to the function:

        Ax : dict or matplotlib.axes.Axes
            If `stratify_by` is not used, returns a single
            matplotlib.axes.Axes object. If `stratify_by` is used,
            returns a dictionary of Axes objects, with keys
            representing the stratification groups.

        Matrix : dict, optional
            Contains processed DataFrames of computed matrices with row and
            column labels applied. If `stratify_by` is used, the keys represent
            the stratification groups. For example:
            - `results['Matrix']['GroupA']` for a specific stratification group.
            - If `stratify_by` is not used, the table is accessible via
            `results['Matrix']['annotation']`.


    """

    # List all available methods
    available_methods = [
        "Neighborhood Enrichment",
        "Cluster Interaction Matrix"
    ]
    available_methods_str = ",".join(available_methods)

    # pacakge each methods into a function to allow
    # centralized control and improve flexibility
    def _Neighborhood_Enrichment_Analysis(
                adata,
                categorical_annotation,
                ax,
                return_matrix=False,
                title=None,
                seed=None,
                **kwargs
            ):
        """
        Perform Neighborhood Enrichment analysis.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object.

        categorical_annotation : str
            Name of the annotation column to analyze.

        ax : matplotlib.axes.Axes
            Axes to plot the enrichment results.

        return_matrix : bool
            If True, returns the enrichment matrix.

        title : str, optional
            Title of the plot.

        seed : int, optional
            Random seed for reproducibility.

        **kwargs : dict
            Additional keyword arguments for the plot.

        Returns
        -------
        ax or [ax, matrix]
            The plot axes or axes and enrichment matrix.
        """
        # Calculate Neighborhood_Enrichment
        if return_matrix:
            matrix = sq.gr.nhood_enrichment(
                        adata,
                        copy=True,
                        seed=seed,
                        cluster_key=categorical_annotation
                )

            sq.gr.nhood_enrichment(
                        adata,
                        seed=seed,
                        cluster_key=categorical_annotation
                )
        else:
            sq.gr.nhood_enrichment(
                        adata,
                        seed=seed,
                        cluster_key=categorical_annotation
                )

        # Plot Neighborhood_Enrichment
        sq.pl.nhood_enrichment(
                    adata,
                    cluster_key=categorical_annotation,
                    title=title,
                    ax=ax,
                    **kwargs
            )

        if return_matrix:
            return [ax, matrix]
        else:
            return ax

    def _Cluster_Interaction_Matrix_Analysis(
                adata,
                categorical_annotation,
                ax,
                return_matrix=False,
                title=None,
                **kwargs
            ):
        """
        Perform Cluster Interaction Matrix analysis.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object.

        categorical_annotation : str
            Name of the annotation column to analyze.

        ax : matplotlib.axes.Axes
            Axes to plot the interaction matrix.

        return_matrix : bool
            If True, returns the interaction matrix.

        title : str, optional
            Title of the plot.

        **kwargs : dict
            Additional keyword arguments for the plot.

        Returns
        -------
        ax or [ax, matrix]
            The plot axes or axes and interaction matrix.
        """
        # Calculate Cluster_Interaction_Matrix

        if return_matrix:
            matrix = sq.gr.interaction_matrix(
                    adata,
                    cluster_key=categorical_annotation,
                    copy=True
            )

            sq.gr.interaction_matrix(
                    adata,
                    cluster_key=categorical_annotation
            )

        else:
            sq.gr.interaction_matrix(
                    adata,
                    cluster_key=categorical_annotation
            )

        sq.pl.interaction_matrix(
                    adata,
                    title=title,
                    cluster_key=categorical_annotation,
                    ax=ax,
                    **kwargs
            )

        if return_matrix:
            return [ax, matrix]
        else:
            return ax

    # Perfrom the actual analysis, first call sq.gr.spatial_neighbors
    # to calculate neighboring graph, then do different analysis.
    def _perform_analysis(
            adata,
            analysis_method,
            categorical_annotation,
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
        """
        Perform the specified spatial analysis method.

        Parameters
        ----------
        Same as parent function.

        Returns
        -------
        ax or [ax, matrix]
            The plot axes or axes and matrix results.
        """
        sq.gr.spatial_neighbors(
            adata,
            coord_type=coord_type,
            n_rings=n_rings,
            n_neighs=n_neighs,
            radius=radius
        )

        if analysis_method == "Neighborhood Enrichment":
            ax = _Neighborhood_Enrichment_Analysis(
                    adata,
                    categorical_annotation,
                    ax,
                    return_matrix,
                    title,
                    seed,
                    **kwargs)

        elif analysis_method == "Cluster Interaction Matrix":
            ax = _Cluster_Interaction_Matrix_Analysis(
                    adata,
                    categorical_annotation,
                    ax,
                    return_matrix,
                    title,
                    **kwargs)

        return ax

    def _get_labels(
        fig,
        unique_annotations,
        verbose=False
    ):
        """
        Extract row and column labels from plot axes.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure containing the plots.

        unique_annotations : list
            List of unique annotation labels.

        verbose : bool, default False
            If True, print debugging information.

        Returns
        -------
        list
            List of row labels.
        """
        row_labels = []
        # Iterate over all axes to check if any contain the row labels
        for i, ax in enumerate(fig.axes):
            if verbose:
                print(f"Inspecting axis {i}...")
            # Try to extract labels from the y-axis of each axis
            yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
            xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]

            if yticklabels and xticklabels:
                raise ValueError(
                    "Both x- and y-axis labels found on axis. "
                    "Unable to determine row labels."
                )

            elif yticklabels and not xticklabels:
                if set(yticklabels) <= set(unique_annotations):
                    if verbose:
                        print(f"Row labels found on axis {i}: {yticklabels}")
                    row_labels = yticklabels[::-1]

            # Try extracting other possible labels (x-axis, title, etc.)
            elif xticklabels and not yticklabels:
                if set(xticklabels) <= set(unique_annotations):
                    if verbose:
                        print(
                            f"Column labels found on axis {i}: {xticklabels}"
                        )
                    row_labels = xticklabels[::-1]
            else:
                    if verbose:
                        print(
                            "No labels found on axis. Unable to determine row labels."
                        )

        return row_labels

    # Use to process the output from different
    # spatial analysis method in squidpy
    def _process_matrixs(
        matrixs,
        row_labels,
        plot_label=None
    ):
        """
        Process the output matrices for saving and visualization.

        This function organizes matrices produced during the spatial analysis,
        adding appropriate labels and creating a dictionary for easy access.
        The processed matrices are returned in a labeled format suitable
        for saving or further analysis.

        Parameters
        ----------
        matrixs : dict
            Dictionary of raw matrices generated from spatial analyses.

        row_labels : list
            List of row and column labels for the matrices, used to annotate
            the resulting DataFrames.

        plot_label : str, optional
            Additional label to append to the matrix file names, useful for
            distinguishing between different stratification groups or analysis
            scenarios.

        Returns
        -------
        dict
            A dictionary of labeled matrices, where the keys are the file names
            (including the annotation and analysis method) and the values are
            pandas DataFrames representing the matrices.

        Notes
        -----
        For "Cluster Interaction Matrix", a single matrix is processed.
        For "Neighborhood Enrichment", multiple matrices (e.g., z-score and
        enrichment counts) are processed separately.
        """
        return_dict = {}
        if analysis_method == "Cluster Interaction Matrix":
            if not isinstance(matrixs, pd.DataFrame):
                matrix = pd.DataFrame(matrixs)
            if len(row_labels) > 0:
                matrix.index = row_labels
                matrix.columns = row_labels

            if plot_label is None:
                file_name = f"{annotation}_{output_file_cat_list[0]}" + \
                            "_interaction_matrix.csv"
            else:
                file_name = f"{annotation}_{output_file_cat_list[0]}" + \
                            f"_{plot_label}_interaction_matrix.csv"

            return_dict[file_name] = matrix

        elif analysis_method == "Neighborhood Enrichment":
            for i, matrix in enumerate(matrixs):
                # Convert each 2D array to a DataFrame
                if not isinstance(matrix, pd.DataFrame):
                    matrix = pd.DataFrame(matrix)

                if len(row_labels) > 0:
                    matrix.index = row_labels
                    matrix.columns = row_labels

                if plot_label is None:
                    file_name = f"{annotation}_{output_file_cat_list[i]}" + \
                                "_interaction_matrix.csv"
                else:
                    file_name = f"{annotation}_{output_file_cat_list[i]}" + \
                                f"_{plot_label}" + \
                                "_interaction_matrix.csv"

                return_dict[file_name] = matrix
        return return_dict

    # -----------------------------------------------
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

    # Create a categorical column data for plotting\
    # This is to avoid modifying the original annotation to comply with
    # the squidpy function requirements

    categorical_annotation = annotation + "_plot"

    adata.obs[categorical_annotation] = pd.Categorical(
        adata.obs[annotation])

    if stratify_by:
        if isinstance(stratify_by, list):
            adata.obs['_spac_utils_concat_obs'] = \
                adata.obs[stratify_by].astype(str).agg('_'.join, axis=1)
        else:
            adata.obs['_spac_utils_concat_obs'] = \
                adata.obs[stratify_by]

    # Partial function for the _perform_analysis function
    # to allow for uniform parameter passing for both stratified
    # and non-stratified analysis
    perform_analysis_prefilled = partial(
        _perform_analysis,
        analysis_method=analysis_method,
        categorical_annotation=categorical_annotation,
        coord_type=coord_type,
        n_rings=n_rings,
        n_neighs=n_neighs,
        radius=radius,
        return_matrix=return_matrix,
        seed=seed,
        cmap=cmap
    )

    # Compute a connectivity matrix from spatial coordinates
    if stratify_by:
        ax_dictionary = {}
        matrix_dictionary = {}
        unique_values = adata.obs['_spac_utils_concat_obs'].unique()

        for subset_key in unique_values:
            # Subset the original AnnData object based on the unique value
            subset_adata = adata[
                adata.obs['_spac_utils_concat_obs'] == subset_key
            ].copy()

            fig, ax = plt.subplots()

            image_title = f"Group: {subset_key}"

            ax = perform_analysis_prefilled(
                            adata=subset_adata,
                            ax=ax,
                            title=image_title,
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
        ax = perform_analysis_prefilled(
                adata=adata,
                ax=ax,
                **kwargs
            )

        if return_matrix:
            results = {
                "Ax": ax[0],
                "Matrix": ax[1]
            }
        else:
            results = {"Ax": ax}

    # Adding post processing methods for updating images and retrieve

    # Acquire the annotation labels and acquire
    # the column names from axes and matrixs

    if return_matrix:
        output_file_cat_grosarry = {
            "Neighborhood Enrichment": ["z_score", "enrichment_counts"],
            "Cluster Interaction Matrix": ["interaction_counts"]
        }

        output_file_cat_list = output_file_cat_grosarry[analysis_method]

        unique_annotations = list(adata.obs[annotation].unique())

        _matrixs = results['Matrix']
        _axs = results['Ax']

        def _processes_function_return(
            matrix,
            ax,
            plot_label=None
        ):
            fig = ax.get_figure()
            row_labels = _get_labels(
                    fig,
                    unique_annotations
                )
            result_dict = _process_matrixs(
                matrix,
                row_labels,
                plot_label
            )
            return result_dict

        table_results = {}
        if stratify_by:
            for key in _axs:
                _ax = _axs[key]
                _matrix = _matrixs[key]
                result_dict = _processes_function_return(
                    _matrix,
                    _ax,
                    key
                )
                table_results[key] = result_dict

        else:
            table_results['annotation'] = _processes_function_return(
                _matrixs,
                _axs
            )

        results['Matrix'] = table_results

    # Clean up the temporary columns
    adata.obs.drop(categorical_annotation, axis=1, inplace=True)
    if stratify_by:
        adata.obs.drop('_spac_utils_concat_obs', axis=1, inplace=True)

    return results


def ripley_l(
    adata,
    annotation,
    phenotypes,
    distances,
    regions=None,
    spatial_key="spatial",
    n_simulations=1,
    area=None,
    seed=42,
    edge_correction=True
):
    """
    Calculate Ripley's L statistic for spatial data in `adata`.

    Ripley's L statistic is a spatial point pattern analysis metric that
    quantifies clustering or regularity in point patterns across various
    distances. This function calculates the statistic for each region in
    `adata` (if provided) or for all cells if regions are not specified.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix containing the spatial coordinates and
        cell phenotypes.
    annotation : str
        The key in `adata.obs` representing the annotation for cell phenotypes.
    phenotypes : list of str
        A list containing two phenotypes for which the Ripley L statistic
        will be calculated. If the two phenotypes are the same, the
        calculation is done for the same type; if different, it considers
        interactions between the two.
    distances : array-like
        An array of distances at which to calculate Ripley's L statistic.
        The values must be positive and incremental.
    regions : str or None, optional
        The key in `adata.obs` representing regions for stratifying the
        data. If `None`, all cells will be treated as one region.
    spatial_key : str, optional
        The key in `adata.obsm` representing the spatial coordinates.
        Default is `"spatial"`.
    n_simulations : int, optional
        Number of simulations to perform for significance testing. Default is 100.
    area : float or None, optional
        The area of the spatial region of interest. If `None`, the area
        will be inferred from the data. Default is `None`.
    seed : int, optional
        Random seed for simulation reproducibility. Default is 42.
    edge_correction: boo, optional
        If True, apply edge correction to the Ripley's L calculation.


    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Ripley's L results for each region or
        the entire dataset if `regions` is `None`. The DataFrame includes
        the following columns:
        - `region`: The region label or 'all' if no regions are specified.
        - `center_phenotype`: The first phenotype in `phenotypes`.
        - `neighbor_phenotype`: The second phenotype in `phenotypes`.
        - `ripley_l`: The Ripley's L statistic calculated for the region.
        - `config`: A dictionary with configuration settings used for the calculation.

    Notes
    -----
    Ripley's L is an adjusted version of Ripley's K that corrects for the
    inherent increase in point-to-point distances as the distance grows.
    This statistic is used to evaluate spatial clustering or dispersion
    of points (cells) in biological datasets.

    The function uses pre-defined distances and performs simulations to
    assess the significance of observed patterns. The results are stored
    in the `.uns` attribute of `adata` under the key `'ripley_l'`, or in
    a new DataFrame if no prior results exist.

    Examples
    --------
    Calculate Ripley's L for two phenotypes in a single region dataset:

    >>> result = ripley_l(adata, annotation='cell_type', phenotypes=['A', 'B'], distances=np.linspace(0, 500, 100))

    Calculate Ripley's L for multiple regions in `adata`:

    >>> result = ripley_l(adata, annotation='cell_type', phenotypes=['A', 'B'], distances=np.linspace(0, 500, 100), regions='region_key')

    """

    # Check that distances is array-like with incremental positive values
    check_distances(distances)

    # Check that annotation and phenotypes exist in adata.obs
    check_annotation(adata, annotations=[annotation], should_exist=True)

    check_label(adata, annotation, phenotypes)

    # Convert annotations to categorical
    adata.obs[annotation] = pd.Categorical(adata.obs[annotation])

    if regions is not None:
        check_annotation(adata, annotations=[regions], should_exist=True)

    # Import ripley function from the spac library
    from spac._ripley import ripley
    from functools import partial

    # Partial function for Ripley calculation
    ripley_func = partial(
        ripley,
        cluster_key=annotation,
        mode='L',
        spatial_key=spatial_key,
        phenotypes=phenotypes,
        support=distances,
        n_simulations=n_simulations,
        seed=seed,
        area=area,
        copy=True,
        edge_correction=edge_correction
    )

    # Check if adata already has ripley_l results,
    # else initialize a result DataFrame
    if 'ripley_l' in adata.uns.keys():
        results = adata.uns['ripley_l']
    else:
        results = None

    # Function to process Ripley L calculation for a region
    def process_region(adata_region, region_label):
        # Calculate number of cells in the region for the phenotypes
        # n_cells = get_ncells(adata_region, annotation, phenotypes)

        region_cells = adata_region.n_obs
        # Log the region and cell info
        print(
            f'Processing region:"{region_label}".\n'
            f'Cells in region:"{region_cells}"'
        )

        cell_counts = adata_region.obs[annotation].value_counts()

        if region_cells < 3:
            message = (
                f'WARNING, not enough cells in region "{region_label}". '
                f'Number of cells "{region_cells}". '
                'Skipping Ripley L.'
            )
            print(message)
            ripley_result = None
        elif not phenotypes[0] in cell_counts.index:
            message = (
                f'WARNING, phenotype "{phenotypes[0]}" '
                f'not found in region "{region_label}", skipping Ripley L.'
            )
            print(message)
            ripley_result = None
        elif not phenotypes[1] in cell_counts.index:
            message = (
                f'WARNING, phenotype "{phenotypes[1]}" '
                f'not found in region "{region_label}", skipping Ripley L.'
            )
            print(message)
            ripley_result = None
        else:
            # Calculate Ripley's L statistic using the partial function
            ripley_result = ripley_func(adata=adata_region)
            message = "Ripley's L successfully calculated."

        # Create a result entry for the region
        new_result = {
            'region': region_label,
            'center_phenotype': phenotypes[0],
            'neighbor_phenotype': phenotypes[1],
            'ripley_l': ripley_result,
            'region_cells': region_cells,
            'message': message,
            'n_simulations': n_simulations,
            'seed': seed
        }

        return new_result

    def append_results(results, new_result):
        # Convert the new result to a DataFrame
        new_df = pd.DataFrame([new_result])

        # Check if the dataframe exists and concatenate
        if results is not None:
            results = pd.concat([results, new_df], ignore_index=True)
        else:
            results = new_df
        return results

    # If regions are provided, process each region,
    # else process all cells as a single region
    if regions is not None:
        for region in adata.obs[regions].unique():
            adata_region = adata[adata.obs[regions] == region]
            new_result = process_region(adata_region, region)
            results = append_results(results, new_result)
    else:
        new_result = process_region(adata, 'all')
        results = append_results(results, new_result)

    # Save the results in the AnnData object
    adata.uns['ripley_l'] = results
    return results


def neighborhood_profile(
    adata,
    phenotypes,
    distances,
    regions=None,
    spatial_key="spatial",
    normalize=None,
    associated_table_name="neighborhood_profile"
):

    """
    Calculate the neighborhood profile for every cell in all slides in an analysis
    and update the input AnnData object in place.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the spatial coordinates and phenotypes.

    phenotypes : str
        The name of the column in adata.obs that contains the phenotypes.

    distances : list
        The list of increasing distances for the neighborhood profile.

    spatial_key : str, optional
        The key in adata.obs that contains the spatial coordinates. Default is
        'spatial'.

    normalize : str or None, optional
        If 'total_cells', normalize the neighborhood profile based on the
        total number of cells in each bin.
        If 'bin_area', normalize the neighborhood profile based on the area
        of every bin.  Default is None.

    associated_table_name : str, optional
        The name of the column in adata.obsm that will contain the
        neighborhood profile. Default is 'neighborhood_profile'.

    regions : str or None, optional
        The name of the column in adata.obs that contains the regions.
        If None, all cells in adata will be used. Default is None.


    Returns
    -------
    None
        The function modifies the input AnnData object in place, adding a new
        column containing the neighborhood profile to adata.obsm.

    Notes
    -----
    The input AnnData object 'adata' is modified in place. The function adds a
    new column containing the neighborhood profile to adata.obsm, named by the
    parameter 'associated_table_name'. The associated_table_name is a 3D array of
    shape (n_cells, n_phenotypes, n_bins) where n_cells is the number of cells
    in the all slides, n_phenotypes is the number of unique phenotypes, and
    n_bins is the number of bins in the distances list.

    A dictionary is added to adata.uns[associated_table_name] with the two keys
    "bins" and "labels". "labels" will store all the values in the phenotype
    annotation.
    """

    # Check that distances is array like with incremental positive values
    check_distances(distances)

    # Check that phenotypes is adata.obs
    check_annotation(
        adata,
        annotations=[phenotypes],
        should_exist=True)

    # Check that phenotypes is adata.obs
    if regions is not None:
        check_annotation(
            adata,
            annotations=[regions],
            should_exist=True)

    check_table(
        adata=adata,
        tables=spatial_key,
        should_exist=True,
        associated_table=True
    )

    # Check the values of normalize
    if normalize is not None and normalize not in ['total_cells', 'bin_area']:
        raise ValueError((f'normalize must be "total_cells", "bin_area"'
                          f' or None. Got "{normalize}"'))

    # Check that the associated_table_name does not exist.
    # Raise a warning othewise
    check_table(
        adata=adata,
        tables=associated_table_name,
        should_exist=False,
        associated_table=True,
        warning=True
    )

    logger = logging.getLogger()

    # Convert the phenotypes to integers using label encoder
    labels = adata.obs[phenotypes].values
    le = LabelEncoder().fit(labels)
    n_phenotypes = len(le.classes_)

    # Create a place holder for the neighborhood profile
    all_cells_profiles = np.zeros(
        (adata.n_obs, n_phenotypes, len(distances)-1))

    # If regions is None, use all cells in adata
    if regions is not None:
        # Calculate the neighborhood profile for every slide
        for i, region in enumerate(adata.obs[regions].unique()):
            adata_region = adata[adata.obs[regions] == region]
            logger.info(f"Processing region:{region} \
                        n_cells:{len(adata_region)}")
            positions = adata_region.obsm[spatial_key]
            labels_id = le.transform(adata_region.obs[phenotypes].values)
            region_profiles = _neighborhood_profile_core(
               positions,
               labels_id,
               n_phenotypes,
               distances,
               normalize
            )

            # Updated profiles of the cells of the current slide
            all_cells_profiles[adata.obs[regions] == region] = (
               region_profiles
            )
    else:
        logger.info(("Processing all cells as a single region."
                     f" n_cells:{len(adata)}"))
        positions = adata.obsm[spatial_key]
        labels_id = le.transform(labels)
        all_cells_profiles = _neighborhood_profile_core(
            positions,
            labels_id,
            n_phenotypes,
            distances,
            normalize
        )

    # Add the neighborhood profile to the AnnData object
    adata.obsm[associated_table_name] = all_cells_profiles

    # Store the bins and the lables in uns
    summary = {"bins": distances, "labels": le.classes_}
    if associated_table_name in adata.uns:
        logger.warning(f"The analysis already contains the \
                       unstructured value:{associated_table_name}. \
                       It will be overwriten")
    adata.uns[associated_table_name] = summary


def _neighborhood_profile_core(
        coord,
        phenotypes,
        n_phenotypes,
        distances_bins,
        normalize=None
):
    """
    Calculate the neighborhood profile for every cell in a region.

    Parameters
    ----------
    coord : numpy.ndarray
        The coordinates of the cells in the region. Should be a 2D array of
        shape (n_cells, 2) representing x, y coordinates.

    phenotypes : numpy.ndarray
        The phenotypes of the cells in the region.

    n_phenotypes : int
        The number of unique phenotypes in the region.

    distances_bins : list
        The bins defining the distance ranges for the neighborhood profile.

    normalize : str or None, optional
        If 'total_cells', normalize the neighborhood profile based on the
        total number of cells in each bin.
        If 'bin_area', normalize the neighborhood profile based on the area
        of every bin.

    Returns
    -------
    numpy.ndarray
        A 3D array containing the neighborhood profile for every cell in the
        region. The dimensions are (n_cells, n_phenotypes, n_intervals).

    Notes
    -----
    - The function calculates the neighborhood profile for each cell, which
      represents the distribution of neighboring cells' phenotypes within
      different distance intervals.
    - The 'distances_bins' parameter should be a list defining the bins for
      the distance intervals. It is assumed that the bins are incremental,
      starting from 0.
    """

    # TODO Check that distances bins is incremental

    max_distance = distances_bins[-1]
    kdtree = KDTree(coord)

    # indexes is a list of neighbors coordinate for every
    # cell
    indexes = kdtree.query_ball_tree(kdtree, r=max_distance)

    # Create phenotype bins to include the integer equivalent of
    # every phenotype to use the histogram2d function instead of
    # a for loop over every phenotype
    phenotype_bins = np.arange(-0.5, n_phenotypes + 0.5, 1)
    n_intervals = len(distances_bins) - 1

    neighborhood_profile = []
    for i, neighbors in enumerate(indexes):

        # Query_ball_tree will include the point itself
        neighbors.remove(i)

        # To potentially save on calculating the histogram
        if len(neighbors) == 0:
            neighborhood_profile.append(np.zeros((n_phenotypes, n_intervals)))
        else:
            neighbor_coords = coord[neighbors]
            dist_matrix = distance_matrix(coord[i:i+1], neighbor_coords)[0]
            neighbors_phenotypes = phenotypes[neighbors]
            # Returns a 2D histogram of size n_phenotypes * n_intervals
            histograms_array, _ , _ = np.histogram2d(neighbors_phenotypes,
                                                     dist_matrix,
                                                     bins=[
                                                       phenotype_bins,
                                                       distances_bins
                                                       ])
            neighborhood_profile.append(histograms_array)

    neighborhood_array = np.stack(neighborhood_profile)
    if normalize == "total_cells":
        bins_sum = neighborhood_array.sum(axis=1)
        bins_sum[bins_sum == 0] = 1
        neighborhood_array = neighborhood_array / bins_sum[:, np.newaxis, :]
    elif normalize == "bin_area":
        circles_areas = np.pi * np.array(distances_bins)**2
        bins_areas = np.diff(circles_areas)
        neighborhood_array = neighborhood_array / bins_areas[np.newaxis, np.newaxis, :]

    return neighborhood_array


def calculate_nearest_neighbor(
    adata,
    annotation,
    spatial_associated_table='spatial',
    imageid=None,
    label='spatial_distance',
    verbose=True
):
    """
    Computes the shortest distance from each cell to the nearest cell of
    each phenotype (via scimap.tl.spatial_distance) and stores the
    resulting DataFrame in `adata.obsm[label]`.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with spatial information.
    annotation : str
        Column name in `adata.obs` containing cell annotationsi (i.e.
        phenotypes).
    spatial_associated_table : str, optional
        Key in `adata.obsm` where spatial coordinates are stored. Default is
        'spatial'.
    imageid : str, optional
        The column in `adata.obs` specifying image IDs. If None,
        a dummy image column is created temporarily. Spatial distances are
        computed across the entire dataseti as if it's one image.
    label : str, optional
        The key under which results are stored  in `adata.obsm`. Default is
        'spatial_distance'.
    verbose : bool, optional
        If True, prints progress messages. Default is True.

    Returns
    -------
    None
        Modifies `adata` in place by storing a DataFrame of
        spatial distances in `adata.obsm[label]`.

    Example
    -------
    For a dataset with two cells (CellA, CellB) both of the same phenotype
    "type1", the output might look like:

    >>> adata.obsm['spatial_distance']
           type1
    CellA    0.0
    CellB    0.0

    For a dataset with two phenotypes "type1" and "type2", the output might
    look like:

    >>> adata.obsm['spatial_distance']
           type1     type2
    CellA   0.00  1.414214
    CellB  1.414214  0.00

    Input:
        adata.obs:
            cell_type   imageid
            type1       image1
            type1       image1
            type2       image1

        adata.obsm['spatial']:
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

    Output stored in adata.obsm['spatial_distance']:
            type1   type2
        0    0.0    1.414
        1    1.414  0.0
        2    2.236  1.0

    Raises
    ------
    ValueError
        If `spatial_associated_table` is not found in `adata.obsm`.
        If spatial coordinates are missing or invalid.
    """

    import scimap as sm

    # Input validation for annotation
    check_annotation(adata, annotations=annotation)

    # Validate and extract spatial coordinates
    check_table(
        adata,
        tables=spatial_associated_table,
        associated_table=True,
        should_exist=True
    )
    coords = adata.obsm[spatial_associated_table]

    if coords.shape[1] < 2:
        raise ValueError(
            "The input data must include coordinates with at least "
            "two dimensions, such as X and Y positions."
        )

    # Check for missing coordinates
    if np.isnan(coords).any():
        missing_cells = np.where(np.isnan(coords).any(axis=1))[0]
        raise ValueError(
            f"Missing values found in spatial coordinates for cells "
            f"at indices: {missing_cells}."
        )

    if verbose:
        print("Preparing data for spatial distance calculation...")

    # Add coordinates to adata.obs temporarily
    adata.obs['_x_coord'] = coords[:, 0]
    adata.obs['_y_coord'] = coords[:, 1]

    use_z = False
    if coords.shape[1] > 2:
        adata.obs['_z_coord'] = coords[:, 2]
        use_z = True

    # Handle imageid logic
    dummy_column_created = False
    original_imageid = imageid
    if imageid is None:
        dummy_column_created = True
        imageid = '_dummy_imageid'
        adata.obs[imageid] = 'dummy_image'  # Treat all cells as one 'image'

    sm.tl.spatial_distance(
        adata=adata,
        x_coordinate='_x_coord',
        y_coordinate='_y_coord',
        z_coordinate=('_z_coord' if use_z else None),
        phenotype=annotation,
        imageid=imageid,
        verbose=verbose,
        label=label
    )

    # The scimap function stores the result in adata.uns[label].
    # Need to align it to adata.obs_names before placing into .obsm.
    result_df = adata.uns.pop(label)  # remove from uns and capture

    # Reindex the results to match adata.obs_names.
    result_df = result_df.reindex(adata.obs_names)

    adata.obsm[label] = result_df

    # Remove temporary coordinates from adata.obs
    drop_cols = ['_x_coord', '_y_coord']
    if use_z:
        drop_cols.append('_z_coord')
    adata.obs.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Remove dummy column if it was created
    if dummy_column_created:
        adata.obs.drop(columns=[imageid], inplace=True, errors='ignore')
        imageid = original_imageid  # restore the original state

    if verbose:
        print(f"Spatial distances stored in adata.obsm['{label}']")
        print("Preview of the distance DataFrame:\n", adata.obsm[label].head())
