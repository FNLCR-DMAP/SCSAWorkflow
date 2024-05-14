import re
import anndata as ad
import numpy as np
import matplotlib.cm as cm
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def regex_search_list(
        regex_patterns,
        list_to_search):
    """
    Perfrom regex (regular expression) search in a list and
    return list of strings matching the regex pattern

    Parameters
    ----------
    regex_pattern : str or a list of str
        The regex pattern to look for,
        single pattern or a list of patterns.

    list_to_search : list of str
        A list of string to seach for string that matches regex pattern.

    Returns
    -------
    list of str
        A list of strings containing results from search.

    Example
    -------
    >>> regex_pattern = ["A", "^B.*"]
    >>> list_to_search = ["ABC", "BC", "AC", "AB"]
    >>> result = regex_search_list(regex_pattern, list_to_search)
    >>> print(result)
    ['BC']
    """

    if not isinstance(list_to_search, list):
        raise TypeError("Please provide a list to search.")

    if not isinstance(regex_patterns, list):
        if not isinstance(regex_patterns, str):
            error_text = "Regrex pattern provided " + \
                "is not list nor a string"
            raise TypeError(error_text)
        else:
            regex_patterns = [regex_patterns]
    else:
        if not all(isinstance(item, str) for item in regex_patterns):
            error_text = "All items in the pattern " + \
                "list must be of type string."
            raise TypeError(error_text)

    def regex_search(pattern, str):
        try:
            found = re.search(pattern, str)
            if found is not None:
                return found.group(0)
        except re.error as e:
            raise ValueError(
                "Error occurred when searching with regex:\n{}\n"
                "Please review your regex pattern: {}\nIf using * at the start"
                ", always have a . before the asterisk.".format(e, pattern)
            )

    all_results = []

    for pattern in regex_patterns:
        print(pattern)
        str_found = [
            regex_search(
                pattern,
                x) for x in list_to_search if regex_search(
                pattern,
                x) in list_to_search
        ]
        all_results.extend(str_found)

    all_results = list(all_results)

    return all_results


def check_list_in_list(
        input,
        input_name,
        input_type,
        target_list,
        need_exist=True,
        warning=False
):

    """
    Check if items in a given list exist in a target list.

    This function is used to validate whether all or none of the
    items in a given list exist in a target list.
    It helps to ensure that the input list contains only valid elements
    that are present in the target list.

    Parameters
    ----------
    input : str or list of str or None
        The input list or a single string element. If it is a string,
        it will be converted to a list containing
        only that string element. If `None`, no validation will be performed.
    input_name : str
        The name of the input list used for displaying helpful error messages.
    input_type : str
        The type of items in the input list
        (e.g., "item", "element", "category").
    target_list : list of str
        The target list containing valid items that the input list elements
        should be compared against.
    need_exist : bool, optional (default=True)
        Determines whether to check if elements exist in the
        target list (True), or if they should not exist (False).

     warning: bool, optional (default=False)
        If true, generate a warning instead of raising an exception



    Raises
    ------
    ValueError
        If the `input` is not a string or a list of strings.
        If `need_exist` is True and any element of the input
            list does not exist in the target list.
        If `need_exist` is False and any element of the input
            list exists in the target list.


    Warns
    -----
    UserWarning
        If the specified behavior is not present 
        and `warning` is True.


    """

    if input is not None:
        if isinstance(input, str):
            input = [input]
        elif not isinstance(input, list):
            raise ValueError(
                f"The '{input_name}' parameter "
                "should be a string or a list of strings."
            )

        target_list_str = "\n".join(target_list)

        if need_exist:
            for item in input:
                if item not in target_list:
                    message = (
                        f"The {input_type} '{item}' "
                        "does not exist in the provided dataset.\n"
                        f"Existing {input_type}s are:\n"
                        f"{target_list_str}"
                    )
                    if warning is False:
                        raise ValueError(message)
                    else:
                        warnings.warn(message)
        else:
            for item in input:
                if item in target_list:
                    message = (
                        f"The {input_type} '{item}' "
                        "exist in the provided dataset.\n"
                        f"Existing {input_type}s are:\n"
                    )
                    if warning is False:
                        raise ValueError(message)
                    else:
                        warnings.warn(message)


def check_table(
        adata,
        tables=None,
        should_exist=True,
        associated_table=False,
        warning=False):

    """
    Perform common error checks for table (layers) or derived tables (obsm) in
    anndata related objects.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be checked.

    tables : str or list of str, optional
        The term "table" is equivalent to layer in anndata structure.
        The layer(s) to check for existence in adata.layers.keys().

    should_exist : bool, optional (default=True)
        Determines whether to check if elements exist in the
        target list (True), or if they should not exist (False).

    associtated_table : bool, optional (default=False)
        Determines whether to check if the passed tables names
        should exist as layers or in obsm in the andata object.

    warning : bool, optional (default=False)
        If True, generate a warning instead of raising an exception.

    Raises
    ------
    TypeError
        If adata is not an instance of anndata.AnnData.

    ValueError
        If any of the specified layers, annotations, obsm, 
        or features do not exist.


    Warns
    -----
    UserWarning
        If any of the specified layers, annotations, obsm,
        or features do not exist, and `warning` is True.

    """

    # Check if adata is an instance of anndata.AnnData
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            "Input dataset should be "
            "an instance of anndata.AnnData, "
            "please check the input dataset source."
            )

    # Check for tables
    if associated_table is False:
        existing_tables = list(adata.layers.keys())
        input_name = "tables"
        input_type = "table"
    else:
        existing_tables = list(adata.obsm.keys())
        input_name = "associated tables"
        input_type = "associated table"

    check_list_in_list(
            input=tables,
            input_name=input_name,
            input_type=input_type,
            target_list=existing_tables,
            need_exist=should_exist,
            warning=warning
        )


def check_annotation(
        adata,
        annotations=None,
        parameter_name=None,
        should_exist=True):

    """
    Perform common error checks for annotations in
    anndata related objects.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be checked.

    annotations :  str or list of str, optional
        The annotation(s) to check for existence in adata.obs.

    should_exist : bool, optional (default=True)
        Determines whether to check if elements exist in the
        target list (True), or if they should not exist (False).

    Raises
    ------
    TypeError
        If adata is not an instance of anndata.AnnData.

    ValueError
        If any of the specified layers, annotations, or features do not exist.

    """

    # Check if adata is an instance of anndata.AnnData
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            "Input dataset should be "
            "an instance of anndata.AnnData, "
            "please check the input dataset source."
            )

    # Check for tables
    existing_annotation = adata.obs.columns.to_list()

    if not parameter_name:
        parameter_name="annotations"

    check_list_in_list(
            input=annotations,
            input_name=parameter_name,
            input_type="annotation",
            target_list=existing_annotation,
            need_exist=should_exist
        )


def check_feature(
        adata,
        features=None,
        should_exist=True):

    """
    Perform common error checks for features in
    anndata related objects.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be checked.

    features : str or list of str, optional
        The feature(s) to check for existence in adata.var_names.

    should_exist : bool, optional (default=True)
        Determines whether to check if elements exist in the
        target list (True), or if they should not exist (False).

    Raises
    ------
    TypeError
        If adata is not an instance of anndata.AnnData.

    ValueError
        If any of the specified layers, annotations, or features do not exist.

    """

    # Check if adata is an instance of anndata.AnnData
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            "Input dataset should be "
            "an instance of anndata.AnnData, "
            "please check the input dataset source."
            )

    # Check for tables
    existing_features = adata.var_names.to_list()
    check_list_in_list(
            input=features,
            input_name="features",
            input_type="feature",
            target_list=existing_features,
            need_exist=should_exist
        )


def check_column_name(
    column_name,
    field_name,
    symbol_checklist="!?,.",
):

    if column_name.find(" ") != -1:
        raise ValueError(f"The {column_name} for {field_name} contains a space character.")
    else:
        if any(symbol in column_name for symbol in symbol_checklist):
            raise ValueError(f"One of the symbols in {symbol_checklist} is present in {column_name} for {field_name}.")


def text_to_others(
    parameter,
    text="None",
    to_None=True,
    to_False=False,
    to_True=False,
    to_Int=False,
    to_Float=False
):
    def check_same(
        parameter,
        text,
        target
    ):
        if parameter == text:
            parameter = target
            return parameter
        else:
            return parameter

    if to_None:
        parameter = check_same(
            parameter,
            text,
            None
        )

    if to_False:
        parameter = check_same(
            parameter,
            text,
            False
        )

    if to_True:
        parameter = check_same(
            parameter,
            text,
            True
        )

    if isinstance(parameter, str):
        if to_Int and to_Float:
            raise ValueError("Please select one numeric conversion at a time, thank you.")

        if to_Int:
            parameter = int(parameter)

        if to_Float:
            parameter = float(parameter)

    return parameter


def annotation_category_relations(
    adata,
    source_annotation,
    target_annotation,
    prefix=False
):
    """
    Calculates the count of unique relationships between two
    annotations in an AnnData object.
    Relationship is defined as a unique pair of values, one from the
    'source_annotation' and one from the 'target_annotation'.

    Returns a DataFrame with columns 'source_annotation', 'target_annotation',
    'count', 'percentage_source', and 'percentage_target'.
    Where 'count' represents the number of occurrences of each relationship,
    percentage_source represents the percentage of the count of link
    over the total count of the source label, and percentage_target represents
    the percentage of the count of link over the total count of the target.

    If the `prefix` is set to True, it appends "source_" and "target_"
    prefixes to labels in the "source" and "target" columns, respectively.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape `n_obs` * `n_vars`.
        Rows correspond to cells and columns to genes.
    source_annotation : str
        The name of the source annotation column in the `adata` object.
    target_annotation : str
        The name of the target annotation column in the `adata` object.
    prefix : bool, optional
        If True, appends "source_" and "target_" prefixes to the
        "source" and "target" columns, respectively.

    Returns
    -------
    relationships : pandas.DataFrame
        A DataFrame with the source and target categories,
        their counts and their percentages.
    """

    check_annotation(
        adata,
        [
            source_annotation,
            target_annotation
        ]
    )

    # Iterate through annotation columns and calculate label relationships
    logging.info((f"Source: {source_annotation}"))
    logging.info((f"Target: {target_annotation}"))

    if source_annotation == source_annotation:
        logging.info("Source and Target are the same")
        target_annotation_copy = f"{target_annotation}_copy"
        adata.obs[target_annotation_copy] = adata.obs[target_annotation]
        target_annotation = target_annotation_copy

    # Calculate label relationships between source and target columns
    relationships = adata.obs.groupby(
        [source_annotation, target_annotation]
        ).size().reset_index(name='count')

    adata.obs.drop(columns=[target_annotation_copy], inplace=True)

    # Calculate the total count for each source
    total_counts = (
        relationships.groupby(source_annotation)['count'].transform('sum')
    )
    # Calculate the percentage of the total count for each source
    relationships['percentage_source'] = (
        (relationships['count'] / total_counts * 100).round(1)
    )

    total_counts_target = (
        relationships.groupby(target_annotation)['count'].transform('sum')
    )

    # Calculate the percentage of the total count for each target
    relationships['percentage_target'] = (
        (relationships['count'] / total_counts_target * 100).round(1)
    )

    relationships.rename(
        columns={
            source_annotation: "source",
            target_annotation: "target"
        },
        inplace=True
    )

    relationships["source"] = relationships["source"].astype(str)
    relationships["target"] = relationships["target"].astype(str)
    relationships["count"] = relationships["count"].astype('int64')
    relationships[
        "percentage_source"
        ] = relationships["percentage_source"].astype(float)
    relationships[
        "percentage_target"
        ] = relationships["percentage_target"].astype(float)

    # Reset the index of the label_relations DataFrame
    relationships.reset_index(drop=True, inplace=True)
    if prefix:
        # Add "Source_" prefix to the "Source" column
        relationships["source"] = relationships[
            "source"
        ].apply(lambda x: "source_" + x)

        # Add "Target_" prefix to the "Target" column
        relationships["target"] = relationships[
            "target"
        ].apply(lambda x: "target_" + x)

    return relationships


def color_mapping(
        labels,
        color_map='viridis',
        opacity=1.0
):
    """
    Map a list of labels to colors using a specified
    matplotlib colormap and opacity.

    This function takes a list of labels and maps each one to a color from the
    specified colormap. If the colormap is continuous, it linearly interpolates
    between the colors. For discrete colormap, it calculates the number of
    categories per color and interpolates between the colors.

    For more information on colormaps, see:
    https://matplotlib.org/stable/users/explain/colors/colormaps.html

    Parameters
    ----------
    labels : list
        The list of labels to map to colors.
    color_map : str, optional
        The name of the colormap to use. Default is 'viridis'.
    opacity : float, optional
        The opacity of the colors. Must be between 0 and 1. Default is 1.0.

    Returns
    -------
    label_colors : list[str]
        A list of strings, each representing an rgba color in CSS format.
        The opacity of each color is set to the provided `opacity` value.

    Raises
    ------
    ValueError
        If the opacity is not between 0 and 1,
        or if the colormap name is invalid.
    """

    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between 0 and 1")

    try:
        cmap = cm.get_cmap(color_map)
    except ValueError:
        raise ValueError(f"Invalid color map name: {color_map}")

    if cmap.N > 50:  # This is a continuous colormap
        label_colors = [
            cmap(i / (len(labels) - 1)) for i in range(len(labels))
        ]
    else:  # This is a discrete colormap
        # Calculate the number of categories per color
        categories_per_color = np.ceil(len(labels) / cmap.N)

        # Interpolate between the colors

        label_colors = [
            cmap(i / (categories_per_color * cmap.N - 1))
            for i in range(len(labels))
        ]

    label_colors = [
        f'rgba({int(color[0]*255)},'
        f'{int(color[1]*255)},'
        f'{int(color[2]*255)},{opacity})'
        for color in label_colors
    ]

    return label_colors
