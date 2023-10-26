import re
import anndata as ad


def regex_search_list(
        regex_patterns,
        list_to_search):
    """
    Perfrom regex (regular expression) search in a list and
    return list of strings matching the regex pattern

    Parameters:
    -----------
        regex_pattern : str or a list of str
            The regex pattern to look for,
            single pattern or a list of patterns.

        list_to_search : list of str
            A list of string to seach for string that matches regex pattern.

    Returns:
    --------
        list of str
            A list of strings containing results from search.

    Example:
    --------
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
        found = re.search(pattern, str)
        if found is None:
            pass
        else:
            return found.group(0)

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
        need_exist=True
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

    Raises
    ------
    ValueError
        If the `input` is not a string or a list of strings.
        If `need_exist` is True and any element of the input
            list does not exist in the target list.
        If `need_exist` is False and any element of the input
            list exists in the target list.
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
                    raise ValueError(
                        f"The {input_type} '{item}' "
                        "does not exist in the provided dataset.\n"
                        f"Existing {input_type}s are:\n"
                        f"{target_list_str}"
                    )
        else:
            for item in input:
                if item in target_list:
                    raise ValueError(
                        f"The {input_type} '{item}' "
                        "exist in the provided dataset.\n"
                        f"Existing {input_type}s are:\n"
                        f"{target_list_str}"
                    )


def check_table(
        adata,
        tables=None,
        should_exist=True):

    """
    Perform common error checks for table (layers) in
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
    existing_tables = list(adata.layers.keys())
    check_list_in_list(
            input=tables,
            input_name="tables",
            input_type="table",
            target_list=existing_tables,
            need_exist=should_exist
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
