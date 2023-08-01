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


def anndata_checks(
        adata,
        layers=None,
        obs=None,
        features=None):
    """
    Perform common error checks for anndata related objects.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be checked.

    layers : str or list of str, optional
        The layer(s) to check for existence in adata.layers.keys().

    obs : str or list of str, optional
        The observation(s) to check for existence in adata.obs.

    features : str or list of str, optional
        The feature(s) to check for existence in adata.var_names.

    Raises
    ------
    TypeError
        If adata is not an instance of anndata.AnnData.

    ValueError
        If any of the specified layers, observations, or features do not exist.

    Example
    -------
    >>> import anndata as ad
    >>> adata = ad.AnnData(...)
    >>> anndata_checks(
        adata,
        layers=['layer1', 'layer2'],
        obs=['obs1', 'obs2'],
        features=['feature1', 'feature2'])
    """

    # Check if adata is an instance of anndata.AnnData
    if not isinstance(adata, ad.AnnData):
        raise TypeError(
            "Input dataset should be "
            "an instance of anndata.AnnData, "
            "please check the input dataset source."
            )

    # Check for specified layers existence
    if layers is not None:
        if isinstance(layers, str):
            layers = [layers]
        elif not isinstance(layers, list):
            raise ValueError("The 'layers' parameter should be \
                             a string or a list of strings.")
        layer_list = list(adata.layers.keys())
        for layer in layers:
            if layer not in layer_list:
                existing_layer_str = "\n".join(layer_list)
                raise ValueError(
                    f"The table '{layer}' "
                    "does not exist in the provided dataset.\n"
                    "Existing tables are:\n"
                    f"{existing_layer_str}"
                )

    # Check for specified observations existence
    if obs is not None:
        if isinstance(obs, str):
            obs = [obs]
        elif not isinstance(obs, list):
            raise ValueError("The 'obs' parameter should be \
                             a string or a list of strings.")
        existing_obs = adata.obs.columns.to_list()
        for observation in obs:
            if observation not in existing_obs:
                existing_obs_str = "\n".join(existing_obs)
                raise ValueError(
                    f"The observation '{observation}' "
                    "does not exist in the provided dataset.\n"
                    "Existing observations are:\n"
                    f"{existing_obs_str}"
                )

    # Check for specified features existence
    if features is not None:
        if isinstance(features, str):
            features = [features]
        elif not isinstance(features, list):
            raise ValueError("The 'features' parameter should be a \
                             string or a list of strings.")
        var_name_list = adata.var_names.to_list()
        for feature in features:
            if feature not in var_name_list:
                existing_var_str = "\n".join(var_name_list)
                raise ValueError(
                    f"The feature '{feature}' "
                    "does not exist in the provided dataset.\n"
                    "Existing features are:\n"
                    f"{existing_var_str}"
                )
