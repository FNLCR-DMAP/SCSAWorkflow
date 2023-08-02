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
        features=None,
        new_layers=None,
        new_obs=None,
        new_features=None):
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

    new_layers : str or lust of str, optional
        The layer name to check if exists in adata.layers.keys().

    new_obs : str or list of str, optional
        The observation name to check if exists in adata.obs.

    new_features : str or list of str, optional
        The feature(s) to check if exists in adata.layers.keys().

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

    layer_list = list(adata.layers.keys())
    existing_layer_str = "\n".join(layer_list)

    # Check for specified layers existence
    if layers is not None:
        if isinstance(layers, str):
            layers = [layers]
        elif not isinstance(layers, list):
            raise ValueError("The 'layers' parameter should be \
                             a string or a list of strings.")
        for layer in layers:
            if layer not in layer_list:
                raise ValueError(
                    f"The table '{layer}' "
                    "does not exist in the provided dataset.\n"
                    "Existing tables are:\n"
                    f"{existing_layer_str}"
                )

    if new_layers is not None:
        if isinstance(new_layers, str):
            new_layers = [new_layers]
        elif not isinstance(new_layers, list):
            raise ValueError("The 'new_layers' parameter should be \
                             a string or a list of strings.")
        for layer in new_layers:
            if layer in layer_list:
                raise ValueError(
                    f"The new table '{layer}' "
                    "exist in the provided dataset.\n"
                    "Existing tables are:\n"
                    f"{existing_layer_str}"
                )

    # Check for specified observations existence

    existing_obs = adata.obs.columns.to_list()
    existing_obs_str = "\n".join(existing_obs)

    if obs is not None:
        if isinstance(obs, str):
            obs = [obs]
        elif not isinstance(obs, list):
            raise ValueError("The 'obs' parameter should be \
                             a string or a list of strings.")
        for observation in obs:
            if observation not in existing_obs:
                raise ValueError(
                    f"The observation '{observation}' "
                    "does not exist in the provided dataset.\n"
                    "Existing observations are:\n"
                    f"{existing_obs_str}"
                )

    if new_obs is not None:
        if isinstance(new_obs, str):
            new_obs = [new_obs]
        elif not isinstance(new_obs, list):
            raise ValueError("The 'new_obs' parameter should be \
                             a string or a list of strings.")

        for observation in new_obs:
            if observation in existing_obs:
                raise ValueError(
                    f"The new observation '{observation}' "
                    "exists in the provided dataset.\n"
                    "Existing observations are:\n"
                    f"{existing_obs_str}"
                )

    # Check for specified features existence

    var_name_list = adata.var_names.to_list()
    existing_var_str = "\n".join(var_name_list)

    if features is not None:
        if isinstance(features, str):
            features = [features]
        elif not isinstance(features, list):
            raise ValueError("The 'features' parameter should be a \
                             string or a list of strings.")

        for feature in features:
            if feature not in var_name_list:
                raise ValueError(
                    f"The feature '{feature}' "
                    "does not exist in the provided dataset.\n"
                    "Existing features are:\n"
                    f"{existing_var_str}"
                )

    if new_features is not None:
        if isinstance(new_features, str):
            new_features = [new_features]
        elif not isinstance(new_features, list):
            raise ValueError("The 'new_features' parameter should be a \
                             string or a list of strings.")

        for feature in new_features:
            if feature in var_name_list:
                raise ValueError(
                    f"The new feature '{feature}' "
                    "exists in the provided dataset.\n"
                    "Existing features are:\n"
                    f"{existing_var_str}"
                )
