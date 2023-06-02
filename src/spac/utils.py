import re


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
