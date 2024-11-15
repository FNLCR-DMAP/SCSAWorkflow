
import pandas as pd


def is_binary_0_1(column):
    """
    Check if a pandas Series contains only binary values (0 and 1).

    Parameters
    ----------
    column : pandas.Series
        The pandas Series to check.

    Returns
    -------
    bool
        True if the Series contains only 0 and 1, False otherwise.

    Notes
    -----
    The function considers a Series to be binary if it contains exactly
    the values 0 and 1, and no other values.
    """
    unique_values = set(column.unique())
    return unique_values == {0, 1}


def decode_phenotype(data, phenotype_code, **kwargs):
    """
    Convert a phenotype code into a dictionary mapping
    feature (marker) names to values for that marker's
    classification as '+' or '-'.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the columns that will be used to decode
        the phenotype.
    phenotype_code : str
        The phenotype code string, which should end with '+' or '-'.
    **kwargs : keyword arguments
        Optional keyword arguments to specify prefix and suffix to be added
        to the column names.
        - prefix : str, optional
            Prefix to be added to the column names for the
            feature classification. Default is ''.
        - suffix : str, optional
            Suffix to be added to the column names for the
            feature classification. Default is ''.

    Returns
    -------
    dict
        A dictionary where the keys are column names and the values are the
        corresponding phenotype classification.

    Raises
    ------
    ValueError
        If the phenotype code does not end with '+' or '-' or if any columns
        specified in the phenotype code do not exist in the DataFrame.

    Notes
    -----
    The function splits the phenotype code on '+' and '-' characters to
    determine the phenotype columns and values. It checks if the columns
    exist in the DataFrame and whether they are binary or string types to
    properly map values.
    """

    import re
    # The phenotype code should end with '+' or '-'
    if not (phenotype_code.endswith('+') or phenotype_code.endswith('-')):
        raise ValueError(
            (
                f'The passed phenotype code "{phenotype_code}"'
                ' should end with "+" or "-"'
            )
        )

    # Split the phenotype definition on '+' and '-' characters
    phenotypes = re.split(r'\+|-', phenotype_code)
    phenotypes.remove('')

    prefix = kwargs.get("prefix", '')
    suffix = kwargs.get("suffix", '')
    phenotypes_columns = [f"{prefix}{name}{suffix}" for name in phenotypes]

    existing_columns = data.columns
    for value in phenotypes_columns:
        if value not in existing_columns:
            raise ValueError(
                (
                    f'The feature "{value}" does not exist in the input table.'
                    f' Existing columns are "{existing_columns.tolist()}"'
                )
            )

    phenotype_values = re.findall(r'[A-Za-z0-9]+[+-]', phenotype_code)

    phenotype_dict = {}
    for value, column in zip(phenotype_values, phenotypes_columns):
        if pd.api.types.is_string_dtype(data[column]):
            phenotype_dict[column] = value
        if is_binary_0_1(data[column]):
            if value.endswith('+'):
                phenotype_dict[column] = 1
            elif value.endswith('-'):
                phenotype_dict[column] = 0

    return phenotype_dict


def generate_phenotypes_dict(data_df, phenotypes_df, prefix='', suffix=''):
    """
    Generate a dictionary of phenotype names to their corresponding
    decoding rules.

    Parameters
    ----------
    data_df : pandas.DataFrame
        The DataFrame containing the columns that will be used to decode
        the phenotypes.
    phenotypes_df : pandas.DataFrame
        A DataFrame containing phenotype definitions with columns:
        - "phenotype_name" : str
            The name of the phenotype.
        - "phenotype_code" : str
            The code used to decode the phenotype.
    prefix : str, optional
        Prefix to be added to the column names. Default is ''.
    suffix : str, optional
        Suffix to be added to the column names. Default is ''.

    Returns
    -------
    dict
        A dictionary where the keys are phenotype names and the values are
        dictionaries mapping column names to values.

    Notes
    -----
    The function iterates over each row in the `phenotypes_df` DataFrame and
    decodes the phenotype using the `decode_phenotype` function.
    """
    all_phenotypes = {}
    for index, row in phenotypes_df.iterrows():
        phenotype_name = row["phenotype_name"]
        phenotype_code = row["phenotype_code"]
        all_phenotypes[phenotype_name] = decode_phenotype(
            data_df,
            phenotype_code,
            prefix=prefix,
            suffix=suffix
        )
    return all_phenotypes


def apply_phenotypes(data_df, phenotypes_dic):
    """
    Add binary columns to the DataFrame indicating if each cell matches a
    phenotype.

    Parameters
    ----------
    data_df : pandas.DataFrame
        The DataFrame to which binary phenotype columns will be added.
    phenotypes_dic : dict
        A dictionary where the keys are phenotype names and the values are
        dictionaries mapping column names to values.

    Returns
    -------
    dict
        A dictionary where the keys are phenotype names and the values are
        the counts of rows that match each phenotype.

    Notes
    -----
    The function creates binary columns in the DataFrame for each phenotype
    and counts the number of rows matching each phenotype.
    """
    return_dic = {}
    for phenotype, rule in phenotypes_dic.items():
        matching_rows = (
            data_df[list(rule.keys())]
            .eq(list(rule.values()))
            .all(axis=1)
        )
        count_matching_rows = matching_rows.sum()

        return_dic[phenotype] = count_matching_rows

        data_df.loc[matching_rows, phenotype] = 1
        data_df.loc[~matching_rows, phenotype] = 0

    return return_dic


def combine_phenotypes(data_df, phenotype_columns, multiple=True):
    """
    Combine multiple binary phenotype columns into a new column in a vectorized manner.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame containing the phenotype columns.
    phenotype_columns : list of str
        List of binary phenotype column names.
    multiple : bool, optional
        Whether to concatenate the names of multiple positive phenotypes.
        If False, all multiple positive phenotypes are labeled as
        "no_label". Default is True.

    Returns
    -------
    pandas.Series
        A Series representing the combined phenotype for each row.
    """
    # Create a mask for each phenotype column where values are 1 (positive)
    phenotype_masks = data_df[phenotype_columns].astype(bool)

    # Create a series of phenotype names with a comma and space
    # after each name. That series will be used to join names
    # of positive phenotypes in the vectorized operation below.
    phenotypes_series = pd.Index(phenotype_columns) + ", "

    # For each row, join the names of positive phenotypes by doing
    # a dot product between the mask and the series of phenotype names.
    combined_phenotypes = \
        phenotype_masks.dot(phenotypes_series).str.rstrip(", ")

    # Set all with 0 positive phenotypes to "no_label"
    counts_positive = phenotype_masks.sum(axis=1)
    combined_phenotypes[counts_positive == 0] = "no_label"

    # Handle the case when multiple is False:
    if not multiple:

        # set all with >1 positive phenotypes to "no_label"
        combined_phenotypes[counts_positive > 1] = "no_label"

    return combined_phenotypes


def assign_manual_phenotypes(
        data_df,
        phenotypes_df,
        annotation="manual_phenotype",
        prefix='',
        suffix='',
        multiple=True,
        drop_binary_code=True):
    """
    Assign manual phenotypes to the DataFrame and generate summaries.

    Parameters
    ----------
    data_df : pandas.DataFrame
        The DataFrame to which manual phenotypes will be assigned.
    phenotypes_df : pandas.DataFrame
        A DataFrame containing phenotype definitions with columns:
        - "phenotype_name" : str
            The name of the phenotype.
        - "phenotype_code" : str
            The code used to decode the phenotype.
    annotation : str, optional
        The name of the column to store the combined phenotype. Default is
        "manual_phenotype".
    prefix : str, optional
        Prefix to be added to the column names. Default is ''.
    suffix : str, optional
        Suffix to be added to the column names. Default is ''.
    multiple : bool, optional
        Whether to concatenate the names of multiple positive phenotypes.
        Default is True.
    drop_binary_code : bool, optional
        Whether to drop the binary phenotype columns. Default is True.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - "phenotypes_counts": dict
            Counts of cells matching each defined phenotype.
        - "assigned_phenotype_counts": dict
            Counts of cells matching different numbers of phenotypes.
        - "multiple_phenotypes_summary": pandas.DataFrame
            Summary of cells with multiple phenotypes.

    Notes
    -----
    The function generates a combined phenotype column, prints summaries of
    cells matching multiple phenotypes, and returns a dictionary with
    detailed counts and summaries.



    Examples
    --------
    Suppose `data_df` is a DataFrame with binary phenotype columns and 
    `phenotypes_df` contains the following definitions:

    >>> data_df = pd.DataFrame({
    ...     'cd4_phenotype': [0, 1, 0, 1],
    ...     'cd8_phenotype': [0, 0, 1, 1]
    ... })
    >>> phenotypes_df = pd.DataFrame([
    ...     {"phenotype_name": "cd4_cells", "phenotype_code": "cd4+"},
    ...     {"phenotype_name": "cd8_cells", "phenotype_code": "cd8+"},
    ...     {"phenotype_name": "cd4_cd8", "phenotype_code": "cd4+cd8+"}
    ... ])
    >>> result = assign_manual_phenotypes(
    ...     data_df,
    ...     phenotypes_df,
    ...     annotation="manual",
    ...     prefix='',
    ...     suffix='_phenotype',
    ...     multiple=True
    ... )

    The `data_df` DataFrame will be edited in place to include a new column
    `"manual"` with the combined phenotype labels:

    >>> print(data_df)
       cd4_phenotype  cd8_phenotype manual
    0              0              0 no_label
    1              1              0 cd4_cells
    2              0              1 cd8_cells
    3              1              1 cd8_cells, cd4_cd8

    The result dictionary contains counts and summaries as follows:

    >>> print(result["phenotypes_counts"])
    {'cd4_cells': 1, 'cd8_cells': 2, 'cd4_cd8': 1}

    >>> print(result["assigned_phenotype_counts"])
    0    1
    1    2
    2    1
    Name: num_phenotypes, dtype: int64

    >>> print(result["multiple_phenotypes_summary"])
                   manual  count
    0  cd8_cells, cd4_cd8      1
    """

    phenotypes_dic = generate_phenotypes_dict(
        data_df,
        phenotypes_df,
        prefix, suffix
    )

    phenotypes_counts = apply_phenotypes(data_df, phenotypes_dic)

    print("\n#####################################\n")
    # Print the counts of cells in every phenotype
    for phenotype, count in phenotypes_counts.items():
        print(f"{phenotype}: {count} cell(s)")

    phenotypes_columns = phenotypes_dic.keys()

    data_df[annotation] = combine_phenotypes(
        data_df,
        phenotypes_columns,
        multiple)

    number_phenotypes_columns = "num_phenotypes"
    data_df[number_phenotypes_columns] = (
        data_df[phenotypes_columns].sum(axis=1).astype(int)
    )
    summary = data_df[number_phenotypes_columns].value_counts().sort_index()

    print("\n#####################################\n")
    print("Summary of cells that matched multiple phenotypes:")
    for num_phenotypes, count in summary.items():
        print(
            f"Cells that matched {num_phenotypes}"
            f" phenotype(s): {count} cells"
        )

    print("\n#####################################\n")
    print(f'Summary of cells with multiple phenotypes in "{annotation}"\n\n')
    multiple_phenotypes = (
        data_df[data_df[number_phenotypes_columns] > 1]
        .groupby(annotation)
        .size()
        .sort_values(ascending=False)
    )

    multiple_phenotypes_df = multiple_phenotypes.reset_index(name='count')
    print(multiple_phenotypes_df.to_string(index=False))

    return_dic = {}
    return_dic["assigned_phenotype_counts"] = summary
    return_dic["multiple_phenotypes_summary"] = multiple_phenotypes_df
    return_dic["phenotypes_counts"] = phenotypes_counts

    if drop_binary_code is True:
        # Remove the columns defined by the keys of the dictionary phenotypes_counts
        data_df.drop(columns=phenotypes_counts.keys(), inplace=True)
    return return_dic
