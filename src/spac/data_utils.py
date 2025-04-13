import re
import os
import pandas as pd
import anndata as ad
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import logging
from collections import defaultdict
from spac.utils import regex_search_list, check_list_in_list, check_annotation
from anndata import AnnData


def append_annotation(
    data: pd.DataFrame,
    annotation: dict
) -> pd.DataFrame:
    """
    Append a new annotation with single value to
    a Pandas DataFrame based on mapping rules.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to which the new observation will be appended.

    annotation : dict
        dictionary of string pairs representing
        the new annotation and its value.
        Each pair should have this format:
        <new annotation column name>:<value of the annotation>
        The values must be a single string or numeric value.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new observation appended.
    """

    if not isinstance(annotation, dict):
        error_msg = "Annotation must be provided as a dictionary."
        raise ValueError(error_msg)

    for new_column, value in annotation.items():
        if not isinstance(new_column, str):
            error_msg = f"The key {new_column} is not " + \
                "a single string, please check."
            raise ValueError(error_msg)

        if not isinstance(value, (str, int, float)):
            error_msg = f"The value {value} in {new_column} is not " + \
                "a single string or numeric value, please check."
            raise ValueError(error_msg)

        if new_column in data.columns:
            error_msg = f"'{new_column}' already exists in the DataFrame."
            raise ValueError(error_msg)

        data[new_column] = value

    return data


def ingest_cells(dataframe,
                 regex_str,
                 x_col=None,
                 y_col=None,
                 annotation=None):

    """
    Read the csv file into an anndata object.

    The function will also intialize features and spatial coordiantes.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data frame that contains cells as rows, and cells informations as
        columns.

    regex_str : str or list of str
        A string or a list of strings representing python regular expression
        for the features columns in the data frame.  x_col : str The column
        name for the x coordinate of the cell.

    y_col : str
        The column name for the y coordinate of the cell.

    annotation : str or list of str
        The column name for the region that the cells. If a list is passed,
        multiple annotations will be created in the returned AnnData object.


    Returns
    -------
    anndata.AnnData
        The generated AnnData object
    """

    if not isinstance(regex_str, list):
        regex_list = [regex_str]
    else:
        regex_list = regex_str

    all_columns = list(dataframe.columns)
    all_features = []

    for column in regex_list:
        current_features = regex_search_list(
            [column],
            all_columns
        )

        if len(current_features) == 0:
            error_message = "Provided regex pattern(s) or feature(s):\n" + \
                f'"{column}"\n' + \
                "does not match any in the dataset, please review the input."
            raise ValueError(error_message)

        all_features.extend(current_features)

    features_df = dataframe[all_features]
    adata = ad.AnnData(
        features_df,
        dtype=features_df[all_features[0]].dtype)

    if annotation is not None:
        if isinstance(annotation, str):
            list_of_annotation = [annotation]
        else:
            list_of_annotation = annotation

        for annotation in list_of_annotation:

            # As selecting one column of the dataframe returns a series which
            # AnnData converts to NaN, then I convert it to a list before
            # assignment.
            adata.obs[annotation] = dataframe[annotation].tolist()

    if x_col is not None and y_col is not None:
        numpy_array = dataframe[[x_col, y_col]].to_numpy().astype('float32')
        adata.obsm["spatial"] = numpy_array
    return adata


def concatinate_regions(regions):
    """
    Concatinate data from multiple regions and create new indexes.

    Parameters
    ----------
    regions : list of anndata.AnnData
        AnnData objects to be concatinated.

    Returns
    -------
    anndata.AnnData
        New AnddData object with the concatinated values in AnnData.X

    """
    all_adata = ad.concat(regions)
    all_adata.obs_names_make_unique()
    return all_adata


def rescale_features(features, min_quantile=0.01, max_quantile=0.99):
    """
    Clip and rescale features outside the minimum and maximum quantile.

    The rescaled features will be between 0 and 1.

    Parameters
    ----------
    features : pandas.Dataframe
        The DataRrame of features.

    min_quantile : float
        The minimum quantile to be consider zero.

    max_quantile: float
        The maximum quantile to be considerd 1.

    Returns
    -------
    pandas.DataFrame
        The created DataFrame with normalized features.
    """
    markers_max_quantile = features.quantile(max_quantile)
    markers_min_quantile = features.quantile(min_quantile)

    features_clipped = features.clip(
        markers_min_quantile,
        markers_max_quantile,
        axis=1)

    scaler = MinMaxScaler()
    np_features_scaled = scaler.fit_transform(
        features_clipped.to_numpy())

    features_scaled = pd.DataFrame(
        np_features_scaled,
        columns=features_clipped.columns)

    return features_scaled


def add_rescaled_features(adata, min_quantile, max_quantile, layer):
    """
    Clip and rescale the features matrix.

    The results will be added into a new layer in the AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    min_quantile : float
        The minimum quantile to rescale to zero.

    max_quantile : float
        The maximum quantile to rescale to one.

    layer : str
        The name of the new layer to add to the anndata object.
    """

    original = adata.to_df()
    rescaled = rescale_features(original, min_quantile, max_quantile)
    adata.layers[layer] = rescaled


def subtract_min_per_region(adata, annotation, layer, min_quantile=0.01):
    """
    Substract the minimum quantile of every marker per region.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    annotation: str
        The name of the annotation in `adata` to define batches.

    min_quantile : float
        The minimum quantile to rescale to zero.

    layer : str
        The name of the new layer to add to the AnnData object.
    """
    regions = adata.obs[annotation].unique().tolist()
    original = adata.to_df()

    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs[annotation] == region]
        new_features = subtract_min_quantile(region_cells, min_quantile)
        new_df_list.append(new_features)

    new_df = pd.concat(new_df_list)
    adata.layers[layer] = new_df


def subtract_min_quantile(features, min_quantile=.01):
    """
    Subtract the features defined by the minimum quantile from all columns.

    Parameters
    ----------

    features : pandas.DataFrame
        The dataframe of features.

    min_quantile: float
        The minimum quantile to be consider zero.

    Returns
    -------
    pandas.DataFrame
        dataframe with rescaled features.
    """
    columns_min_quantile = features.quantile(min_quantile)

    subtracted_min = features - columns_min_quantile

    # Clip negative values to zero
    subtracted_min.clip(lower=0, axis=1, inplace=True)

    return subtracted_min


def load_csv_files(file_names):

    """
    Read the csv file(s) into a pandas dataframe.

    Parameters
    ----------
    file_names : str or list
        A list of csv file paths to be
        combined into single list of dataframe output

    Returns
    -------
    pandas.dataframe
        A pandas dataframe of all the csv files. The returned dataset
        will have an extra column called "loaded_file_name" containing
        source file name.
    """

    # meta_schema = []
    dataframe_list = []
    dataframe_name = []

    if not isinstance(file_names, list):
        if not isinstance(file_names, str):
            file_name_type = type(file_names)
            error_message = "file_names should be list or string" + \
                            ", but got " + str(file_name_type) + "."
            raise TypeError(error_message)
        else:
            file_names = [file_names]

    for file_name in file_names:

        # Try to load the csv into pandas DataFrame.
        # Check if the file exists
        if not os.path.exists(file_name):
            error_message = f"The file '{file_name}' does not exist."
            raise FileNotFoundError(error_message)

        # Check if the file is readable
        if not os.access(file_name, os.R_OK):
            error_message = "The file " + file_name + \
                    " cannot be read due to insufficient permissions."
            raise PermissionError(error_message)

        try:
            current_df = pd.read_csv(file_name)
        except pd.errors.EmptyDataError:
            error_message = "The file is empty or does not contain any data."
            raise TypeError(error_message)
        except pd.errors.ParserError:
            error_message = "The file could not be parsed. " + \
                            "Please check that the file is a valid CSV."
            raise TypeError(error_message)

        current_df["loaded_file_name"] = file_name

        # current_schema = current_df.columns.to_list()

        # if len(meta_schema) == 0:
        #     meta_schema = current_schema
        #     print("Meta schema acquired. Columns are:")
        #     for column_name in meta_schema:
        #         print(column_name)

        # if len(meta_schema) == len(current_schema):
        #     if set(meta_schema) != set(current_schema):
        #         error_message = "Column in current file does not match " + \
        #                 f"the meta_schema, got:\n {current_schema}. "
        #         raise ValueError(error_message)
        # else:
        #     error_message = "Column in current file does not match " + \
        #                 f"the meta_schema, got:\n {current_schema}. "
        #     raise ValueError(error_message)

        dataframe_list.append(current_df)
        dataframe_name.append(file_name)

    logging.info("CSVs are converted into dataframes and combined"
                 " into a list!")
    logging.info("Total of " + str(len(dataframe_list)) +
                 " dataframes in the list.")
    for i, each_file in enumerate(dataframe_list):
        logging.info(f"File name: {dataframe_name[i]}")
        logging.info("Info: ")
        logging.info(each_file.info())
        logging.info("Description: ")
        logging.info(each_file.describe())
        logging.info("\n")

    logging.info("Combining Dataframes into Single Dataframe...")
    combined_dataframe = combine_dfs(dataframe_list)

    return combined_dataframe


def select_values(data, annotation, values=None, exclude_values=None):
    """
    Selects values from either a pandas DataFrame or an AnnData object based
    on the annotation and values.

    Parameters
    ----------
    data : pandas.DataFrame or anndata.AnnData
        The input data. Can be a DataFrame for tabular data or an AnnData
        object.
    annotation : str
        The column name in a DataFrame or the annotation key in an AnnData
        object to be used for selection.
    values : str or list of str
        List of values for the annotation to include. If None, all values are
        considered for selection.
    exclude_values : str or list of str
        List of values for the annotation to exclude. Can't be combined with
        values.

    Returns
    -------
    pandas.DataFrame or anndata.AnnData
        The filtered DataFrame or AnnData object containing only the selected
        rows based on the annotation and values.
    """

    # Make sure that either values or exclude_values is set, but not both
    if values is not None and exclude_values is not None:
        error_msg = "Only use with values to include or exclude, but not both."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # If values and exclude_values are both None, return the original data
    if values is None and exclude_values is None:
        print("No values or exclude_values provided. Returning original data.")
        return data

    # Ensure values are in a list format if not None
    if values is not None and not isinstance(values, list):
        values = [values]

    # Ensure exclude_values are in a list format if not None
    if exclude_values is not None and not isinstance(exclude_values, list):
        exclude_values = [exclude_values]

    if isinstance(data, pd.DataFrame):
        return _select_values_dataframe(
            data,
            annotation,
            values,
            exclude_values)
    elif isinstance(data, ad.AnnData):
        return _select_values_anndata(data, annotation, values, exclude_values)
    else:
        error_msg = (
            "Unsupported data type. Data must be either a pandas DataFrame"
            " or an AnnData object."
        )
        logging.error(error_msg)
        raise TypeError(error_msg)


def _select_values_dataframe(data, annotation, values, exclude_values):
    possible_annotations = data.columns.tolist()

    # Check if the annotation exists using check_list_in_list
    check_list_in_list(
        input=[annotation],
        input_name="annotation",
        input_type="column name/annotation key",
        target_list=possible_annotations,
        need_exist=True
    )

    # Validate provided values against unique ones, if not None
    unique_values = data[annotation].astype(str).unique().tolist()
    check_list_in_list(
        values,
        "values to include",
        "label",
        unique_values,
        need_exist=True
    )
    check_list_in_list(
        exclude_values,
        "values to exclude",
        "label",
        unique_values,
        need_exist=True
    )

    # Proceed with filtering based on values or exclude_values
    if values is not None:
        filtered_data = data[data[annotation].isin(values)]
    elif exclude_values is not None:
        filtered_data = data[~data[annotation].isin(exclude_values)]

    count = filtered_data.shape[0]
    logging.info(
        f"Summary of returned dataset: {count} cells",
        " match the selected labels."
        )

    return filtered_data


def _select_values_anndata(data, annotation, values, exclude_values):
    possible_annotations = data.obs.columns.tolist()

    # Check if the annotation exists using check_list_in_list
    check_list_in_list(
        input=[annotation],
        input_name="annotation",
        input_type="column name/annotation key",
        target_list=possible_annotations,
        need_exist=True
    )

    # Validate provided values against unique ones, if not None
    unique_values = data.obs[annotation].astype(str).unique().tolist()
    check_list_in_list(
        values,
        "values to include",
        "label",
        unique_values,
        need_exist=True
    )
    check_list_in_list(
        exclude_values,
        "values to exclude",
        "label",
        unique_values,
        need_exist=True
    )

    # Proceed with filtering based on values or exclude_values
    if values is not None:
        filtered_data = data[data.obs[annotation].isin(values)].copy()
    elif exclude_values is not None:
        filtered_data = data[~data.obs[annotation].isin(exclude_values)].copy()

    count = filtered_data.n_obs
    logging.info(
        f"Summary of returned dataset: {count}"
        " cells match the selected labels."
        )

    return filtered_data


def downsample_cells(input_data, annotations, n_samples=None, stratify=False,
                     rand=False, combined_col_name='_combined_',
                     min_threshold=5):
    """
    Custom downsampling of data based on one or more annotations.

    This function offers two primary modes of operation:
    1. **Grouping (stratify=False)**:
       - For a single annotation: The data is grouped by unique values of the
         annotation, and 'n_samples' rows are selected from each group.
       - For multiple annotations: The data is grouped based on unique
         combinations of the annotations, and 'n_samples' rows are selected
         from each combined group.

    2. **Stratification (stratify=True)**:
       - Annotations (single or multiple) are combined into a new column.
       - Proportionate stratified sampling is performed based on the unique
         combinations in the new column, ensuring that the downsampled dataset
         maintains the proportionate representation of each combined group
         from the original dataset.

    Parameters
    ----------
    input_data : pd.DataFrame
        The input data frame.
    annotations : str or list of str
        The column name(s) to downsample on. If multiple column names are
        provided, their values are combined using an underscore as a separator.
    n_samples : int, default=None
        The number of samples to return. Behavior differs based on the
        'stratify' parameter:
        - stratify=False: Returns 'n_samples' for each unique value (or
          combination) of annotations.
        - stratify=True: Returns a total of 'n_samples' stratified by the
          frequency of every label or combined labels in the annotation(s).
    stratify : bool, default=False
        If true, perform proportionate stratified sampling based on the unique
        combinations of annotations. This ensures that the downsampled dataset
        maintains the proportionate representation of each combined group from
        the original dataset.
    rand : bool, default=False
        If true and stratify is True, randomly select the returned cells.
        Otherwise, choose the first n cells.
    combined_col_name : str, default='_combined_'
        Name of the column that will store combined values when multiple
        annotation columns are provided.
    min_threshold : int, default=5
        The minimum number of samples a combined group should have in the
        original dataset to be considered in the downsampled dataset. Groups
        with fewer samples than this threshold will be excluded from the
        stratification process. Adjusting this parameter determines the
        minimum presence a combined group should have in the original dataset
        to appear in the downsampled version.

    Returns
    -------
    output_data: pd.DataFrame
        The proportionately stratified downsampled data frame.

    Notes
    -----
    This function emphasizes proportionate stratified sampling, ensuring that
    the downsampled dataset is a representative subset of the original data
    with respect to the combined annotations. Due to this proportionate nature,
    not all unique combinations from the original dataset might be present in
    the downsampled dataset, especially if a particular combination has very
    few samples in the original dataset. The `min_threshold` parameter can be
    adjusted to determine the minimum number of samples a combined group
    should have in the original dataset to appear in the downsampled version.
    """

    logging.basicConfig(level=logging.WARNING)
    # Convert annotations to list if it's a string
    if isinstance(annotations, str):
        annotations = [annotations]

    # Check if the columns to downsample on exist
    missing_columns = [
        col for col in annotations if col not in input_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Columns {missing_columns} do not exist in the dataframe"
        )

    # If n_samples is None, return the input data without processing
    if n_samples is None:
        return input_data.copy()

    # Combine annotations into a single column if multiple annotations
    if len(annotations) > 1:
        input_data[combined_col_name] = input_data[annotations].apply(
            lambda row: '_'.join(row.values.astype(str)), axis=1)
        grouping_col = combined_col_name
    else:
        grouping_col = annotations[0]

    # Stratify selection
    if stratify:
        # Calculate proportions
        freqs = input_data[grouping_col].value_counts(normalize=True)

        # Exclude groups with fewer samples than the min_threshold
        filtered_freqs = freqs[freqs * len(input_data) >= min_threshold]

        # Log warning for groups that are excluded
        excluded_groups = freqs[~freqs.index.isin(filtered_freqs.index)]
        for group, count in excluded_groups.items():
            frequency = freqs.get(group, 0)
            logging.warning(
                f"Group '{group}' with count {count} "
                f"(frequency: {frequency:.4f}) "
                f"is excluded due to low frequency."
            )

        freqs = freqs[freqs.index.isin(filtered_freqs.index)]

        samples_per_group = (freqs * n_samples).round().astype(int)

        # Identify groups that have non-zero frequency
        # but zero samples after rounding
        zero_sample_groups = samples_per_group[samples_per_group == 0]
        groups_with_zero_samples = zero_sample_groups.index
        group_freqs = freqs[groups_with_zero_samples]
        original_counts = group_freqs * len(input_data)

        # Ensure each group has at least one sample if its frequency
        # is non-zero
        condition = samples_per_group == 0
        samples_per_group[condition] = freqs[condition].apply(
            lambda x: 1 if x > 0 else 0
        )

        # Log a warning for the adjusted groups
        if not original_counts.empty:
            group_count_pairs = [
                f"'{group}': {count}"
                for group, count in original_counts.items()
            ]
            summary = ', '.join(group_count_pairs)

            logging.warning(
                f"Groups adjusted to have at least one sample"
                f" due to non-zero frequency: {summary}."
            )

        # If have extra samples due to rounding, remove them from the
        # largest groups
        removed_samples = defaultdict(int)
        while samples_per_group.sum() > n_samples:
            max_group = samples_per_group.idxmax()
            samples_per_group[max_group] -= 1
            removed_samples[max_group] += 1

        # Log warning about the number of samples removed from each group
        for group, count in removed_samples.items():
            logging.warning(
                f"{count} sample(s) were removed from group '{group}'"
                f" due to rounding adjustments."
            )

        # Sample data
        sampled_data = []
        for group, group_data in input_data.groupby(grouping_col):
            sample_count = samples_per_group.get(group, 0)
            sample_size = min(sample_count, len(group_data))
            if rand:
                sampled_data.append(group_data.sample(sample_size))
            else:
                sampled_data.append(group_data.head(sample_size))

        # Concatenate all samples
        output_data = pd.concat(sampled_data)

    else:
        output_data = input_data.groupby(grouping_col, group_keys=False).apply(
            lambda x: x.head(min(n_samples, len(x)))
        ).reset_index(drop=True)

    # Log the final counts for each label in the downsampled dataset
    label_counts = output_data[grouping_col].value_counts()
    for label, count in label_counts.items():
        logging.info(f"Final count for label '{label}': {count}")

    # Log the total number of rows in the resulting data
    logging.info(f"Number of rows in the returned data: {len(output_data)}")

    return output_data


def calculate_centroid(
    data,
    x_min,
    x_max,
    y_min,
    y_max,
    new_x,
    new_y
):
    """
    Calculate the spatial coordinates of the cell centroid as the average of
    min and max coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        The input data frame. The dataframe should contain four columns for
        x_min, x_max, y_min, and y_max for centroid calculation.
    x_min : str
        column name with minimum x value
    x_max : str
        column name with maximum x value
    y_min : str
        column name with minimum y value
    y_max : str
        column name with maximum y value
    new_x : str
        the new column name of the x dimension of the cientroid,
        allowing characters are alphabetic, digits and underscore
    new_y : str
        the new column name of the y dimension of the centroid,
        allowing characters are alphabetic, digits and underscore

    Returns
    -------
    data : pd.DataFrame
        dataframe with two new centroid columns addded. Note that the
        dataframe is modified in place.

    """

    # Check for valid column names
    invalid_chars = r'[^a-zA-Z0-9_]'

    for name in [new_x, new_y]:
        if re.search(invalid_chars, name):
            error_string = "Column name " + str(name) + \
                " contains invalid characters. " + \
                "Use only alphanumeric characters and underscores."

            raise ValueError(error_string)

    # check if the columns exist in the dataframe
    for col in [x_min,
                x_max,
                y_min,
                y_max]:
        if col not in data.columns:
            raise ValueError(f"Column {col} does not exist in the dataframe.")

    # Calculate the centroids
    x_centroid = (data[x_min] + data[x_max]) / 2
    y_centroid = (data[y_min] + data[y_max]) / 2

    # Assign new centroid columns to the DataFrame in one operation
    data[[new_x, new_y]] = pd.concat(
        [x_centroid, y_centroid], axis=1, keys=[new_x, new_y]
    )

    # Return the modified DataFrame
    return data


def bin2cat(data, one_hot_annotations, new_annotation):
    """
    Combine a set of columns representing
    a binary one hot encoding of categories
    into a new categorical column.

    Parameters
    ----------
    data : pandas.DataFrame
        The pandas dataframe containing the one hot encoded annotations.

    one_hot_annotations : str or list of str
        A string or a list of strings representing
        python regular expression of the one hot encoded annotations
        columns in the data frame.

    new_annotation: str
        The column name for new categorical annotation to be created.

    Returns
    -------
    pandas.DataFrame
        DataFrame with new categorical column added.

    Example:
    --------
    >>> data = pd.DataFrame({
    ...    'A': [1, 1, 0, 0],
    ...     'B': [0, 0, 1, 0]
    ... })
    >>> one_hot_annotations = ['A', 'B']
    >>> new_annotation = 'new_category'
    >>> result = bin2cat(data, one_hot_annotations, new_annotation)
    >>> print(result[new_annotation])
    0      A
    1      A
    2      B
    3    NaN
    Name: new_category, dtype: object
    """

    if isinstance(one_hot_annotations, str):
        one_hot_annotations = [one_hot_annotations]
    elif not isinstance(one_hot_annotations, list):
        error_string = "one_hot_annotations should " + \
                         "be a string or a list of strings."
        raise ValueError(error_string)

    if new_annotation in data.columns:
        raise ValueError("Column name for new annotation already exists.")

    if len(one_hot_annotations) > 0:
        # Add regrex to find cell labels

        all_columns = list(data.columns)
        all_cell_labels = regex_search_list(
                one_hot_annotations,
                all_columns
            )

        if len(all_cell_labels) > 0:
            cell_labels_df = data.loc[:, all_cell_labels]

            def get_columns_with_1(row):
                column_names = cell_labels_df.columns[row == 1]
                if len(column_names) > 1:
                    raise ValueError(f"Multiple instance found:{column_names}")
                elif len(column_names) == 1:
                    return column_names[0]
                else:
                    return np.nan

            column_names_with_1 = cell_labels_df.apply(
                get_columns_with_1,
                axis=1)
            column_names_with_1 = column_names_with_1.tolist()
            data[new_annotation] = column_names_with_1
            return data
        else:
            error_string = "No column was found in the dataframe " + \
                "with current regrex pattern(s)."
            raise ValueError(error_string)


def combine_dfs(dataframes: list):
    """
    Combined multiple pandas dataframes into one.
    Schema of the first dataframe is considered primary.
    A warming will be printed if schema of current dataframe
    is different than the primary.

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        A list of pandas dataframe to be combined

    Return
    ------
    A pd.DataFrame of combined dataframs.
    """
    # Check if input is list
    if not isinstance(dataframes, list):
        raise ValueError("Input is not a list, please check.")

    # Check if the input list is empty
    if not dataframes:
        raise ValueError("Input list is empty, please check.")

    # Initialize the combined dataframe with the first dataframe
    combined_df = dataframes[0]

    # Loop through the remaining dataframes and combine them
    for i, df in enumerate(dataframes[1:], start=2):
        if not combined_df.columns.equals(df.columns):
            warning_message = f"Schema of DataFrame {i} " + \
                "is different from the primary DataFrame."
            warnings.warn(warning_message, UserWarning)

        # Add missing columns to the combined dataframe and fill with NaN
        for col in df.columns:
            if col not in combined_df.columns:
                combined_df[col] = np.nan

        # Concatenate the dataframes vertically
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Reset the index of the combined dataframe
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df


def add_pin_color_rules(
    adata,
    label_color_dict: dict,
    color_map_name: str = "_spac_colors",
    overwrite: bool = True
) -> Tuple[dict, str]:
    """
    Adds pin color rules to the AnnData object and scans for matching labels.

    This function scans unique labels in each adata.obs and column names in all
    adata tables, to find the labels defined by the pin color rule.

    Parameters
    ----------
    adata
        The anndata object containing upstream analysis.
    label_color_dict : dict
        Dictionary of pin color rules with label as key and color as value.
    color_map_name : str
        The name to use for storing pin color rules in `adata.uns`.
    overwrite : bool, optional
        Whether to overwrite existing pin color rules in `adata.uns` with the
        same name, by default True.

    Returns
    -------
    label_matches : dict
        Dictionary with the matching labels in each
        section (obs, var, X, etc.).
    result_str : str
        Summary string with the matching labels in each
        section (obs, var, X, etc.).

    Raises
    ------
    ValueError
        If `color_map_name` already exists in `adata.uns`
        and `overwrite` is False.
    """

    # Check if the pin color rule already exists in adata.uns
    if color_map_name in adata.uns and not overwrite:
        raise ValueError(
            f"`{color_map_name}` already exists in `adata.uns` ",
            "and `overwrite` is set to False."
        )

    # Add or overwrite pin color rules in adata.uns
    adata.uns[color_map_name] = label_color_dict

    # Initialize a dictionary to store matching labels
    label_matches = {
        'obs': {},
        'var': {},
        'X': {}
    }

    # Initialize the report string
    result_str = "\nobs:\n"

    # Scan unique labels in adata.obs
    for col in adata.obs.columns:
        unique_labels = adata.obs[col].unique()
        matching_labels = [
            label for label in unique_labels if label in label_color_dict
        ]
        label_matches['obs'][col] = matching_labels
        result_str += f"Annotation {col} in obs has matching labels: "
        result_str += f"{matching_labels}\n"

    result_str += "\nvar:\n"
    # Scan unique labels in adata.var
    for col in adata.var.columns:
        unique_labels = adata.var[col].unique()
        matching_labels = [
            label for label in unique_labels if label in label_color_dict
        ]
        label_matches['var'][col] = matching_labels
        result_str += f"Column {col} in var has matching labels: "
        result_str += f"{matching_labels}\n"

    # Scan column names in adata.X
    if isinstance(adata.X, pd.DataFrame):
        col_names = adata.X.columns
    else:
        col_names = [f'feature{i+1}' for i in range(adata.X.shape[1])]
        # If X is a numpy array or sparse matrix

    result_str += "\nRaw data table X:\n"
    matching_labels = [
        label for label in col_names if label in label_color_dict
    ]
    label_matches['X']['column_names'] = matching_labels
    result_str += "Raw data table column names have matching labels: "
    result_str += f"{matching_labels}\n"

    result_str = "\nLabels in the analysis:\n" + result_str

    # Check for labels in label_color_dict that
    # do not match any labels in label_matches
    unmatched_labels = set(label_color_dict.keys()) - set(
        label
        for section in label_matches.values()
        for col in section.values()
        for label in col
    )
    # Append warning for unmatched labels
    if unmatched_labels:
        for label in unmatched_labels:
            result_str = f"{label}\n" + result_str
        result_str = (
            "\nWARNING: The following labels do not match any labels in "
            "the analysis:\n" + result_str
        )
    for label, color in label_color_dict.items():
        result_str = f"{label}: {color}\n" + result_str
    result_str = "Labels with color pinned:\n" + result_str
    result_str = (
        f"Pin Color Rule Labels Count for `{color_map_name}`:\n" + result_str
    )

    adata.uns[color_map_name+"_summary"] = result_str

    return label_matches, result_str

def combine_annotations(
    adata: AnnData,
    annotations: list,
    separator: str,
    new_annotation_name: str
) -> AnnData:
    """
    Combine multiple annotations into a new annotation using a defined separator.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object whose .obs will be modified.

    annotations : list
        List of annotation column names to combine.

    separator : str
        Separator to use when combining annotations.

    new_annotation_name : str
        The name of the new annotation to be created.

    Returns
    -------
    AnnData
        The AnnData object with the combined annotation added.
    """

    # Check that the list is not emply
    if len(annotations) == 0:
        raise ValueError('Annotations list cannot be empty.')
    # Validate input annotations using utility function
    check_annotation(adata, annotations=annotations)

    if type(annotations) is not list:
        raise ValueError(
            f'Annotations must be a list. Got {type(annotations)}'
        )
    # Ensure separator is a string
    if not isinstance(separator, str):
        raise ValueError(
            f'Separator must be a string. Got {type(separator)}'
        )

    # Check if new annotation name already exists
    if new_annotation_name in adata.obs.columns:
        raise ValueError(
            f"'{new_annotation_name}' already exists in adata.obs.")

    # Combine annotations into the new column

    # Convert selected annotations to string type
    annotations_str = adata.obs[annotations].astype(str)

    # Combine annotations using the separator
    combined_annotation = annotations_str.agg(separator.join, axis=1)

    # Assign the combined result to the new annotation column
    adata.obs[new_annotation_name] = combined_annotation

    return adata


def summarize_dataframe(
    df: pd.DataFrame,
    columns,
    print_nan_locations: bool = False
) -> dict:
    """
    Summarize specified columns in a DataFrame.

    For numeric columns, computes summary statistics.
    For categorical columns, returns unique labels and frequencies.
    In both cases, missing values (None/NaN) are flagged and their row indices
    identified.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    columns : str or list of str
        The column name or list of column names to analyze.
    print_nan_locations : bool, optional
        If True, prints the row indices where None/NaN values occur.
        Default is False.

    Returns
    -------
    dict
        A dictionary where each key is a column name and its value is another
        dictionary with:
          - 'data_type': either 'numeric' or 'categorical'
          - 'missing_indices': list of row indices with missing values
          - 'summary': summary statistics if numeric or unique labels with
          counts if categorical
    """
    # Convert a single column string to list
    if isinstance(columns, str):
        columns = [columns]

    results = {}
    for col in columns:
        col_info = {}
        # Identify missing values (None or NaN)
        missing_mask = df[col].isnull()
        missing_indices = df.index[missing_mask].tolist()
        col_info['missing_indices'] = missing_indices
        col_info['count_missing_indices'] = len(missing_indices)

        # Optionally print locations of missing values
        if print_nan_locations and missing_indices:
            print(
                f"Column '{col}' has missing values at rows:"
                " {missing_indices}"
            )

        # If the column is numeric, compute summary statistics
        if pd.api.types.is_numeric_dtype(df[col]):
            data = df[col]
            stats = {
                'count': int(data.count()),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                '25%': data.quantile(0.25),
                '50%': data.median(),
                '75%': data.quantile(0.75),
                'max': data.max()
            }
            col_info['data_type'] = 'numeric'
            col_info['summary'] = stats
        else:
            # Otherwise, treat as categorical
            unique_values = df[col].dropna().unique().tolist()
            value_counts = df[col].value_counts(dropna=True).to_dict()
            col_info['data_type'] = 'categorical'
            col_info['summary'] = {
                'unique_values': unique_values,
                'value_counts': value_counts
            }
        results[col] = col_info

        # Also print a summary to standard output
        print(f"Summary for column '{col}':")
        print(f"Type: {col_info['data_type']}")
        print("Count missing indices:", col_info['count_missing_indices'])
        print("Missing indices:", col_info['missing_indices'])
        print("Details:", col_info['summary'])
        print("-" * 40)
    return results
