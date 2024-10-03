import re
import os
import pandas as pd
import anndata as ad
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
import logging
from collections import defaultdict
from spac.utils import regex_search_list, check_list_in_list


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


def combine_dfs_depracated(dataframes, annotations):

    """
    Combine a list of pandas dataframe into single pandas dataframe.

    Parameters
    ----------
    dataframes : list of tuple
        A list containing (file name, pandas dataframe) to be combined
        into single dataframe output

    annotations : pandas.DataFrame
        A pandas data frame where the index is the file name, and
        the columns are various annotations to
        add to all cells in a given dataframe.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame of all the cells
        where each cell has a unique index.
    """

    meta_schema = []
    combined_dataframe = pd.DataFrame()
    if not str(type(annotations)) == "<class 'pandas.core.frame.DataFrame'>":
        annotations_type = str(type(annotations))
        error_message = "annotations should be a pandas dataframe, " + \
            "but got " + annotations_type + "."
        raise TypeError(error_message)

    for current_df_list in dataframes:

        file_name = current_df_list[0]
        current_df = current_df_list[1]

        # Check is schema of each data_frame matches.
        # Check for length first, then check if columns match
        # The overall schema is based on the first file read.
        current_schema = current_df.columns.to_list()

        if len(meta_schema) == 0:
            meta_schema = current_schema
            print("Meta schema acquired. Columns are:")
            for column_name in meta_schema:
                print(column_name)

        if len(meta_schema) == len(current_schema):
            if set(meta_schema) != set(current_schema):
                error_message = "Column in current file does not match " + \
                        "the meta_schema, got:\n {current_schema}. "
                raise ValueError(error_message)
        else:
            error_message = "Column in current file does not match " + \
                        "the meta_schema, got:\n {current_schema}. "
            raise ValueError(error_message)

        # Check if the annotations DataFrame has the required index
        if file_name not in annotations.index:
            error_message = "Missing data in the annotations DataFrame" + \
                f"for the file '{file_name}'."
            raise ValueError(error_message)

        # Add annotations in to the dataframe
        file_annotations = annotations.loc[file_name]

        for file_annotation_name, file_annotation_value in \
                file_annotations.iteritems():
            current_df[file_annotation_name] = file_annotation_value

        if combined_dataframe.empty:
            combined_dataframe = current_df.copy()
        else:
            # Concatenate the DataFrames, with error handling
            try:
                combined_dataframe = pd.concat(
                    [combined_dataframe, current_df]
                    )
            except (ValueError, TypeError) as e:
                print('Error concatenating DataFrames:', e)

    # Reset index of the combined_dataframe
    combined_dataframe.reset_index(drop=True, inplace=True)

    print("CSVs are combined into single dataframe!")
    print(combined_dataframe.info())

    return combined_dataframe


def select_values(data, annotation, values=None):
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

    Returns
    -------
    pandas.DataFrame or anndata.AnnData
        The filtered DataFrame or AnnData object containing only the selected
        rows based on the annotation and values.
    """
    # Ensure values are in a list format if not None
    if values is not None and not isinstance(values, list):
        values = [values]

    # Initialize possible_annotations based on the data type
    if isinstance(data, pd.DataFrame):
        possible_annotations = data.columns.tolist()
    elif isinstance(data, ad.AnnData):
        possible_annotations = data.obs.columns.tolist()
    else:
        error_msg = (
            "Unsupported data type. Data must be either a pandas DataFrame"
            " or an AnnData object."
        )
        logging.error(error_msg)
        raise TypeError(error_msg)

    # Check if the annotation exists using check_list_in_list
    check_list_in_list(
        input=[annotation],
        input_name="annotation",
        input_type="column name/annotation key",
        target_list=possible_annotations,
        need_exist=True
    )

    # Validate provided values against unique ones, if not None
    if values is not None:
        if isinstance(data, pd.DataFrame):
            unique_values = data[annotation].astype(str).unique().tolist()
        elif isinstance(data, ad.AnnData):
            unique_values = data.obs[annotation].astype(str).unique().tolist()
        check_list_in_list(
            values, "values", "label", unique_values, need_exist=True
        )

    # Proceed with filtering based on data type and count matching cells
    if isinstance(data, pd.DataFrame):
        filtered_data = data if values is None else \
            data[data[annotation].isin(values)]
        count = filtered_data.shape[0]
    elif isinstance(data, ad.AnnData):
        filtered_data = data if values is None else \
            data[data.obs[annotation].isin(values)]
        count = filtered_data.n_obs

    logging.info(f"Summary of returned dataset: {count} cells "
                 "match the selected labels.")

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
