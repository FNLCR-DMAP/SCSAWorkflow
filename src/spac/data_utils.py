import re
import os
import pandas as pd
import anndata as ad
from sklearn.preprocessing import MinMaxScaler


def ingest_cells(dataframe, regex_str, x_col=None, y_col=None, obs=None):

    """
    Read the csv file into an anndata object.

    The function will also intialize intensities and spatial coordiantes.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The data frame that contains cells as rows, and cells informations as
        columns.

    regex_str : str or list of str
        A string or a list of strings representing python regular expression
        for the intensities columns in the data frame.  x_col : str The column
        name for the x coordinate of the cell.

    y_col : str
        The column name for the y coordinate of the cell.

    obs : str or list of str
        The column name for the re gion that the cells. If a list is passed,
        multiple observations will be created in the returned AnnData object.


    Returns
    -------
    anndata.AnnData
        The generated AnnData object
    """

    if not isinstance(regex_str, list):
        regex_list = [regex_str]
    else:
        regex_list = regex_str

    all_intensities = []
    all_columns = list(dataframe.columns)
    for regex in regex_list:
        intensities_regex = re.compile(regex)
        intensities = list(
            filter(intensities_regex.match, all_columns))
        all_intensities.extend(intensities)

    intensities_df = dataframe[all_intensities]
    adata = ad.AnnData(
        intensities_df,
        dtype=intensities_df[all_intensities[0]].dtype)

    if obs is not None:
        if isinstance(obs, str):
            list_of_obs = [obs]
        else:
            list_of_obs = obs

        for observation in list_of_obs:

            # As selecting one column of the dataframe returns a series which
            # AnnData converts to NaN, then I convert it to a list before
            # assignment.
            adata.obs[observation] = dataframe[observation].tolist()

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


def rescale_intensities(intensities, min_quantile=0.01, max_quantile=0.99):
    """
    Clip and rescale intensities outside the minimum and maximum quantile.

    The rescaled intensities will be between 0 and 1.

    Parameters
    ----------
    intensities : pandas.Dataframe
        The DataRrame of intensities.

    min_quantile : float
        The minimum quantile to be consider zero.

    max_quantile: float
        The maximum quantile to be considerd 1.

    Returns
    -------
    pandas.DataFrame
        The created DataFrame with normalized intensities.
    """
    markers_max_quantile = intensities.quantile(max_quantile)
    markers_min_quantile = intensities.quantile(min_quantile)

    intensities_clipped = intensities.clip(
        markers_min_quantile,
        markers_max_quantile,
        axis=1)

    scaler = MinMaxScaler()
    np_intensities_scaled = scaler.fit_transform(
        intensities_clipped.to_numpy())

    intensities_scaled = pd.DataFrame(
        np_intensities_scaled,
        columns=intensities_clipped.columns)

    return intensities_scaled


def add_rescaled_intensity(adata, min_quantile, max_quantile, layer):
    """
    Clip and rescale the intensities matrix.

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
    rescaled = rescale_intensities(original, min_quantile, max_quantile)
    adata.layers[layer] = rescaled


def subtract_min_per_region(adata, obs, layer, min_quantile=0.01):
    """
    Substract the minimum quantile of every marker per region.

    Parameters
    ----------
    adata : anndata.AnnData
         The AnnData object.

    obs: str
        The name of the observation in `adata` to define batches.

    min_quantile : float
        The minimum quantile to rescale to zero.

    layer : str
        The name of the new layer to add to the AnnData object.
    """
    regions = adata.obs[obs].unique().tolist()
    original = adata.to_df()

    new_df_list = []
    for region in regions:
        region_cells = original[adata.obs[obs] == region]
        new_intensities = subtract_min_quantile(region_cells, min_quantile)
        new_df_list.append(new_intensities)

    new_df = pd.concat(new_df_list)
    adata.layers[layer] = new_df


def subtract_min_quantile(intensities, min_quantile=.01):
    """
    Subtract the intensity defined by the minimum quantile from all columns.

    Parameters
    ----------

    intensities : pandas.DataFrame
        The dataframe of intensities.

    min_quantile: float
        The minimum quantile to be consider zero.

    Returns
    -------
    pandas.DataFrame
        dataframe with rescaled intensities.
    """
    columns_min_quantile = intensities.quantile(min_quantile)

    subtracted_min = intensities - columns_min_quantile

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
    list
        A list of pandas dataframe of all the csv files.
    """

    meta_schema = []
    dataframe_list = []

    if not isinstance(file_names, list):
        if not isinstance(file_names, str):
            file_name_type = type(file_names)
            error_message = "file_names should be list or string" + \
                            ", but got " + str(file_name_type) + "."
            raise TypeError(error_message)
        else:
            file_names = [file_names]

    for file_name in file_names:

        # Check if the file exists
        if not os.path.exists(file_name):
            error_message = f"The file '{file_name}' does not exist."
            raise FileNotFoundError(error_message)

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

        current_schema = current_df.columns.to_list()

        if len(meta_schema) == 0:
            meta_schema = current_schema
            print("Meta schema acquired. Columns are:")
            for column_name in meta_schema:
                print(column_name)

        if len(meta_schema) == len(current_schema):
            if set(meta_schema) != set(current_schema):
                error_message = "Column in current file does not match " + \
                        f"the meta_schema, got:\n {current_schema}. "
                raise ValueError(error_message)
        else:
            error_message = "Column in current file does not match " + \
                        f"the meta_schema, got:\n {current_schema}. "
            raise ValueError(error_message)

        dataframe_list.append([file_name, current_df])

    print("CSVs are converted into dataframes and combined into a list!")
    print("Total of " + str(len(dataframe_list)) + " dataframes in the list.")
    for each_file in dataframe_list:
        print("File name: ", each_file[0])
        print("Info: ")
        print(each_file[1].info())
        print("Description: ")
        print(each_file[1].describe())
        print()

    return dataframe_list


def combine_dfs(dataframes, observations):

    """
    Combine a list of pandas dataframe into single pandas dataframe.

    Parameters
    ----------
    dataframes : list of tuple
        A list containing (file name, pandas dataframe) to be combined
        into single dataframe output

    observations : pandas.DataFrame
        A pandas data frame where the index is the file name, and
        the columns are various observations to
        add to all cells in a given dataframe.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame of all the cells
        where each cell has a unique index.
    """

    meta_schema = []
    combined_dataframe = pd.DataFrame()
    if not str(type(observations)) == "<class 'pandas.core.frame.DataFrame'>":
        observations_type = str(type(observations))
        error_message = "observations should be a pandas dataframe, " + \
            "but got " + observations_type + "."
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

        # Check if the observations DataFrame has the required index
        if file_name not in observations.index:
            error_message = "Missing data in the observations DataFrame" + \
                f"for the file '{file_name}'."
            raise ValueError(error_message)

        # Add observations in to the dataframe
        file_observations = observations.loc[file_name]

        for file_obs_name, file_obs_value in file_observations.iteritems():
            current_df[file_obs_name] = file_obs_value

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


def select_values(data, observation_name, values=None):
    """
    Selects rows from input dataframe matching specified values in a column.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe.
    observation_name : str
        The column name to be used for selection.
    values : list, optional
        List of values for observation_name to include.
        If None, return all values.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing only the selected rows.

    Raises
    ------
    ValueError
        If observation_name does not exist or one or more values passed
        do not exist in the specified column.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'column1': ['A', 'B', 'A', 'B', 'A'],
    ...     'column2': [1, 2, 3, 4, 5]
    ... })
    >>> select_values(df, 'column1', ['A'])
      column1  column2
    0       A        1
    2       A        3
    4       A        5
    """
    # Check if the DataFrame is empty
    if not data.empty:
        # If DataFrame is not empty, check if observation_name exists
        if observation_name not in data.columns:
            raise ValueError(
                f"Column {observation_name} does not exist in the dataframe"
            )

        # If values exist in observation_name column, filter data
        if values is not None:
            data = data[data[observation_name].isin(values)]

    return data


def downsample_cells(data, observation_name, n_samples=None,
                     stratify=False, rand=False):
    """
    Reduces the number of cells in the data by either selecting n_samples from
    every possible value of observation_name, or returning n_samples
    stratified by the frequency of values in observation_name.

    Parameters
    ----------
    data : pd.DataFrame
        The input data frame.
    observation_name : str
        The column name to downsample on.
    n_samples : int, default=None
        The max number of samples to return for each group if stratify is
        False, or in total if stratify is True. If None, all samples returned.
    stratify : bool, default=False
        If true, stratify the returned values based on their input frequency.
    rand : bool, default=False
        If true and stratify is True, randomly select the returned cells.
        Otherwise, choose the first n cells.

    Returns
    -------
    data : pd.DataFrame
        The downsampled data frame.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...    'observation': ['a', 'a', 'a', 'b', 'b', 'c'],
    ...    'value': [1, 2, 3, 4, 5, 6]
    ... })
    >>> print(downsample_cells(df, 'observation', n_samples=2))
    """
    # Check if the column to downsample on exists
    if observation_name not in data.columns:
        raise ValueError(
            f"Column {observation_name} does not exist in the dataframe"
        )

    if n_samples is not None:
        # Stratify selection
        if stratify:
            # Determine frequencies of each group
            freqs = data[observation_name].value_counts(normalize=True)
            n_samples_per_group = (freqs * n_samples).astype(int)
            samples = []
            # Group by observation_name and sample from each group
            for group, group_data in data.groupby(observation_name):
                n_group_samples = n_samples_per_group.get(group, 0)
                if rand:
                    # Randomly select the returned cells
                    samples.append(group_data.sample(min(n_group_samples,
                                                         len(group_data))))
                else:
                    # Choose the first n cells
                    samples.append(group_data.head(min(n_group_samples,
                                                       len(group_data))))
            # Concatenate all samples
            data = pd.concat(samples)
        else:
            # Non-stratified selection
            # Select the first n cells from each group
            data = data.groupby(observation_name).apply(
                lambda x: x.head(n=min(n_samples, len(x)))
            ).reset_index(drop=True)

    return data
