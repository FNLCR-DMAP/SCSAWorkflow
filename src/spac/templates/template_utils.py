from pathlib import Path
import pickle
from typing import Any, Dict, Union, Optional, List
import json
import pandas as pd
import anndata as ad
import re
import logging
logger = logging.getLogger(__name__)


def load_input(file_path: Union[str, Path]):
    """
    Load input data from either h5ad or pickle file.

    Parameters
    ----------
    file_path : str or Path
        Path to input file (h5ad or pickle)

    Returns
    -------
    Loaded data object (typically AnnData)
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Check file extension
    suffix = path.suffix.lower()

    if suffix in ['.h5ad', '.h5']:
        # Load h5ad file
        try:
            import anndata as ad
            return ad.read_h5ad(path)
        except ImportError:
            raise ImportError(
                "anndata package required to read h5ad files"
            )
        except Exception as e:
            raise ValueError(f"Error reading h5ad file: {e}")

    elif suffix in ['.pickle', '.pkl', '.p']:
        # Load pickle file
        with path.open('rb') as fh:
            return pickle.load(fh)

    else:
        # Try to detect file type by content
        try:
            # First try h5ad
            import anndata as ad
            return ad.read_h5ad(path)
        except Exception:
            # Fall back to pickle
            try:
                with path.open('rb') as fh:
                    return pickle.load(fh)
            except Exception as e:
                raise ValueError(
                    f"Unable to load file '{file_path}'. "
                    f"Supported formats: h5ad, pickle. Error: {e}"
                )


def save_outputs(
    outputs: Dict[str, Any],
    output_dir: Union[str, Path] = "."
) -> Dict[str, str]:
    """
    Save multiple outputs to files and return a dict {filename: absolute_path}.
    (Always a dict, even if just one file.)

    Parameters
    ----------
    outputs : dict
        Dictionary where:
        - key: filename (with extension)
        - value: object to save
    output_dir : str or Path
        Directory to save files

    Returns
    -------
    dict
        Dictionary of saved file paths

    Example
    -------
    >>> outputs = {
    ...     "adata.pickle": adata,  # Preferred format
    ...     "results.csv": results_df,
    ...     "adata.h5ad": adata  # Still supported
    ... }
    >>> saved = save_outputs(outputs, "results/")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for filename, obj in outputs.items():
        filepath = output_dir / filename

        # Save based on file extension
        if filename.endswith('.csv'):
            obj.to_csv(filepath, index=False)
        elif filename.endswith('.h5ad'):
            # Still support h5ad, but not the default
            if type(obj) is not ad.AnnData:
                raise TypeError(
                    f"Object for '{str(filename)}' must be AnnData, "
                    f"got {type(obj)}"
                )
            logger.info(f"Saving AnnData to {str(filepath)}")
            logger.debug(f"AnnData object: {obj}")
            obj.write_h5ad(str(filepath))
            logger.info(f"Saved AnnData to {str(filepath)}")
        elif filename.endswith(('.pickle', '.pkl', '.p')):
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        elif hasattr(obj, "savefig"):
            obj.savefig(filepath.with_suffix('.png'))
            filepath = filepath.with_suffix('.png')
        else:
            # Default to pickle
            filepath = filepath.with_suffix('.pickle')
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)

        print(type(filepath))
        print(type(filename))
        saved_files[str(filename)] = str(filepath)
        print(f"Saved: {filepath}")

    return saved_files


def parse_params(
    json_input: Union[str, Path, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Parse parameters from JSON file, string, or dict.

    Parameters
    ----------
    json_input : str, Path, or dict
        JSON file path, JSON string, or dictionary

    Returns
    -------
    dict
        Parsed parameters
    """
    if isinstance(json_input, dict):
        return json_input

    if isinstance(json_input, (str, Path)):
        path = Path(json_input)

        # Check if it's a file path
        if path.exists() or str(json_input).endswith('.json'):
            with open(path, 'r') as file:
                return json.load(file)
        else:
            # It's a JSON string
            return json.loads(str(json_input))

    raise TypeError(
        "json_input must be dict, JSON string, or path to JSON file"
    )


def text_to_value(
    var: Any,
    default_none_text: str = "None",
    value_to_convert_to: Any = None,
    to_float: bool = False,
    to_int: bool = False,
    param_name: str = ''
):
    """
    Converts a string to a specified value or type. Handles conversion to
    float or integer and provides a default value if the input string
    matches a specified 'None' text.

    Parameters
    ----------
    var : str
        The input string to be converted.
    default_none_text : str, optional
        The string that represents a 'None' value. If `var` matches this
        string, it will be converted to `value_to_convert_to`.
        Default is "None".
    value_to_convert_to : any, optional
        The value to assign to `var` if it matches `default_none_text` or
        is an empty string. Default is None.
    to_float : bool, optional
        If True, attempt to convert `var` to a float. Default is False.
    to_int : bool, optional
        If True, attempt to convert `var` to an integer. Default is False.
    param_name : str, optional
        The name of the parameter, used in error messages for conversion
        failures. Default is ''.

    Returns
    -------
    any
        The converted value, which may be the original string, a float,
        an integer, or the specified `value_to_convert_to`.

    Raises
    ------
    ValueError
        If `to_float` or `to_int` is set to True and conversion fails.

    Notes
    -----
    - If both `to_float` and `to_int` are set to True, the function will
      prioritize conversion to float.
    - If the string `var` matches `default_none_text` or is an empty
      string, `value_to_convert_to` is returned.

    Examples
    --------
    Convert a string representing a float:

    >>> text_to_value("3.14", to_float=True)
    3.14

    Handle a 'None' string:

    >>> text_to_value("None", value_to_convert_to=None)
    None

    Convert a string to an integer:

    >>> text_to_value("42", to_int=True)
    42

    Handle invalid conversion:

    >>> text_to_value("abc", to_int=True, param_name="test_param")
    Error: can't convert test_param to integer. Received:"abc"
    'abc'
    """
    # Handle non-string inputs
    if not isinstance(var, str):
        var = str(var)

    none_condition = (
        var.lower().strip() == default_none_text.lower().strip() or
        var.strip() == ''
    )

    if none_condition:
        var = value_to_convert_to

    elif to_float:
        try:
            var = float(var)
        except ValueError:
            error_msg = (
                f'Error: can\'t convert {param_name} to float. '
                f'Received:"{var}"'
            )
            raise ValueError(error_msg)

    elif to_int:
        try:
            var = int(var)
        except ValueError:
            error_msg = (
                f'Error: can\'t convert {param_name} to integer. '
                f'Received:"{var}"'
            )
            raise ValueError(error_msg)

    return var


def convert_to_floats(text_list: List[Any]) -> List[float]:
    """
    Convert list of text values to floats.

    Parameters
    ----------
    text_list : list
        List of values to convert

    Returns
    -------
    list
        List of float values

    Raises
    ------
    ValueError
        If any value cannot be converted to float
    """
    float_list = []
    for value in text_list:
        try:
            float_list.append(float(value))
        except ValueError:
            msg = f"Failed to convert value: '{value}' to float."
            raise ValueError(msg)
    return float_list


def convert_pickle_to_h5ad(
    pickle_path: Union[str, Path],
    h5ad_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Convert a pickle file containing AnnData to h5ad format.

    Parameters
    ----------
    pickle_path : str or Path
        Path to input pickle file
    h5ad_path : str or Path, optional
        Path for output h5ad file. If None, uses same name with .h5ad
        extension

    Returns
    -------
    str
        Path to saved h5ad file
    """
    pickle_path = Path(pickle_path)

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    # Load from pickle
    with pickle_path.open('rb') as fh:
        adata = pickle.load(fh)

    # Check if it's AnnData
    try:
        import anndata as ad
        if not isinstance(adata, ad.AnnData):
            raise TypeError(
                f"Loaded object is not AnnData, got {type(adata)}"
            )
    except ImportError:
        raise ImportError(
            "anndata package required for conversion to h5ad"
        )

    # Determine output path
    if h5ad_path is None:
        h5ad_path = pickle_path.with_suffix('.h5ad')
    else:
        h5ad_path = Path(h5ad_path)

    # Save as h5ad
    adata.write_h5ad(h5ad_path)

    return str(h5ad_path)


def spell_out_special_characters(text: str) -> str:
    """
    Clean column names by replacing special characters with text equivalents.

    Handles biological marker names like:
    - "CD4+" → "CD4_pos"
    - "CD8-" → "CD8_neg"
    - "CD4+CD20-" → "CD4_pos_CD20_neg"
    - "CD4+/CD20-" → "CD4_pos_slashCD20_neg"
    - "CD4+ CD20-" → "CD4_pos_CD20_neg"
    - "Area µm²" → "Area_um2"

    Parameters
    ----------
    text : str
        The text to clean

    Returns
    -------
    str
        Cleaned text with special characters replaced
    """
    # Replace spaces with underscores
    text = text.replace(' ', '_')

    # Replace specific substrings for units
    text = text.replace('µm²', 'um2')
    text = text.replace('µm', 'um')

    # Handle hyphens between alphanumeric characters FIRST
    # (before + and - replacements)
    # This pattern matches a hyphen that has alphanumeric on both sides
    text = re.sub(r'(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])', '_', text)

    # Now replace remaining '+' with '_pos_' and '-' with '_neg_'
    text = text.replace('+', '_pos_')
    text = text.replace('-', '_neg_')

    # Mapping for specific characters
    special_char_map = {
        'µ': 'u',       # Micro symbol replaced with 'u'
        '²': '2',       # Superscript two replaced with '2'
        '@': 'at',
        '#': 'hash',
        '$': 'dollar',
        '%': 'percent',
        '&': 'and',
        '*': 'asterisk',
        '/': 'slash',
        '\\': 'backslash',
        '=': 'equals',
        '^': 'caret',
        '!': 'exclamation',
        '?': 'question',
        '~': 'tilde',
        '|': 'pipe',
        ',': '',        # Remove commas
        '(': '',        # Remove parentheses
        ')': '',        # Remove parentheses
        '[': '',        # Remove brackets
        ']': '',        # Remove brackets
        '{': '',        # Remove braces
        '}': '',        # Remove braces
    }

    # Replace special characters using special_char_map
    for char, replacement in special_char_map.items():
        text = text.replace(char, replacement)

    # Remove any remaining disallowed characters 
    # (keep only alphanumeric and underscore)
    text = re.sub(r'[^a-zA-Z0-9_]', '', text)

    # Remove multiple consecutive underscores and 
    # replace with single underscore
    text = re.sub(r'_+', '_', text)
    
    # Strip both leading and trailing underscores
    text = text.strip('_')

    return text


def load_csv_files(
    csv_dir: Union[str, Path],
    files_config: pd.DataFrame,
    string_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load and combine CSV files based on configuration.

    Parameters
    ----------
    csv_dir : str or Path
        Directory containing CSV files
    files_config : pd.DataFrame
        Configuration dataframe with 'file_name' column and optional 
        metadata
    string_columns : list, optional
        Columns to force as string type

    Returns
    -------
    pd.DataFrame
        Combined dataframe with all CSV data
    """
    import pprint
    from spac.data_utils import combine_dfs
    from spac.utils import check_list_in_list

    csv_dir = Path(csv_dir)
    filename = "file_name"

    # Clean configuration
    files_config = files_config.applymap(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    # Get column names
    all_column_names = files_config.columns.tolist()
    filtered_column_names = [
        col for col in all_column_names if col not in [filename]
    ]

    # Validate string_columns
    if string_columns is None:
        string_columns = []
    elif not isinstance(string_columns, list):
        raise ValueError(
            "String Columns must be a *list* of column names (strings)."
        )

    # Handle ["None"] or [""] => empty list
    if (len(string_columns) == 1 and
        isinstance(string_columns[0], str) and
        text_to_value(string_columns[0]) is None):
        string_columns = []

    # Extract data types
    dtypes = files_config.dtypes.to_dict()

    # Clean column names
    def clean_column_name(column_name):
        original = column_name
        cleaned = spell_out_special_characters(column_name)
        # Ensure doesn't start with digit
        if cleaned and cleaned[0].isdigit():
            cleaned = f'col_{cleaned}'
        if original != cleaned:
            print(f'Column Name Updated: "{original}" -> "{cleaned}"')
        return cleaned

    # Get files to process
    files_config = files_config.astype(str)
    files_to_use = [
        f.strip() for f in files_config[filename].tolist()
    ]

    # Check all files exist
    missing_files = []
    for file_name in files_to_use:
        if not (csv_dir / file_name).exists():
            missing_files.append(file_name)

    if missing_files:
        raise TypeError(
            f"The following files are not found: "
            f"{', '.join(missing_files)}"
        )

    # Prepare dtype override
    dtype_override = (
        {col: str for col in string_columns} if string_columns else None
    )

    # Process files
    processed_df_list = []
    first_file = True

    for file_name in files_to_use:
        file_path = csv_dir / file_name
        file_locations = files_config[
            files_config[filename] == file_name
        ].index.tolist()

        # Check for duplicate file names
        if len(file_locations) > 1:
            print(
                f'Multiple entries for file: "{file_name}", exiting...'
            )
            return None

        try:
            current_df = pd.read_csv(file_path, dtype=dtype_override)
            print(f'\nProcessing file: "{file_name}"')
            current_df.columns = [
                clean_column_name(col) for col in current_df.columns
            ]

            # Validate string_columns exist
            if first_file and string_columns:
                check_list_in_list(
                    input=string_columns,
                    input_name='string_columns',
                    input_type='column',
                    target_list=list(current_df.columns),
                    need_exist=True,
                    warning=False
                )
                first_file = False

        except pd.errors.EmptyDataError:
            raise TypeError(f'The file: "{file_name}" is empty.')
        except pd.errors.ParserError:
            raise TypeError(
                f'The file "{file_name}" could not be parsed. '
                'Please check that the file is a valid CSV.'
            )

        current_df[filename] = file_name

        # Reorder columns
        cols = current_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index(filename)))
        current_df = current_df[cols]

        processed_df_list.append(current_df)
        print(f'File: "{file_name}" Processed!\n')

    # Combine dataframes
    final_df = combine_dfs(processed_df_list)

    # Ensure string columns remain strings
    for col in string_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(str)

    # Add metadata columns
    if filtered_column_names:
        for column in filtered_column_names:
            # Map values from config
            file_to_value = (
                files_config.set_index(filename)[column].to_dict()
            )
            final_df[column] = final_df[filename].map(file_to_value)
            # Ensure correct dtype
            final_df[column] = final_df[column].astype(dtypes[column])

            print(f'\n\nColumn "{column}" Mapping: ')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(file_to_value)

    print("\n\nFinal Dataframe Info")
    print(final_df.info())

    return final_df


def string_list_to_dictionary(
    input_list: List[str],
    key_name: str = "key",
    value_name: str = "color"
) -> Dict[str, str]:
    """
    Validate that a list contains strings in the "key:value" format
    and return the parsed dictionary. Reports all invalid entries with
    custom key and value names in error messages.

    Parameters
    ----------
    input_list : list
        List of strings to validate and parse
    key_name : str, optional
        Name to describe the 'key' part in error messages. Default is "key"
    value_name : str, optional
        Name to describe the 'value' part in error messages. Default is "color"

    Returns
    -------
    dict
        A dictionary parsed from the input list if all entries are valid

    Raises
    ------
    TypeError
        If input is not a list
    ValueError
        If any entry in the list is not a valid "key:value" format

    Examples
    --------
    >>> string_list_to_dictionary(["red:#FF0000", "blue:#0000FF"])
    {'red': '#FF0000', 'blue': '#0000FF'}
    
    >>> string_list_to_dictionary(["TypeA:Cancer", "TypeB:Normal"], "cell_type", "diagnosis")
    {'TypeA': 'Cancer', 'TypeB': 'Normal'}
    """
    if not isinstance(input_list, list):
        raise TypeError("Input must be a list.")

    parsed_dict = {}
    errors = []
    seen_keys = set()

    for entry in input_list:
        if not isinstance(entry, str):
            errors.append(
                f"\nInvalid entry '{entry}': Must be a string in the "
                f"'{key_name}:{value_name}' format."
            )
            continue
        if ":" not in entry:
            errors.append(
                f"\nInvalid entry '{entry}': Missing ':' separator to "
                f"separate '{key_name}' and '{value_name}'."
            )
            continue

        key, *value = map(str.strip, entry.split(":", 1))
        if not key or not value:
            errors.append(
                f"\nInvalid entry '{entry}': Both '{key_name}' and "
                f"'{value_name}' must be non-empty."
            )
            continue

        if key in seen_keys:
            errors.append(f"\nDuplicate {key_name} '{key}' found.")
        else:
            seen_keys.add(key)
            parsed_dict[key] = value[0]

        # Add to dictionary if valid
        parsed_dict[key] = value[0]

    # Raise error if there are invalid entries
    if errors:
        raise ValueError(
            "\nValidation failed for the following entries:\n" +
            "\n".join(errors)
        )

    return parsed_dict