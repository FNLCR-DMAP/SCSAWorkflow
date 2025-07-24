from pathlib import Path
import pickle
from typing import Any, Dict, Union, Optional
import json
import anndata as ad


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


def save_outputs(outputs: Dict[str, Any],
                 output_dir: Union[str, Path] = ".") -> Dict[str, str]:
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
    ...     "adata.h5ad": adata,
    ...     "results.csv": results_df,
    ...     "adata.pickle": adata
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
            if type(obj) is not ad.AnnData:
                raise TypeError(
                    f"Object for '{filename}' must be AnnData, got {type(obj)}"
                )
           
            obj.write_h5ad(filepath)
        elif filename.endswith(('.pickle', '.pkl')):
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        elif hasattr(obj, "savefig"):
            obj.savefig(filepath.with_suffix('.png'))
            filepath = filepath.with_suffix('.png')
        else:
            # Default to pickle
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
        
        saved_files[filename] = str(filepath.resolve())
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