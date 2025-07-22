"""
Improved template utilities for SPAC templates.
Designed to work seamlessly with NIDAP-style templates and maintain compatibility.
"""
from pathlib import Path
import pickle
from typing import Any, Dict, Union, Optional, Tuple
import json
import pandas as pd


def load_input(file_path: Union[str, Path], format: Optional[str] = None):
    """
    Load input data from either h5ad or pickle file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to input file (h5ad or pickle)
    format : str, optional
        Force specific format ('h5ad' or 'pickle'). If None, auto-detect.
    
    Returns
    -------
    Loaded data object (typically AnnData)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Determine format
    if format is None:
        suffix = path.suffix.lower()
        if suffix in ['.h5ad', '.h5']:
            format = 'h5ad'
        elif suffix in ['.pickle', '.pkl', '.p']:
            format = 'pickle'
        else:
            # Try to auto-detect
            format = 'auto'
    
    # Load based on format
    if format == 'h5ad':
        try:
            import anndata as ad
            return ad.read_h5ad(path)
        except ImportError:
            raise ImportError("anndata package required to read h5ad files")
        except Exception as e:
            if format == 'auto':
                # Try pickle as fallback
                format = 'pickle'
            else:
                raise ValueError(f"Error reading h5ad file: {e}")
    
    if format == 'pickle' or format == 'auto':
        try:
            with path.open('rb') as fh:
                return pickle.load(fh)
        except Exception as e:
            raise ValueError(
                f"Unable to load file '{file_path}'. "
                f"Supported formats: h5ad, pickle. Error: {e}"
            )


def save_output(
    data: Any,
    output_path: Union[str, Path],
    format: Optional[str] = None
) -> str:
    """
    Save a single output to file. Returns the absolute path as a string.
    This matches NIDAP behavior more closely.
    
    Parameters
    ----------
    data : Any
        Data to save (AnnData, DataFrame, etc.)
    output_path : str or Path
        Output file path
    format : str, optional
        Force specific format. If None, auto-detect from extension.
    
    Returns
    -------
    str
        Absolute path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format is None:
        suffix = output_path.suffix.lower()
        if suffix in ['.h5ad', '.h5']:
            format = 'h5ad'
        elif suffix in ['.pickle', '.pkl', '.p']:
            format = 'pickle'
        elif suffix == '.csv':
            format = 'csv'
        else:
            # Default based on data type
            if hasattr(data, 'write_h5ad'):
                format = 'h5ad'
            elif isinstance(data, pd.DataFrame):
                format = 'csv'
            else:
                format = 'pickle'
    
    # Save based on format
    if format == 'h5ad':
        if hasattr(data, 'write_h5ad'):
            data.write_h5ad(output_path)
        else:
            raise ValueError(f"Cannot save {type(data)} as h5ad")
    
    elif format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Cannot save {type(data)} as CSV")
    
    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    abs_path = str(output_path.resolve())
    print(f"Saved: {abs_path}")
    return abs_path


def save_outputs(
    outputs: Union[Dict[str, Any], Any],
    output_path: Optional[Union[str, Path]] = None
) -> Union[Dict[str, str], str]:
    """
    Flexible save function that handles both single outputs and multiple outputs.
    
    Parameters
    ----------
    outputs : dict or Any
        If dict: {filename: data} pairs to save
        If not dict: single data object to save
    output_path : str or Path, optional
        For single output: the output path
        For dict: the output directory
    
    Returns
    -------
    dict or str
        If input was dict: returns {filename: absolute_path}
        If input was single object: returns absolute_path string
    """
    if isinstance(outputs, dict):
        # Multiple outputs
        output_dir = Path(output_path) if output_path else Path(".")
        saved_files = {}
        
        for filename, data in outputs.items():
            filepath = output_dir / filename
            abs_path = save_output(data, filepath)
            saved_files[filename] = abs_path
        
        return saved_files
    else:
        # Single output
        if output_path is None:
            raise ValueError("output_path required for single output")
        
        return save_output(outputs, output_path)


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
        if path.exists() or (isinstance(json_input, str) and json_input.endswith('.json')):
            with open(path, 'r') as file:
                return json.load(file)
        else:
            # Try as JSON string
            try:
                return json.loads(str(json_input))
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON string or file path: {json_input}")
    
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
) -> Any:
    """
    Converts a string to a specified value or type.
    Exact copy from NIDAP code_workbook_utils.
    """
    if var is None:
        return value_to_convert_to
    
    # Convert to string for comparison
    var_str = str(var).strip()
    
    if (var_str.lower() == default_none_text.lower()) or var_str == '':
        return value_to_convert_to
    
    if to_float:
        try:
            return float(var_str)
        except ValueError:
            raise ValueError(
                f"Error: can't convert {param_name} to float. "
                f"Received:\"{var}\""
            )
    
    if to_int:
        try:
            return int(var_str)
        except ValueError:
            raise ValueError(
                f"Error: can't convert {param_name} to integer. "
                f"Received:\"{var}\""
            )
    
    return var


def prepare_ripley_for_h5ad(adata) -> None:
    """
    Prepare Ripley L results for h5ad serialization.
    Converts complex objects to h5ad-compatible formats.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with ripley_l results in uns
    """
    if 'ripley_l' not in adata.uns:
        return
    
    ripley_data = adata.uns['ripley_l']
    
    # If it's already a DataFrame, it should be fine
    if isinstance(ripley_data, pd.DataFrame):
        # Make sure all columns are h5ad-compatible types
        for col in ripley_data.columns:
            if ripley_data[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    ripley_data[col] = pd.to_numeric(ripley_data[col], errors='raise')
                except:
                    # Convert to string if numeric conversion fails
                    ripley_data[col] = ripley_data[col].astype(str)
        
        adata.uns['ripley_l'] = ripley_data
    
    elif isinstance(ripley_data, dict):
        # Complex dict structure - need to flatten or convert
        # This depends on the exact structure from plot_ripley_l
        print("Warning: Complex ripley_l structure detected. May need custom conversion.")
        # You might need to implement specific conversion logic here
        # based on what plot_ripley_l expects


def get_output_format(params: Dict[str, Any]) -> str:
    """
    Determine output format from parameters.
    
    Parameters
    ----------
    params : dict
        Parameters dictionary
    
    Returns
    -------
    str
        'pickle' or 'h5ad'
    """
    output_file = params.get('Output_File', '')
    
    if isinstance(output_file, str):
        if output_file.endswith(('.pickle', '.pkl')):
            return 'pickle'
        elif output_file.endswith('.h5ad'):
            return 'h5ad'
    
    # Default to pickle for maximum compatibility
    return 'pickle'


# Convenience functions for templates
def save_adata_output(adata, params: Dict[str, Any]) -> str:
    """
    Save AnnData output based on parameters.
    Automatically handles pickle vs h5ad format.
    
    Parameters
    ----------
    adata : AnnData
        Data to save
    params : dict
        Parameters dict containing Output_File
    
    Returns
    -------
    str
        Path to saved file
    """
    output_path = params.get('Output_File', 'transform_output.pickle')
    format = get_output_format(params)
    
    if format == 'h5ad':
        # Prepare data for h5ad if needed
        prepare_ripley_for_h5ad(adata)
    
    return save_output(adata, output_path, format=format)


def load_upstream_data(params: Dict[str, Any]):
    """
    Load upstream analysis data based on parameters.
    
    Parameters
    ----------
    params : dict
        Parameters dict containing Upstream_Analysis
    
    Returns
    -------
    AnnData
        Loaded data
    """
    upstream_path = params.get('Upstream_Analysis')
    if not upstream_path:
        raise ValueError("Upstream_Analysis parameter is required")
    
    return load_input(upstream_path)

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