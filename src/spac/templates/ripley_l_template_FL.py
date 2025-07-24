"""
Platform-agnostic Ripley-L template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.ripley_l_template import run_from_json
>>> run_from_json("examples/ripley_l_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spac.spatial_analysis import ripley_l
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


# def _prepare_ripley_uns_for_h5ad(adata) -> None:
#     """
#     Minimal fix for ripley_l results serialization.
#     Converts object columns to appropriate types for H5AD storage.
#     """
#     if "ripley_l" not in adata.uns:
#         return

#     rl = adata.uns.get("ripley_l")
#     if isinstance(rl, pd.DataFrame):
#         # Create a copy to avoid modifying the original
#         clean_df = rl.copy()

#         # Process each object column
#         for col in clean_df.columns:
#             if clean_df[col].dtype == "object":
#                 # Try to convert to numeric
#                 try:
#                     clean_df[col] = pd.to_numeric(
#                         clean_df[col], errors='raise'
#                     )
#                 except Exception:
#                     # If that fails, convert to string
#                     clean_df[col] = clean_df[col].astype(str)

#         # Replace with cleaned version
#         adata.uns["ripley_l"] = clean_df

# def _prepare_ripley_uns_for_h5ad(adata) -> None:
#     """
#     Enhanced fix for ripley_l results serialization.
#     Ensures proper data types and structure for H5AD storage.
#     """
#     if "ripley_l" not in adata.uns:
#         return

#     rl = adata.uns.get("ripley_l")
    
#     # Handle case where ripley_l might be a string (corrupted data)
#     if isinstance(rl, str):
#         print(f"Warning: ripley_l data appears to be corrupted (string): {rl[:100]}...")
#         # Try to reconstruct or remove corrupted data
#         del adata.uns["ripley_l"]
#         print("Removed corrupted ripley_l data from adata.uns")
#         return
    
#     # Handle DataFrame case
#     if isinstance(rl, pd.DataFrame):
#         # Create a copy to avoid modifying the original
#         clean_df = rl.copy()

#         # Process each object column
#         for col in clean_df.columns:
#             if clean_df[col].dtype == "object":
#                 # Check if column contains dictionaries or other complex objects
#                 sample_val = clean_df[col].iloc[0] if len(clean_df) > 0 else None
                
#                 if isinstance(sample_val, dict):
#                     # Convert dict columns to JSON strings for H5AD compatibility
#                     clean_df[col] = clean_df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))
#                 else:
#                     # Try to convert to numeric first
#                     try:
#                         clean_df[col] = pd.to_numeric(clean_df[col], errors='raise')
#                     except (ValueError, TypeError):
#                         # If that fails, convert to string
#                         clean_df[col] = clean_df[col].astype(str)

#         # Replace with cleaned version
#         adata.uns["ripley_l"] = clean_df
#         print(f"Cleaned ripley_l DataFrame with shape {clean_df.shape}")
        
#     # Handle dictionary case
#     elif isinstance(rl, dict):
#         clean_dict = {}
#         for key, value in rl.items():
#             if isinstance(value, (pd.DataFrame, pd.Series)):
#                 # Convert pandas objects to dictionaries
#                 clean_dict[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
#             elif isinstance(value, (list, tuple)):
#                 # Ensure lists contain serializable types
#                 clean_dict[key] = [str(item) if not isinstance(item, (int, float, str, bool)) else item for item in value]
#             elif isinstance(value, (np.ndarray,)):
#                 # Convert numpy arrays to lists
#                 clean_dict[key] = value.tolist()
#             else:
#                 clean_dict[key] = value
        
#         adata.uns["ripley_l"] = clean_dict
#         print(f"Cleaned ripley_l dictionary with keys: {list(clean_dict.keys())}")
    
#     else:
#         print(f"Warning: Unexpected ripley_l data type: {type(rl)}")
#         # Convert to string representation as fallback
#         adata.uns["ripley_l"] = str(rl)

def _prepare_ripley_uns_for_h5ad(adata) -> None:
    """
    Fix for ripley_l results serialization.
    The ripley_l function stores results as dictionaries with phenotype-specific keys.
    This function ensures all data is properly serializable for H5AD format.
    """
    # Check all keys in adata.uns that start with "ripley_l_"
    ripley_keys = [key for key in adata.uns.keys() if key.startswith("ripley_l_")]
    
    for key in ripley_keys:
        data = adata.uns[key]
        
        if isinstance(data, dict):
            # Clean the dictionary to ensure all values are serializable
            clean_dict = {}
            for k, v in data.items():
                if isinstance(v, pd.DataFrame):
                    # Convert DataFrame to dict
                    clean_dict[k] = v.to_dict()
                elif isinstance(v, pd.Series):
                    # Convert Series to list
                    clean_dict[k] = v.tolist()
                elif isinstance(v, np.ndarray):
                    # Convert numpy array to list
                    clean_dict[k] = v.tolist()
                elif hasattr(v, '__array__'):
                    # Convert array-like objects to list
                    clean_dict[k] = np.asarray(v).tolist()
                else:
                    # Keep as is if it's already serializable
                    clean_dict[k] = v
            
            adata.uns[key] = clean_dict
            print(f"Cleaned {key} for H5AD storage")
        
        elif isinstance(data, pd.DataFrame):
            # If the ripley result is stored as a DataFrame, remove it
            # H5AD cannot handle DataFrames with mixed types properly
            print(f"Warning: Removing DataFrame {key} from adata.uns as it cannot be serialized to H5AD")
            del adata.uns[key]


def _make_serializable(obj):
    """Convert an object to a serializable format."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(obj)  # Convert to string representation
    elif isinstance(obj, dict):
        # Recursively clean dictionary
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively clean list/tuple
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__array__'):
        return np.asarray(obj).tolist()
    else:
        # Fallback to string representation
        return str(obj)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Execute Ripley-L analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary

    Returns
    -------
    dict
        Dictionary of saved file paths
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    radii = params["Radii"]
    annotation = params["Annotation"]
    phenotypes = [params["Center_Phenotype"], params["Neighbor_Phenotype"]]
    regions = params.get("Stratify_By", "None")
    n_simulations = params.get("Number_of_Simulations", 1)
    area = params.get("Area", "None")
    seed = params.get("Seed", 42)
    spatial_key = params.get("Spatial_Key", "spatial")
    edge_correction = params.get("Edge_Correction", True)

    # Process parameters
    regions = text_to_value(
        regions,
        default_none_text="None"
    )

    area = text_to_value(
        area,
        default_none_text="None",
        value_to_convert_to=None,
        to_float=True,
        param_name='Area'
    )

    # Convert radii to floats
    radii = _convert_to_floats(radii)

    # Run the analysis
    ripley_l(
        adata,
        annotation=annotation,
        phenotypes=phenotypes,
        distances=radii,
        regions=regions,
        n_simulations=n_simulations,
        area=area,
        seed=seed,
        spatial_key=spatial_key,
        edge_correction=edge_correction
    )
    
    # Fix ripley_l results before saving
    _prepare_ripley_uns_for_h5ad(adata)

    # Save outputs
    outfile = params.get("Output_File", "transform_output.h5ad")
    saved_files = save_outputs({outfile: adata})

    print(f"Ripley-L completed → {saved_files[outfile]}")
    print(adata)
    return saved_files


def _convert_to_floats(text_list: List[Any]) -> List[float]:
    """
    Convert list of text values to floats.
    Exact copy from NIDAP template.

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
            msg = f"Failed to convert the radius: '{value}' to float."
            raise ValueError(msg)
    return float_list


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ripley_l_template.py <params.json>")
        sys.exit(1)

    saved_files = run_from_json(sys.argv[1])

    print("\nOutput files:")
    for filename, filepath in saved_files.items():
        print(f"  {filename}: {filepath}")
