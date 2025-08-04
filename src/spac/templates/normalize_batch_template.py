"""
Platform-agnostic Normalize Batch template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.normalize_batch_template import run_from_json
>>> run_from_json("examples/normalize_batch_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import batch_normalize
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], Any]:
    """
    Execute Normalize Batch analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the adata object
        directly for in-memory workflows. Default is True.

    Returns
    -------
    dict or AnnData
        If save_results=True: Dictionary of saved file paths
        If save_results=False: The processed AnnData object
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    all_data = load_input(params["Upstream_Analysis"])

    # Extract parameters
    annotation = params["Annotation"]
    input_layer = params.get("Input_Table_Name", "Original")

    if input_layer == 'Original':
        input_layer = None

    output_layer = params.get("Output_Table_Name", "batch_normalized_table")
    method = params.get("Normalization_Method", "median")
    take_log = params.get("Take_Log", False)
       
    need_normalization = params.get("Need_Normalization", False)
    if need_normalization:
        batch_normalize(
            adata=all_data,
            annotation=annotation,
            input_layer=input_layer,
            output_layer=output_layer,
            method=method,
            log=take_log
        )

        print("Statistics of original data:\n", all_data.to_df().describe())
        print("Statistics of layer data:\n", all_data.to_df(layer=output_layer).describe())
    else:
        print("Statistics of original data:\n", all_data.to_df().describe())
    
    print("Current Analysis contains:\n", all_data)
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: all_data})
        
        print(f"Normalize Batch completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return all_data


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python normalize_batch_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned AnnData object")