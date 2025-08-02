"""
Platform-agnostic Arcsinh Normalization template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.arcsinh_normalization_template import run_from_json
>>> run_from_json("examples/arcsinh_normalization_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import arcsinh_transformation
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
    Execute Arcsinh Normalization analysis with parameters from JSON.
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
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    input_layer = params.get("Table_to_Process", "Original")
    co_factor = params.get("Co_Factor", "5.0")
    percentile = params.get("Percentile", "None")
    output_layer = params.get("Output_Table_Name", "arcsinh")
    per_batch = params.get("Per_Batch", "False")
    annotation = params.get("Annotation", "None")

    input_layer = text_to_value(
        input_layer,
        default_none_text="Original"
    )

    co_factor = text_to_value(
        co_factor,
        default_none_text="None",
        to_float=True,
        param_name="co_factor"
    )

    percentile = text_to_value(
        percentile,
        default_none_text="None",
        to_float=True,
        param_name="percentile"
    )

    if per_batch == "True":
        per_batch = True
    else:
        per_batch = False

    annotation = text_to_value(
        annotation,
        default_none_text="None"
    )

    transformed_data = arcsinh_transformation(
        adata,
        input_layer=input_layer,
        co_factor=co_factor,
        percentile=percentile,
        output_layer=output_layer,
        per_batch=per_batch,
        annotation=annotation
        )
    
    print(f"Transformed data stored in layer: {output_layer}")
    dataframe = pd.DataFrame(transformed_data.layers[output_layer])
    print(dataframe.describe())
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: transformed_data})
        
        print(f"Arcsinh Normalization completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return transformed_data


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python arcsinh_normalization_template.py <params.json>",
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