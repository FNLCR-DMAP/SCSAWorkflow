"""
Platform-agnostic Rename Labels template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.rename_labels_template import run_from_json
>>> run_from_json("examples/rename_labels_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import rename_annotations
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
    Execute Rename Labels analysis with parameters from JSON.
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
    rename_list_path = params["Cluster_Mapping_Dictionary"]
    original_column = params.get("Source_Annotation", "None")
    renamed_column = params.get("New_Annotation", "None")

    # Load the mapping dictionary CSV
    rename_list = pd.read_csv(rename_list_path)

    original_column = text_to_value(original_column)
    renamed_column = text_to_value(renamed_column)

    # Create a new dictionary with the desired format

    dict_list = rename_list.to_dict('records')

    mappings = {d['Original']: d['New'] for d in dict_list}

    print("Cluster Name Mapping: \n", mappings)

    rename_annotations(
        all_data, 
        src_annotation=original_column,
        dest_annotation=renamed_column,
        mappings=mappings)

    print("After Renaming Clusters: \n", all_data)

    # Count and display occurrences of each label in the annotation
    print(f'Count of cells in the output annotation:"{renamed_column}":')
    label_counts = all_data.obs[renamed_column].value_counts()
    print(label_counts)
    print("\n")

    object_to_output = all_data
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: object_to_output})
        
        print(f"Rename Labels completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return object_to_output


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python rename_labels_template.py <params.json>",
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