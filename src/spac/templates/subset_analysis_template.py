"""
Platform-agnostic Subset Analysis template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.subset_analysis_template import run_from_json
>>> run_from_json("examples/subset_analysis_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List
import pandas as pd
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import SPAC functions from NIDAP template
from spac.data_utils import select_values
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
    Execute Subset Analysis with parameters from JSON.
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
    # Use direct dictionary access for required parameters (NIDAP style)
    annotation = params["Annotation_of_interest"]
    labels = params["Labels"]
    
    # Use .get() with defaults for optional parameters from JSON template
    toggle = params.get("Include_Exclude", "Include Selected Labels")

    if toggle == "Include Selected Labels":
        values_to_include = labels
        values_to_exclude = None
    else:
        values_to_include = None
        values_to_exclude = labels

    with warnings.catch_warnings(record=True) as caught_warnings:
        filtered_adata = select_values(
            data=adata,
            annotation=annotation,
            values=values_to_include,
            exclude_values=values_to_exclude
            )
        if caught_warnings:
            for warning in caught_warnings:
                raise ValueError(warning.message)
    
    print(filtered_adata)
    print("\n")

    # Count and display occurrences of each label in the annotation
    label_counts = filtered_adata.obs[annotation].value_counts()
    print(label_counts)
    print("\n")

    dataframe = pd.DataFrame(
        filtered_adata.X, 
        columns=filtered_adata.var.index, 
        index=filtered_adata.obs.index
    )
    print(dataframe.describe())
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: filtered_adata})
        
        print(f"Subset Analysis completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return filtered_adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python subset_analysis_template.py <params.json>",
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