"""
Platform-agnostic Combine Annotations template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.combine_annotations_template import run_from_json
>>> run_from_json("examples/combine_annotations_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import combine_annotations
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], Any]:
    """
    Execute Combine Annotations analysis with parameters from JSON.
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
    annotations_list = params["Annotations_Names"]
    new_annotation = params.get("New_Annotation_Name", "combined_annotation")
    separator = params.get("Separator", "_")

    combine_annotations(
        adata,
        annotations=annotations_list,
        separator=separator,
        new_annotation_name=new_annotation
    )

    print("After combining annotations: \n", adata)
    value_counts = adata.obs[new_annotation].value_counts(dropna=False)
    print(f"Unique labels in {new_annotation}")
    print(value_counts)

    # create the frequency CSV for download
    df_counts = (
        value_counts
        .rename_axis(new_annotation)   # move index to a column name
        .reset_index(name='count')     # two columns: label | count
    )

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        # Also save the counts CSV
        csv_name = f"{new_annotation}_counts.csv"
        
        saved_files = save_outputs({
            output_file: adata,
            csv_name: df_counts
        })
        
        print(f"\nLabel‑count table written to {csv_name}")
        print(f"Combine Annotations completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python combine_annotations_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned data object")