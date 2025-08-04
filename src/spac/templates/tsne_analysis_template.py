"""
Platform-agnostic tSNE Analysis template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.tsne_analysis_template import run_from_json
>>> run_from_json("examples/tsne_analysis_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import tsne
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
    Execute tSNE Analysis analysis with parameters from JSON.
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
    # Select layer to perform tSNE
    Layer_to_Analysis = params.get("Table_to_Process", "Original")

    print(all_data)
    if Layer_to_Analysis == "Original":
        Layer_to_Analysis = None

    print("tSNE Layer: \n", Layer_to_Analysis)

    print("Performing tSNE ...")

    tsne(all_data, layer=Layer_to_Analysis)

    print("tSNE Done!")
    
    print(all_data)

    object_to_output = all_data
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: object_to_output})
        
        print(f"tSNE Analysis completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return object_to_output


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python tsne_analysis_template.py <params.json>",
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