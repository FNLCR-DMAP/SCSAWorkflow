"""
Platform-agnostic tSNE Analysis template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

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
    save_results,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], Any]:
    """
    Execute tSNE Analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Table_to_Process": "Original",
            "outputs": {
                "analysis": {"type": "file", "name": "output.pickle"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the AnnData object
        directly for in-memory workflows. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory.

    Returns
    -------
    dict or AnnData
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {"analysis": "path/to/output.pickle"}
        If save_to_disk=False: The processed AnnData object

    Notes
    -----
    Output Structure:
    - Analysis output is saved as a single pickle file (standardized for analysis outputs)
    - When save_to_disk=False, the AnnData object is returned for programmatic use

    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["analysis"])  # Path to saved pickle file

    >>> # Get results in memory
    >>> adata = run_from_json("params.json", save_to_disk=False)

    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    # Ensure outputs configuration exists with standardized defaults
    # Analysis uses file type per standardized schema
    if "outputs" not in params:
        params["outputs"] = {
            "analysis": {"type": "file", "name": "output.pickle"}
        }

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

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary
        results_dict = {}
        if "analysis" in params["outputs"]:
            results_dict["analysis"] = object_to_output

        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        print(f"tSNE Analysis completed → {saved_files['analysis']}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return object_to_output


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python tsne_analysis_template.py <params.json> [output_dir]",
            file=sys.stderr
        )
        sys.exit(1)

    # Get output directory if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_from_json(
        json_path=sys.argv[1],
        output_dir=output_dir
    )

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned AnnData object")
