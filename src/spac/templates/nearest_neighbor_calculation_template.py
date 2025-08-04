"""
Platform-agnostic Nearest Neighbor Calculation template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.nearest_neighbor_calculation_template import (
...     run_from_json
... )
>>> run_from_json("examples/nearest_neighbor_calculation_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.utils import check_table, check_annotation
from spac.spatial_analysis import calculate_nearest_neighbor
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
    Execute Nearest Neighbor Calculation analysis with parameters from JSON.
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
    annotation = params["Annotation"]
    spatial_associated_table = "spatial"
    imageid = params.get("ImageID", "None")
    label = params.get(
        "Nearest_Neighbor_Associated_Table", "spatial_distance"
    )
    verbose = params.get("Verbose", True)

    # Convert any string "None" to actual None for Python
    imageid = text_to_value(imageid, default_none_text="None")

    print(
        "Running `calculate_nearest_neighbor` with the following parameters:"
    )
    print(f"  annotation: {annotation}")
    print(f"  spatial_associated_table: {spatial_associated_table}")
    print(f"  imageid: {imageid}")
    print(f"  label: {label}")
    print(f"  verbose: {verbose}")

    # Perform the nearest neighbor calculation
    calculate_nearest_neighbor(
        adata=adata,
        annotation=annotation,
        spatial_associated_table=spatial_associated_table,
        imageid=imageid,
        label=label,
        verbose=verbose
    )

    print("Nearest neighbor calculation complete.")
    print("adata.obsm keys:", list(adata.obsm.keys()))
    if label in adata.obsm:
        print(
            f"Preview of adata.obsm['{label}']:\n",
            adata.obsm[label].head()
        )

    print(adata)

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: adata})
        
        print(
            f"Nearest Neighbor Calculation completed → "
            f"{saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python nearest_neighbor_calculation_template.py "
            "<params.json>",
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