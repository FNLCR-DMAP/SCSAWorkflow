"""
Platform-agnostic UMAP transformation template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.umap_transformation_template import run_from_json
>>> run_from_json("examples/umap_transformation_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import SPAC functions from NIDAP template
from spac.transformations import run_umap
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
    Execute UMAP transformation analysis with parameters from JSON.
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

    # Extract parameters - Note: HPC parameters are ignored in SPAC version
    n_neighbors = params.get("Number_of_Neighbors", 75)
    min_dist = params.get("Minimum_Distance_between_Points", 0.1)
    n_components = params.get("Target_Dimension_Number", 2)
    metric = params.get("Computational_Metric", "euclidean")
    random_state = params.get("Random_State", 0)
    transform_seed = params.get("Transform_Seed", 42)
    layer = params.get("Table_to_Process", "Original")

    if layer == "Original":
        layer = None

    updated_dataset = run_umap(
        adata=adata,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        transform_seed=transform_seed,
        layer=layer,
        verbose=True
    )

    # Print adata info as in NIDAP
    print(adata)

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        # Note: Following NIDAP pattern, save the transformed object
        # (which is the same as adata if run_umap modifies in-place)
        saved_files = save_outputs({output_file: updated_dataset})
        
        print(f"UMAP transformation completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            f"Usage: python umap_transformation_template.py <params.json>",
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