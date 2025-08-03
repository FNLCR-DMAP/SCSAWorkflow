"""
Platform-agnostic Phenograph Clustering template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.phenograph_clustering_template import run_from_json
>>> run_from_json("examples/phenograph_clustering_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import phenograph_clustering
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
    Execute Phenograph Clustering analysis with parameters from JSON.
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
    Layer_name = params.get("Table_to_Process", "Original")
    K_cluster = params.get("K_Nearest_Neighbors", 30)
    Seed = params.get("Seed", 42)
    resolution_parameter = params.get("Resolution_Parameter", 1.0)
    output_annotation_name = params.get(
        "Output_Annotation_Name", "phenograph"
    )
    # Used only in HPC profiling mode (not implemented in SPAC)
    resolution_list = params.get("Resolution_List", [])
    
    n_iterations = params.get("Number_of_Iterations", 100)

    if Layer_name == "Original":
        Layer_name = None

    intensities = adata.var.index.to_list()

    print("Before Phenograph Clustering: \n", adata)
    
    phenograph_clustering(
        adata=adata, 
        features=intensities, 
        layer=Layer_name,
        k=K_cluster,
        seed=Seed,
        resolution_parameter=resolution_parameter,
        n_iterations=n_iterations
    )
    if output_annotation_name != "phenograph":
        adata.obs = adata.obs.rename(
            columns={'phenograph': output_annotation_name}
        )

    print("After Phenograph Clustering: \n", adata)

    # Count and display occurrences of each label in the annotation
    print(
        f'Count of cells in the output annotation:'
        f'"{output_annotation_name}":'
    )
    label_counts = adata.obs[output_annotation_name].value_counts()
    print(label_counts)
    print("\n")
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: adata})
        
        print(
            f"Phenograph Clustering completed → "
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
            "Usage: python phenograph_clustering_template.py <params.json>",
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