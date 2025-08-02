"""
Platform-agnostic UTAG Clustering template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.utag_clustering_template import run_from_json
>>> run_from_json("examples/utag_clustering_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import run_utag_clustering
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
    Execute UTAG Clustering analysis with parameters from JSON.
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
    layer = params.get("Table_to_Process", "Original")
    features = params.get("Features", ["All"])
    slide = params.get("Slide_Annotation", "None")
    Distance_threshold = params.get("Distance_Threshold", 20.0)
    K_neighbors = params.get("K_Nearest_Neighbors", 15)
    resolution = params.get("Resolution_Parameter", 1)
    principal_components = params.get("PCA_Components", "None")
    random_seed = params.get("Random_Seed", 42)
    n_jobs = params.get("N_Jobs", 1)
    N_iterations = params.get("Leiden_Iterations", 5)
    Parallel_processes = params.get("Parellel_Processes", False)
    output_annotation = params.get("Output_Annotation_Name", "UTAG")

    # layer: convert "Original" → None
    layer_arg = None if layer.lower().strip() == "original" else layer

    # features: ["All"] → None, else leave list and print selection
    if isinstance(features, list) and any(
        item == "All" for item in features
    ):
        print("Clustering all features")
        features_arg = None
    else:
        feature_str = "\n".join(features)
        print(f"Clustering features:\n{feature_str}")
        features_arg = features

    # slide: "None" → None
    slide_arg = text_to_value(
        slide,
        default_none_text="None",
        value_to_convert_to=None
    )

    # principal_components: "None" or integer string → None or int
    principal_components_arg = text_to_value(
        principal_components,
        default_none_text="None",
        value_to_convert_to=None,
        to_int=True,
        param_name="principal_components"
    )

    print("\nBefore UTAG Clustering: \n", adata)

    run_utag_clustering(
        adata,
        features=features_arg,
        k=K_neighbors,
        resolution=resolution,
        max_dist=Distance_threshold,
        n_pcs=principal_components_arg,
        random_state=random_seed,
        n_jobs=n_jobs,
        n_iterations=N_iterations,
        slide_key=slide_arg,
        layer=layer_arg,
        output_annotation=output_annotation,
        parallel=Parallel_processes,
    )

    print("\nAfter UTAG Clustering: \n", adata)

    print(
        "\nUTAG Cluster Count: \n",
        len(adata.obs[output_annotation].unique().tolist())
    )

    print(
        "\nUTAG Cluster Names: \n",
        adata.obs[output_annotation].unique().tolist()
    )

    # Count and display occurrences of each label in the annotation
    print(
        f'\nCount of cells in the output annotation:'
        f'"{output_annotation}":'
    )
    label_counts = adata.obs[output_annotation].value_counts()
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
        
        print(f"UTAG Clustering completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python utag_clustering_template.py <params.json>",
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