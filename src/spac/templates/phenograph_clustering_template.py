"""
Platform-agnostic Phenograph Clustering template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Follows standardized output schema where analysis is saved as a file.

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
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], Any]:
    """
    Execute Phenograph Clustering analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Table_to_Process": "Original",
            "K_Nearest_Neighbors": 30,
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
    >>> # './output.pickle'
    
    >>> # Get results in memory
    >>> adata = run_from_json("params.json", save_to_disk=False)
    >>> # Can now work with adata object directly
    
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
    
    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary
        results_dict = {}
        if "analysis" in params["outputs"]:
            results_dict["analysis"] = adata
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        print(
            f"Phenograph Clustering completed → "
            f"{saved_files['analysis']}"
        )
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python phenograph_clustering_template.py <params.json> [output_dir]",
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
