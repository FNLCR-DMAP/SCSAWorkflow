"""
Platform-agnostic Ripley-L template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.ripley_l_template import run_from_json
>>> run_from_json("examples/ripley_l_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List
# import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.spatial_analysis import ripley_l
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
    convert_to_floats
)
   

def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], Any]:
    """
    Execute Ripley-L analysis with parameters from JSON.
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
    radii = params["Radii"]
    annotation = params["Annotation"]
    phenotypes = [params["Center_Phenotype"], params["Neighbor_Phenotype"]]
    regions = params.get("Stratify_By", "None")
    n_simulations = params.get("Number_of_Simulations", 100)
    area = params.get("Area", "None")
    seed = params.get("Seed", 42)
    spatial_key = params.get("Spatial_Key", "spatial")
    edge_correction = params.get("Edge_Correction", True)

    # Process parameters
    regions = text_to_value(
        regions,
        default_none_text="None"
    )

    area = text_to_value(
        area,
        default_none_text="None",
        value_to_convert_to=None,
        to_float=True,
        param_name='Area'
    )

    # Convert radii to floats
    radii = convert_to_floats(radii)

    # Run the analysis
    ripley_l(
        adata,
        annotation=annotation,
        phenotypes=phenotypes,
        distances=radii,
        regions=regions,
        n_simulations=n_simulations,
        area=area,
        seed=seed,
        spatial_key=spatial_key,
        edge_correction=edge_correction
    )

    print("Ripley-L analysis completed successfully.")

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        outfile = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if not specified
        if not outfile.endswith(('.pickle', '.pkl', '.h5ad')):
            outfile = outfile.replace('.h5ad', '.pickle')

        print(type(outfile))
        saved_files = save_outputs({outfile: adata})
        print(saved_files)

        print(f"Ripley-L completed → {str(saved_files[outfile])}")
        print(adata)
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ripley_l_template.py <params.json>")
        sys.exit(1)

    saved_files = run_from_json(sys.argv[1])

    if isinstance(saved_files, dict):
        print("\nOutput files:")
        for filename, filepath in saved_files.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned AnnData object")
