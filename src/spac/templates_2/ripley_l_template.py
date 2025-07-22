"""
Platform-agnostic Ripley L template - Final version.
Matches NIDAP functionality while being standalone.
"""
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spac.spatial_analysis import ripley_l
from spac.templates_2.template_utils import (
    load_input,
    save_output,
    parse_params,
    text_to_value,
)


def run_from_json(json_path: Union[str, Path, Dict[str, Any]]) -> str:
    """
    Execute Ripley L analysis with parameters from JSON.
    
    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file or parameter dictionary
    
    Returns
    -------
    str
        Path to saved output file
    """
    # Parse parameters from JSON
    params = parse_params(json_path)
    
    # Load the upstream analysis data
    adata = load_input(params)
    print(f"Loaded {adata.n_obs} cells")
    
    # Extract parameters (matching NIDAP template exactly)
    radii = params["Radii"]
    annotation = params["Annotation"]
    phenotypes = [params["Center_Phenotype"], params["Neighbor_Phenotype"]]
    regions = params.get("Stratify_By", "None")
    n_simulations = params.get("Number_of_Simulations", 100)
    area = params.get("Area", "None")
    seed = params.get("Seed", 42)
    spatial_key = params.get("Spatial_Key", "spatial")
    edge_correction = params.get("Edge_Correction", True)
    
    # Process parameters exactly as NIDAP template
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
    def convert_to_floats(text_list):
        float_list = []
        for value in text_list:
            try:
                float_list.append(float(value))
            except ValueError:
                raise ValueError(f"Failed to convert the radius: '{value}' to float.")
        return float_list
    
    radii = convert_to_floats(radii)
    
    # Run the analysis
    print(f"Running Ripley L analysis: {phenotypes[0]} vs {phenotypes[1]}")
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
    
    # Save output (automatically handles pickle vs h5ad)
    output_path = save_output(adata, params)
    
    print(f"Ripley L completed → {output_path}")
    print(adata)
    
    return output_path


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ripley_l_template.py <params.json>")
        sys.exit(1)
    
    output_path = run_from_json(sys.argv[1])
    print(f"\nOutput saved to: {output_path}")