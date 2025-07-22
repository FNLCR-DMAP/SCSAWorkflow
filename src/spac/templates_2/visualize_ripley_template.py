"""
Platform-agnostic Visualize Ripley L template - Final version.
Matches NIDAP functionality while being standalone.
"""
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spac.visualization import plot_ripley_l
from spac.templates_2.template_utils import (
    load_input,
    save_output,
    parse_params,
)


def run_from_json(json_path: Union[str, Path, Dict[str, Any]]) -> str:
    """
    Execute Visualize Ripley L analysis with parameters from JSON.
    
    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file or parameter dictionary
    
    Returns
    -------
    str
        Path to saved CSV file
    """
    # Parse parameters from JSON
    params = parse_params(json_path)
    
    # Load the upstream analysis data
    adata = load_input(params)
    
    # Extract parameters (matching NIDAP template exactly)
    center_phenotype = params["Center_Phenotype"]
    neighbor_phenotype = params["Neighbor_Phenotype"]
    plot_specific_regions = params.get("Plot_Specific_Regions", False)
    regions_labels = params.get("Regions_Label_s_", [])
    plot_simulations = params.get("Plot_Simulations", True)
    
    # Process regions parameter exactly as in NIDAP template
    if plot_specific_regions:
        if len(regions_labels) == 0:
            raise ValueError(
                'Please identify at least one region in the '
                '"Regions Label(s) parameter'
            )
    else:
        regions_labels = None
    
    # Run the visualization exactly as in NIDAP template
    fig, plots_df = plot_ripley_l(
        adata,
        phenotypes=(center_phenotype, neighbor_phenotype),
        regions=regions_labels,
        sims=plot_simulations,
        return_df=True
    )
    
    plt.show()
    
    # Print the dataframe to console
    print(plots_df.to_string())
    
    # Save output
    output_file = params.get("Output_File", "plots.csv")
    output_path = save_output(plots_df, output_file, format='csv')
    
    print(f"Visualize Ripley L completed → {output_path}")
    
    return output_path


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_ripley_template.py <params.json>")
        sys.exit(1)
    
    output_path = run_from_json(sys.argv[1])
    print(f"\nOutput saved to: {output_path}")