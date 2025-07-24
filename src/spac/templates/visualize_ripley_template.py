"""
Platform-agnostic Visualize Ripley L template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.visualize_ripley_template import run_from_json
>>> run_from_json("examples/visualize_ripley_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import plot_ripley_l
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Visualize Ripley L analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.
    
    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the figure and
        dataframe directly for in-memory workflows. Default is True.
    
    Returns
    -------
    dict or tuple
        If save_results=True: Dictionary of saved file paths
        If save_results=False: Tuple of (figure, dataframe)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)
    
    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])
    
    # Extract parameters
    center_phenotype = params["Center_Phenotype"]
    neighbor_phenotype = params["Neighbor_Phenotype"]
    plot_specific_regions = params.get("Plot_Specific_Regions", False)
    regions_labels = params.get("Regions_Labels", [])
    plot_simulations = params.get("Plot_Simulations", True)

    print(f"running with center_phenotype: {center_phenotype}, neighbor_phenotype: {neighbor_phenotype}")

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
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "plots.csv")
        saved_files = save_outputs({output_file: plots_df})
        
        print(f"Visualize Ripley L completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the figure and dataframe directly for in-memory workflows
        print("Returning figure and dataframe (not saving to file)")
        return fig, plots_df


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_ripley_template.py <params.json>")
        sys.exit(1)
    
    result = run_from_json(sys.argv[1])
    
    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure and dataframe")