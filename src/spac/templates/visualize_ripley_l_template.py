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
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import plot_ripley_l
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    show_plot: bool = True,
    output_dir: Optional[Union[str, Path]] = None
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Visualize Ripley L analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_to_disk : bool, optional
        Whether to save results to file. If False, returns the figure and
        dataframe directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    output_dir : str or Path, optional
        Directory for outputs. If None, uses current directory.

    Returns
    -------
    dict or tuple
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: Tuple of (figure, dataframe)
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

    logging.info(f"Running with center_phenotype: {center_phenotype}, neighbor_phenotype: {neighbor_phenotype}")

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

    if show_plot:
        plt.show()

    # Print the dataframe to console
    logging.info(f"\n{plots_df.to_string()}")

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for dataframe output in config
        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = plots_df
        
        # Add figure if configured (usually not in the original template)
        # but we can add it as an enhancement
        if "figures" in params.get("outputs", {}):
            # Package figure in a dictionary for directory saving
            results_dict["figures"] = {"ripley_l_plot": fig}
        
        # Add analysis output if in config (for compatibility)
        if "analysis" in params.get("outputs", {}):
            results_dict["analysis"] = adata
        
        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        logging.info(f"Visualize Ripley L completed → {list(saved_files.keys())}")
        return saved_files
    else:
        # Return the figure and dataframe directly for in-memory workflows
        logging.info("Returning figure and dataframe (not saving to file)")
        return fig, plots_df


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_ripley_template.py <params.json>", file=sys.stderr)
        sys.exit(1)

    # Set up logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get output directory if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run analysis
    result = run_from_json(sys.argv[1], output_dir=output_dir)

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure and dataframe")
