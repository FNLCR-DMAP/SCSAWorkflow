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
from typing import Any, Dict, Union, List, Optional
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
    json_path: Union[str, Path, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Execute Visualize Ripley L analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.
    
    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    
    Returns
    -------
    dict
        Dictionary of saved file paths
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

    import numpy as np
    # Define which items belong in which first-layer group
    RIPLEY_L_FIRST_LAYER_KEYS = {
        'center_phenotype',
        'neighbor_phenotype',
        'region_cells',
        'n_simulations',
        'seed',
        'message',
        'region',
    }

    def nested_insert(d, keys, value):
        """Insert a value into nested dictionary using a list of keys."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def restructure_and_convert_grouped(flat_uns_dict):
        """Restructure adata.uns ripley_l_* keys into categorized nested structure."""
        nested = {}

        for full_key, value in flat_uns_dict.items():
            if not full_key.startswith("ripley_l"):
                continue

            suffix = full_key[len("ripley_l-"):]  # remove ripley_l_ prefix
            # Convert ndarray to safe type
            if isinstance(value, np.ndarray):
                if value.ndim == 1:
                    value = pd.Series(value)
                elif value.ndim == 2:
                    value = pd.DataFrame(value)
                else:
                    raise ValueError(f"Unsupported ndarray shape: {value.shape} for key '{full_key}'")

            if suffix in RIPLEY_L_FIRST_LAYER_KEYS:
                nested[suffix] = value
            else:
                # Use the first "-" to split: e.g., 0-bins → ['0', 'bins']
                if "-" in suffix:
                    first, rest = suffix.split("-", 1)
                    nested.setdefault("ripley_l", {})
                    nested_insert(nested["ripley_l"], [first, rest], value)
                else:
                    # Unexpected format, put directly under 'ripley_l'
                    nested.setdefault("ripley_l", {})
                    nested["ripley_l"][suffix] = value

        return nested


    # Collect all keys that start with 'ripley_l_'
    ripley_l_flat = {k: v for k, v in adata.uns.items() if k.startswith("ripley_l-")}
    nested_dict = restructure_and_convert_grouped(ripley_l_flat)
    for key, value in nested_dict.items():
        print(f"Key: {key}, Type: {type(value)}")
        if key == "ripley_l":
            nested_dict["ripley_l"] = pd.DataFrame([value])
    # rows  = flatten_nested_to_df(nested_dict)
    df = pd.DataFrame([nested_dict])
    adata.uns['ripley_l'] = df
    # print(df.head(1))

    # Clean up flat keys
    for k in ripley_l_flat:
        del adata.uns[k]
    
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
    
    # Save outputs
    output_file = params.get("Output_File", "plots.csv")
    saved_files = save_outputs({output_file: plots_df})
    
    print(f"Visualize Ripley L completed → {saved_files[output_file]}")
    return saved_files


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_ripley_template.py <params.json>")
        sys.exit(1)
    
    saved_files = run_from_json(sys.argv[1])
    
    print("\nOutput files:")
    for filename, filepath in saved_files.items():
        print(f"  {filename}: {filepath}")