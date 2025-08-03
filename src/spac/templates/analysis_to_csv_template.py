"""
Platform-agnostic Analysis to CSV template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.analysis_to_csv_template import run_from_json
>>> run_from_json("examples/analysis_to_csv_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.utils import check_table
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Analysis to CSV analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the dataframe
        directly for in-memory workflows. Default is True.

    Returns
    -------
    dict or DataFrame
        If save_results=True: Dictionary of saved file paths
        If save_results=False: The processed DataFrame
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    input_layer = params.get("Table_to_Export", "Original")
    save_file = params.get("Save_as_CSV_File", False)

    if input_layer == "Original":
        input_layer = None

    def export_layer_to_csv(
        adata,
        layer=None):
        """
        Exports the specified layer or the default .X data matrix of an 
        AnnData object to a CSV file.
        """
        # Check if the provided layer exists in the AnnData object
        if layer:
            check_table(adata, tables=layer)
            data_to_export = pd.DataFrame(
                adata.layers[layer],
                index=adata.obs.index,
                columns=adata.var.index
            )
        else:
            data_to_export = pd.DataFrame(
                adata.X,
                index=adata.obs.index,
                columns=adata.var.index
            )

        # Join with the observation metadata
        full_data_df = data_to_export.join(adata.obs)

        # Join the spatial coordinates

        # Extract the spatial coordinates
        spatial_df = pd.DataFrame(
            adata.obsm['spatial'],
            index=adata.obs.index,
            columns=['spatial_x', 'spatial_y']
        )

        # Join spatial_df with full_data_df
        full_data_df = full_data_df.join(spatial_df)

        return(full_data_df)

    csv_data = export_layer_to_csv(
        adata=adata,
        layer=input_layer
    )
    
    # Handle results based on save_results flag and save_file parameter
    if save_results and save_file:
        # Save outputs
        output_file = params.get("Output_File", "analysis.csv")
        saved_files = save_outputs({output_file: csv_data})
        
        print(f"Analysis to CSV completed → {saved_files[output_file]}")
        return saved_files
    else:
        print(csv_data.info())
        # Return the dataframe directly for in-memory workflows
        return csv_data


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python analysis_to_csv_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned DataFrame")