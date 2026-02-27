"""
Platform-agnostic Analysis to CSV template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.analysis_to_csv_template import run_from_json
>>> run_from_json("examples/analysis_to_csv_params.json")
"""
import json
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.utils import check_table
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)

logger = logging.getLogger(__name__)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Analysis to CSV analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Table_to_Export": "Original",
            "Save_as_CSV_File": false,
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the dataframe
        directly for in-memory workflows. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory.

    Returns
    -------
    dict or DataFrame
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {"dataframe": "path/to/dataframe.csv"}
        If save_to_disk=False: The processed DataFrame

    Notes
    -----
    Output Structure:
    - DataFrame is saved as a CSV file when save_to_disk is True 
    - Otherwise, the DataFrame is returned for programmatic use
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    # Ensure outputs configuration exists with standardized defaults
    if "outputs" not in params:
        params["outputs"] = {
            "dataframe": {"type": "file", "name": "dataframe.csv"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    input_layer = params.get("Table_to_Export", "Original")

    if input_layer == "Original":
        input_layer = None

    def export_layer_to_csv(adata, layer=None):
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

        return full_data_df

    csv_data = export_layer_to_csv(
        adata=adata,
        layer=input_layer
    )

    logger.info(f"Exported DataFrame shape: {csv_data.shape}")

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}

        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = csv_data

        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        logger.info("Analysis to CSV completed successfully.")
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        logger.info("Returning DataFrame for in-memory use")
        return csv_data


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python analysis_to_csv_template.py <params.json> "
            "[output_dir]",
            file=sys.stderr
        )
        sys.exit(1)

    # Set up logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get output directory if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_from_json(
        json_path=sys.argv[1],
        output_dir=output_dir
    )

    if isinstance(result, dict):
        print("\nOutput files:")
        for key, paths in result.items():
            if isinstance(paths, list):
                print(f"  {key}:")
                for path in paths:
                    print(f"    - {path}")
            else:
                print(f"  {key}: {paths}")
    else:
        print("\nReturned DataFrame")
        print(f"DataFrame shape: {result.shape}")
