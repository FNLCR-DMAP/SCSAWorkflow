"""
Platform-agnostic Load CSV Files template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.load_csv_files_with_config import run_from_json
>>> run_from_json("examples/load_csv_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.templates.template_utils import (
    save_outputs,
    parse_params,
    text_to_value,
    load_csv_files
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Load CSV Files analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the DataFrame
        directly for in-memory workflows. Default is True.

    Returns
    -------
    dict or DataFrame
        If save_results=True: Dictionary of saved file paths
        If save_results=False: The processed DataFrame
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Extract parameters
    csv_dir = Path(params["CSV_Files"])
    files_config = pd.read_csv(params["CSV_Files_Configuration"])
    string_columns = params.get("String_Columns", [""])

    # Load and combine CSV files
    final_df = load_csv_files(
        csv_dir=csv_dir,
        files_config=files_config,
        string_columns=string_columns
    )

    if final_df is None:
        raise RuntimeError("Failed to process CSV files")

    print("Load CSV Files completed successfully.")

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "combined_data.csv")
        saved_files = save_outputs({output_file: final_df})

        logging.info(f"Load CSV completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the DataFrame directly for in-memory workflows
        logging.info("Returning DataFrame (not saving to file)")
        return final_df


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python load_csv_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    saved_files = run_from_json(sys.argv[1])

    if isinstance(saved_files, dict):
        print("\nOutput files:")
        for filename, filepath in saved_files.items():
            print(f"  {filename}: {filepath}")