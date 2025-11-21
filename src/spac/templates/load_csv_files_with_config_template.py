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
    save_results,
    parse_params,
    text_to_value,
    load_csv_files
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Load CSV Files analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "CSV_Files": "path/to/csv/directory",
            "CSV_Files_Configuration": "path/to/config.csv",
            "String_Columns": ["column1", "column2"],
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves the combined DataFrame
        to a CSV file. If False, returns the DataFrame directly for in-memory 
        workflows. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory. All outputs will be saved relative to this directory.

    Returns
    -------
    dict or DataFrame
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {
                "dataframe": "path/to/dataframe.csv"  # Single file path
            }
        If save_to_disk=False: The processed DataFrame

    Notes
    -----
    Output Structure:
    - The combined CSV data is saved as a single CSV file
    - When save_to_disk=False, DataFrame is returned for programmatic use
    
    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["dataframe"])  # Path to saved CSV file
    
    >>> # Get DataFrame directly without saving
    >>> df = run_from_json("params.json", save_to_disk=False)
    >>> print(df.shape)
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

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for dataframe output in the outputs configuration
        if "dataframe" in params.get("outputs", {}):
            results_dict["dataframe"] = final_df
        else:
            # Fallback for backward compatibility
            # Use Output_File if outputs config is not present
            output_key = params.get("Output_File", "combined_data.csv")
            # Remove .csv extension if present for the key
            output_key = output_key.replace('.csv', '')
            results_dict[output_key] = final_df
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info(f"Load CSV Files completed → {list(saved_files.values())[0]}")
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
