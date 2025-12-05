"""
Platform-agnostic Combine DataFrames template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.combine_dataframes_template import run_from_json
>>> run_from_json("examples/combine_dataframes_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import combine_dfs
from spac.templates.template_utils import (
    save_results,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Combine DataFrames analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "First_Dataframe": "path/to/first.csv",
            "Second_Dataframe": "path/to/second.csv",
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
                "dataframe": "path/to/dataframe.csv"
            }
        If save_to_disk=False: The combined DataFrame

    Notes
    -----
    Output Structure:
    - DataFrame is saved as a single CSV file
    - When save_to_disk=False, the DataFrame is returned for programmatic use
    
    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["dataframe"])  # Path to saved CSV file
    
    >>> # Get results in memory
    >>> combined_df = run_from_json("params.json", save_to_disk=False)
    
    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
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

    # Load the first dataframe
    dataset_A = params["First_Dataframe"]
    if isinstance(dataset_A, pd.DataFrame):
        dataset_A = dataset_A  # Direct DataFrame from previous step
    elif isinstance(dataset_A, (str, Path)):
        # Galaxy passes .dat files, but they contain CSV data
        # Don't check extension - directly read as CSV
        path = Path(dataset_A)
        try:
            dataset_A = pd.read_csv(path)
            logging.info(f"Successfully loaded first DataFrame from: {path}")
        except Exception as e:
            raise ValueError(
                f"Failed to read CSV data from '{path}'. "
                f"This tool expects CSV/tabular format. "
                f"Error: {str(e)}"
            )
    else:
        raise TypeError(
            f"First_Dataframe must be DataFrame or file path. "
            f"Got {type(dataset_A)}"
        )

    # Load the second dataframe
    dataset_B = params["Second_Dataframe"]
    if isinstance(dataset_B, pd.DataFrame):
        dataset_B = dataset_B  # Direct DataFrame from previous step
    elif isinstance(dataset_B, (str, Path)):
        # Galaxy passes .dat files, but they contain CSV data
        # Don't check extension - directly read as CSV
        path = Path(dataset_B)
        try:
            dataset_B = pd.read_csv(path)
            logging.info(f"Successfully loaded second DataFrame from: {path}")
        except Exception as e:
            raise ValueError(
                f"Failed to read CSV data from '{path}'. "
                f"This tool expects CSV/tabular format. "
                f"Error: {str(e)}"
            )
    else:
        raise TypeError(
            f"Second_Dataframe must be DataFrame or file path. "
            f"Got {type(dataset_B)}"
        )

    # Extract parameters
    input_df_lists = [dataset_A, dataset_B]

    logging.info("Information about the first dataset:")
    logging.info(dataset_A.info())
    logging.info("\n\nInformation about the second dataset:")
    logging.info(dataset_B.info())

    combined_dfs = combine_dfs(input_df_lists)
    logging.info("\n\nInformation about the combined dataset:")
    logging.info(combined_dfs.info())

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for dataframe output
        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = combined_dfs
        
        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info("Combine DataFrames completed successfully.")
        return saved_files
    else:
        # Return the DataFrame directly for in-memory workflows
        logging.info("Returning combined DataFrame for in-memory use")
        return combined_dfs


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python combine_dataframes_template.py <params.json> [output_dir]",
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
    
    # Run analysis
    result = run_from_json(
        json_path=sys.argv[1],
        output_dir=output_dir
    )

    # Display results based on return type
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
        print("\nReturned combined DataFrame")
        print(f"DataFrame shape: {result.shape}")
