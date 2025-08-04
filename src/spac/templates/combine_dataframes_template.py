"""
Platform-agnostic Combine DataFrames template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

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
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import SPAC functions from NIDAP template
from spac.data_utils import combine_dfs
from spac.templates.template_utils import (
    save_outputs,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Combine DataFrames analysis with parameters from JSON.
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
        If save_results=False: The combined DataFrame
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the first dataframe
    dataset_A = params["First_Dataframe"]
    if isinstance(dataset_A, pd.DataFrame):
        dataset_A = dataset_A  # Direct DataFrame from previous step
    elif isinstance(dataset_A, (str, Path)):
        path = Path(dataset_A)
        if path.suffix.lower() == '.csv':
            dataset_A = pd.read_csv(path)
        elif path.suffix.lower() in ['.pickle', '.pkl', '.p']:
            with open(path, 'rb') as f:
                dataset_A = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: .csv, .pickle, .pkl, .p"
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
        path = Path(dataset_B)
        if path.suffix.lower() == '.csv':
            dataset_B = pd.read_csv(path)
        elif path.suffix.lower() in ['.pickle', '.pkl', '.p']:
            with open(path, 'rb') as f:
                dataset_B = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: .csv, .pickle, .pkl, .p"
            )
    else:
        raise TypeError(
            f"Second_Dataframe must be DataFrame or file path. "
            f"Got {type(dataset_B)}"
        )

    # Extract parameters
    input_df_lists = [dataset_A, dataset_B]

    print("Information about the first dataset:")
    print(dataset_A.info())
    print("\n\nInformation about the second dataset:")
    print(dataset_B.info())

    combined_dfs = combine_dfs(input_df_lists)
    print("\n\nInformation about the combined dataset:")
    print(combined_dfs.info())

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "combined_dataframes.csv")
        saved_files = save_outputs({output_file: combined_dfs})
        
        print(f"Combine DataFrames completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        print("Returning combined DataFrame (not saving to file)")
        return combined_dfs


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python combine_dataframes_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned combined DataFrame")