"""
Platform-agnostic Append Annotation template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.append_annotation_template import run_from_json
>>> run_from_json("examples/append_annotation_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import append_annotation
from spac.utils import check_column_name
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
    Execute Append Annotation analysis with parameters from JSON.
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

    # Load upstream data - could be DataFrame, CSV, or pickle
    upstream_dataset = params["Upstream_Dataset"]
    if isinstance(upstream_dataset, pd.DataFrame):
        # Direct DataFrame from previous step
        input_dataframe = upstream_dataset
    elif isinstance(upstream_dataset, (str, Path)):
        path = Path(upstream_dataset)
        if path.suffix.lower() == '.csv':
            input_dataframe = pd.read_csv(path)
        elif path.suffix.lower() in ['.pickle', '.pkl', '.p']:
            with open(path, 'rb') as f:
                input_dataframe = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: .csv, .pickle, .pkl"
            )
    else:
        raise TypeError(
            f"Upstream_Dataset must be DataFrame or file path. "
            f"Got {type(upstream_dataset)}"
        )

    # Extract parameters
    dataset_mapping_rules = params.get(
        "Annotation_Pair_List", ["Example:Example"]
    )

    # Initialize an empty dictionary
    parsed_dict = {}

    # Loop through each string pair in the list
    for pair in dataset_mapping_rules:
        # Split the string on the colon
        key, value = pair.split(":")
        check_column_name(key, pair)
        # Add the key-value pair to the dictionary
        parsed_dict[key] = value

    print(f"The pairs to add are:\n{parsed_dict}")

    output_dataframe = append_annotation(
        input_dataframe,
        parsed_dict
    )

    print(output_dataframe.info())

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "append_observations.csv")
        # Ensure CSV extension for DataFrame output
        if not output_file.endswith('.csv'):
            output_file = output_file + '.csv'
        
        saved_files = save_outputs({output_file: output_dataframe})
        
        print(
            f"Append Annotation completed → {saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        print("Returning DataFrame (not saving to file)")
        return output_dataframe


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python append_annotation_template.py <params.json>",
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