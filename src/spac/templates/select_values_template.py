"""
Platform-agnostic Select Values template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.select_values_template import run_from_json
>>> run_from_json("examples/select_values_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import warnings
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import select_values
from spac.templates.template_utils import (
    save_outputs,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Select Values analysis with parameters from JSON.
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
        If save_results=False: The filtered DataFrame
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load upstream data - could be DataFrame, CSV, or pickle
    upstream_dataset = params["Upstream_Dataset"]
    
    if isinstance(upstream_dataset, pd.DataFrame):
        input_dataset = upstream_dataset  # Direct DataFrame from previous step
    elif isinstance(upstream_dataset, (str, Path)):
        path = Path(upstream_dataset)
        if path.suffix.lower() == '.csv':
            input_dataset = pd.read_csv(path)
        elif path.suffix.lower() in ['.pickle', '.pkl', '.p']:
            with open(path, 'rb') as f:
                input_dataset = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: .csv, .pickle, .pkl, .p"
            )
    else:
        raise TypeError(
            f"Upstream_Dataset must be DataFrame or file path. "
            f"Got {type(upstream_dataset)}"
        )

    # Extract parameters
    observation = params["Annotation_of_Interest"]
    values = params["Label_s_of_Interest"]
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        filtered_dataset = select_values(
            data=input_dataset,
            annotation=observation,
            values=values
            )
        if caught_warnings is not None:
            for warning in caught_warnings:
                raise ValueError(warning.message)
    
    print(filtered_dataset.info())

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "select_values.csv")
        saved_files = save_outputs({output_file: filtered_dataset})
        
        print(f"Select Values completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        print("Returning DataFrame (not saving to file)")
        return filtered_dataset


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python select_values_template.py <params.json>",
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