"""
Platform-agnostic Downsample Cells template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.downsample_cells_template import run_from_json
>>> run_from_json("examples/downsample_cells_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import downsample_cells
from spac.utils import check_column_name
from spac.templates.template_utils import (
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Downsample Cells analysis with parameters from JSON.
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
        input_dataset = upstream_dataset  # Direct DF from previous step
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
    annotations = params["Annotations_List"]
    n_samples = params["Number_of_Samples"]
    stratify = params["Stratify_Option"]
    rand = params["Random_Selection"]
    combined_col_name = params.get(
        "New_Combined_Annotation_Name", "_combined_"
    )
    min_threshold = params.get("Minimum_Threshold", 5)

    check_column_name(
        combined_col_name, "New Combined Annotation Name"
    )

    down_sampled_dataset = downsample_cells(
        input_data=input_dataset,
        annotations=annotations,
        n_samples=n_samples,
        stratify=stratify,
        rand=rand,
        combined_col_name=combined_col_name,
        min_threshold=min_threshold
    )

    print("Downsampled! Processed dataset info:")
    print(down_sampled_dataset.info())

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "downsampled_data.csv")
        # Default to CSV format if no recognized extension
        if not output_file.endswith(('.csv', '.pickle', '.pkl')):
            output_file = output_file + '.csv'
        
        saved_files = save_outputs({output_file: down_sampled_dataset})
        
        print(
            f"Downsample Cells completed → {saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        print("Returning DataFrame (not saving to file)")
        return down_sampled_dataset


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python downsample_cells_template.py <params.json>",
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