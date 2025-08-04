"""
Platform-agnostic Calculate Centroid template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.calculate_centroid_template import run_from_json
>>> run_from_json("examples/calculate_centroid_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import calculate_centroid
from spac.utils import check_column_name
from spac.templates.template_utils import (
    save_outputs,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Calculate Centroid analysis with parameters from JSON.
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

    # Extract parameters using .get() with defaults from JSON template
    x_min = params.get("Min_X_Coordinate_Column_Name", "XMin")
    x_max = params.get("Max_X_Coordinate_Column_Name", "XMax")
    y_min = params.get("Min_Y_Coordinate_Column_Name", "YMin")
    y_max = params.get("Max_Y_Coordinate_Column_Name", "YMax")
    new_x = params.get("X_Centroid_Name", "XCentroid")
    new_y = params.get("Y_Centroid_Name", "YCentroid")

    check_column_name(new_x, "X Centroid Name")
    check_column_name(new_y, "Y Centroid Name")

    centroid_calculated = calculate_centroid(
        input_dataset,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        new_x=new_x,
        new_y=new_y
    )

    print(centroid_calculated.info())

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "centroid_calculated.csv")
        # Default to CSV format if no recognized extension
        if not output_file.endswith(('.csv', '.pickle', '.pkl')):
            output_file = output_file + '.csv'
        
        saved_files = save_outputs({output_file: centroid_calculated})
        
        print(f"Calculate Centroid completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the DataFrame directly for in-memory workflows
        print("Returning DataFrame (not saving to file)")
        return centroid_calculated


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python calculate_centroid_template.py <params.json>",
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