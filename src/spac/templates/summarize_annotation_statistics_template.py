"""
Platform-agnostic Summarize Annotation's Statistics template converted from 
NIDAP. Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.summarize_annotation_statistics_template import \
...     run_from_json
>>> run_from_json("examples/summarize_annotation_statistics_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Tuple
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import get_cluster_info
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
    Execute Summarize Annotation's Statistics analysis with parameters from 
    JSON. Replicates the NIDAP template functionality exactly.

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
    layer = params.get("Table_to_Process", "Original")
    features = params.get("Feature_s_", ["All"])
    annotation = params.get("Annotation", "None")

    if layer == "Original":
        layer = None
        
    if len(features) == 1 and features[0] == "All":
        features = None

    if annotation == "None":
        annotation = None

    info = get_cluster_info(
        adata=adata,
        layer=layer,
        annotation=annotation,
        features=features
    )

    df = pd.DataFrame(info)

    # Renaming columns to avoid spaces and special characters
    # Assuming `info` is a pandas DataFrame
    df.columns = [
        col.replace(" ", "_").replace("-", "_") for col in df.columns
    ]

    # Get summary statistics of returned dataset
    print("Summary statistics of the dataset:", df.describe())

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "annotation_summaries.csv")
        saved_files = save_outputs({output_file: df})
        
        print(
            f"Summarize Annotation's Statistics completed → "
            f"{saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        print("Returning DataFrame (not saving to file)")
        return df


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python summarize_annotation_statistics_template.py "
            "<params.json>",
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