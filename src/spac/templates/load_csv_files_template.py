"""
Platform-agnostic Load CSV Files template converted from NIDAP.
Handles both Galaxy (list of file paths) and NIDAP (directory path) inputs.

Usage
-----
>>> from spac.templates.load_csv_files_template import run_from_json
>>> run_from_json("examples/load_csv_params.json")
"""
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.templates.template_utils import (
    save_results,
    parse_params,
    load_csv_files,
)

logger = logging.getLogger(__name__)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Load CSV Files analysis with parameters from JSON.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file or parameter dictionary
    save_to_disk : bool, optional
        Whether to save results to disk. Default is True.
    output_dir : str, optional
        Base directory for outputs.

    Returns
    -------
    dict or DataFrame
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: The processed DataFrame
    """
    params = parse_params(json_path)

    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    if "outputs" not in params:
        params["outputs"] = {"dataframe": {"type": "file", "name": "dataframe.csv"}}

    # Load configuration
    files_config = pd.read_csv(params["CSV_Files_Configuration"])

    # Load and combine CSV files using centralized utility
    final_df = load_csv_files(
        csv_input=params["CSV_Files"],
        files_config=files_config,
        string_columns=params.get("String_Columns", [])
    )

    logger.info(f"Load CSV Files completed: {final_df.shape}")

    # Save or return results
    if save_to_disk:
        saved_files = save_results(
            results={"dataframe": final_df},
            params=params,
            output_base_dir=output_dir
        )
        return saved_files
    else:
        return final_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_csv_files_template.py <params.json> [output_dir]")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    result = run_from_json(sys.argv[1], output_dir=output_dir)

    if isinstance(result, dict):
        for key, path in result.items():
            print(f"{key}: {path}")
