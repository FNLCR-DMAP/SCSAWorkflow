"""
Platform-agnostic Downsample Cells template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.downsample_cells_template import run_from_json
>>> run_from_json("examples/downsample_cells_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Tuple
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import downsample_cells
from spac.utils import check_column_name
from spac.templates.template_utils import (
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Downsample Cells analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Dataset": "path/to/dataframe.csv",
            "Annotations_List": ["cell_type", "tissue"],
            "Number_of_Samples": 1000,
            "Stratify_Option": true,
            "Random_Selection": true,
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves the downsampled DataFrame
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
        If save_to_disk=False: The downsampled DataFrame

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
    >>> downsampled_df = run_from_json("params.json", save_to_disk=False)
    
    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    # DataFrames typically use file type per standardized schema
    if "outputs" not in params:
        params["outputs"] = {
            "dataframe": {"type": "file", "name": "dataframe.csv"}
        }

    # Load upstream data - could be DataFrame, CSV
    upstream_dataset = params["Upstream_Dataset"]
    if isinstance(upstream_dataset, pd.DataFrame):
        input_dataset = upstream_dataset  # Direct DF from previous step
    elif isinstance(upstream_dataset, (str, Path)):
        try:
            input_dataset = pd.read_csv(upstream_dataset)
        except Exception as e:
            raise ValueError(f"Failed to read CSV from {upstream_dataset}: {e}")
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

    logging.info("Downsampled! Processed dataset info:")
    logging.info(down_sampled_dataset.info())

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for dataframe output
        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = down_sampled_dataset
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info("Downsample Cells analysis completed successfully.")
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        logging.info("Returning DataFrame for in-memory use")
        return down_sampled_dataset


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python downsample_cells_template.py <params.json> [output_dir]",
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
        print("\nReturned DataFrame")
        print(f"DataFrame shape: {result.shape}")
