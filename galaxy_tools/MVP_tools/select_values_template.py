"""
Platform-agnostic Select Values template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.select_values_template import run_from_json
>>> run_from_json("examples/select_values_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Tuple
import pandas as pd
import warnings
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import select_values
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
    Execute Select Values analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Dataset": "path/to/dataframe.csv",
            "Annotation_of_Interest": "cell_type",
            "Label_s_of_Interest": ["T cells", "B cells"],
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves the filtered DataFrame
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
        If save_to_disk=False: The filtered DataFrame

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
    >>> filtered_df = run_from_json("params.json", save_to_disk=False)
    
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
        input_dataset = upstream_dataset  # Direct DataFrame from previous step
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

    # Extract parameters - support both "Label_s_of_Interest" and "Labels_of_Interest" 
    # for backward compatibility with JSON template
    observation = params.get("Annotation_of_Interest")
    values = params.get("Label_s_of_Interest") or params.get("Labels_of_Interest")

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        filtered_dataset = select_values(
            data=input_dataset,
            annotation=observation,
            values=values
            )
        # Only process warnings that are relevant to the select_values operation
        if caught_warnings:
            for warning in caught_warnings:
                # Skip deprecation warnings from numpy/pandas
                if (hasattr(warning, 'category') and 
                    issubclass(warning.category, DeprecationWarning)):
                    continue
                # Raise actual operational warnings as errors
                if hasattr(warning, 'message'):
                    raise ValueError(str(warning.message))
    
    logging.info(filtered_dataset.info())

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for dataframe output
        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = filtered_dataset
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info("Select Values analysis completed successfully.")
        return saved_files
    else:
        # Return the dataframe directly for in-memory workflows
        logging.info("Returning DataFrame for in-memory use")
        return filtered_dataset


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python select_values_template.py <params.json> [output_dir]",
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
