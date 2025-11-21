"""
Platform-agnostic Z-Score Normalization template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Follows standardized output schema where analysis is saved as a file.

Usage
-----
>>> from spac.templates.zscore_normalization_template import run_from_json
>>> run_from_json("examples/zscore_normalization_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import pickle
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import z_score_normalization
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], Any]:
    """
    Execute Z-Score Normalization analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Table_to_Process": "Original",
            "Output_Table_Name": "z_scores",
            "outputs": {
                "analysis": {"type": "file", "name": "output.pickle"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves the AnnData object
        to a pickle file. If False, returns the AnnData object directly 
        for in-memory workflows. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory. All outputs will be saved relative to this directory.

    Returns
    -------
    dict or AnnData
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {"analysis": "path/to/output.pickle"}
        If save_to_disk=False: The processed AnnData object for in-memory use

    Notes
    -----
    Output Structure:
    - Analysis output is saved as a single pickle file (standardized for analysis outputs)
    - When save_to_disk=False, the AnnData object is returned for programmatic use
        
    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["analysis"])  # Path to saved pickle file
    >>> # './output.pickle'
    
    >>> # Get results in memory for further processing
    >>> adata = run_from_json("params.json", save_to_disk=False)
    >>> # Can now work with adata object directly
    
    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    # Ensure outputs configuration exists with standardized defaults
    # Analysis uses file type per standardized schema
    if "outputs" not in params:
        params["outputs"] = {
            "analysis": {"type": "file", "name": "output.pickle"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    input_layer = params["Table_to_Process"]
    output_layer = params["Output_Table_Name"]

    if input_layer == "Original":
        input_layer = None

    z_score_normalization(
        adata,
        output_layer=output_layer,
        input_layer=input_layer
    )
    
    # Convert the normalized layer to a DataFrame and print its summary
    post_dataframe = adata.to_df(layer=output_layer)
    logging.info(f"Z-score normalization summary:\n{post_dataframe.describe()}")
    logging.info(f"Transformed data:\n{adata}")

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        if "analysis" in params["outputs"]:
            results_dict["analysis"] = adata
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info(
            f"Z-Score Normalization completed → {saved_files['analysis']}"
        )
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        logging.info("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python zscore_normalization_template.py <params.json> [output_dir]",
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
        for key, path in result.items():
            print(f"  {key}: {path}")
    else:
        print("\nReturned AnnData object for in-memory use")
        print(f"AnnData: {result}")
