"""
Platform-agnostic Subset Analysis template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.subset_analysis_template import run_from_json
>>> run_from_json("examples/subset_analysis_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Tuple
import pandas as pd
import warnings
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import SPAC functions from NIDAP template
from spac.data_utils import select_values
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], Any]:
    """
    Execute Subset Analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Annotation_of_interest": "cell_type",
            "Labels": ["T cells", "B cells"],
            "Include_Exclude": "Include Selected Labels",
            "outputs": {
                "analysis": {"type": "file", "name": "transform_output.pickle"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves the filtered AnnData
        to a pickle file. If False, returns the AnnData object directly for 
        in-memory workflows. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory. All outputs will be saved relative to this directory.

    Returns
    -------
    dict or AnnData
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {
                "analysis": "path/to/transform_output.pickle"
            }
        If save_to_disk=False: The processed AnnData object

    Notes
    -----
    Output Structure:
    - Analysis output is saved as a single pickle file
    - When save_to_disk=False, the AnnData object is returned for programmatic use
    
    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["analysis"])  # Path to saved pickle file
    
    >>> # Get results in memory
    >>> filtered_adata = run_from_json("params.json", save_to_disk=False)
    
    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    # Analysis outputs use file type per standardized schema
    if "outputs" not in params:
        params["outputs"] = {
            "analysis": {"type": "file", "name": "transform_output.pickle"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    # Use direct dictionary access for required parameters (NIDAP style)
    annotation = params["Annotation_of_interest"]
    labels = params["Labels"]
    
    # Use .get() with defaults for optional parameters from JSON template
    toggle = params.get("Include_Exclude", "Include Selected Labels")

    if toggle == "Include Selected Labels":
        values_to_include = labels
        values_to_exclude = None
    else:
        values_to_include = None
        values_to_exclude = labels

    with warnings.catch_warnings(record=True) as caught_warnings:
        filtered_adata = select_values(
            data=adata,
            annotation=annotation,
            values=values_to_include,
            exclude_values=values_to_exclude
            )
        if caught_warnings:
            for warning in caught_warnings:
                raise ValueError(warning.message)
    
    logging.info(filtered_adata)
    logging.info("\n")

    # Count and display occurrences of each label in the annotation
    label_counts = filtered_adata.obs[annotation].value_counts()
    logging.info(label_counts)
    logging.info("\n")

    dataframe = pd.DataFrame(
        filtered_adata.X, 
        columns=filtered_adata.var.index, 
        index=filtered_adata.obs.index
    )
    logging.info(dataframe.describe())
    
    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for analysis output (backward compatibility with "Output_File")
        if "analysis" in params["outputs"]:
            results_dict["analysis"] = filtered_adata
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info("Subset Analysis completed successfully.")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        logging.info("Returning AnnData object for in-memory use")
        return filtered_adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python subset_analysis_template.py <params.json> [output_dir]",
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
        print("\nReturned AnnData object")
        print(f"AnnData shape: {result.shape}")
