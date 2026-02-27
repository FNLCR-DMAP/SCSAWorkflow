#!/usr/bin/env python3
"""
Platform-agnostic Manual Phenotyping template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.manual_phenotyping_template import run_from_json
>>> run_from_json("examples/manual_phenotyping_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.phenotyping import assign_manual_phenotypes
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
    Execute Manual Phenotyping analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Dataset": "path/to/dataframe.csv",
            "Phenotypes_Code": "path/to/phenotypes.csv",
            "Classification_Column_Prefix": "",
            "Classification_Column_Suffix": "",
            "Allow_Multiple_Phenotypes": true,
            "Manual_Annotation_Name": "manual_phenotype",
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves the DataFrame with
        phenotype annotations to a CSV file. If False, returns the DataFrame
        directly for in-memory workflows. Default is True.
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
        If save_to_disk=False: The processed DataFrame with phenotype annotations

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
    >>> phenotyped_df = run_from_json("params.json", save_to_disk=False)
    
    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    if "outputs" not in params:
        params["outputs"] = {
            "dataframe": {"type": "file", "name": "dataframe.csv"}
        }

    # Load upstream data - DataFrame or CSV file
    upstream = params['Upstream_Dataset']
    if isinstance(upstream, pd.DataFrame):
        dataframe = upstream  # Direct DataFrame from previous step
    elif isinstance(upstream, (str, Path)):
        # Galaxy passes .dat files, but they contain CSV data
        # Don't check extension - directly read as CSV
        path = Path(upstream)
        try:
            dataframe = pd.read_csv(path)
            logging.info(f"Successfully loaded CSV data from: {path}")
        except Exception as e:
            raise ValueError(
                f"Failed to read CSV data from '{path}'. "
                f"This tool expects CSV/tabular format. "
                f"Error: {str(e)}"
            )
    else:
        raise TypeError(
            f"Upstream_Dataset must be DataFrame or file path. "
            f"Got {type(upstream)}"
        )

    # Load phenotypes code - DataFrame or CSV file
    phenotypes_input = params['Phenotypes_Code']
    if isinstance(phenotypes_input, pd.DataFrame):
        phenotypes = phenotypes_input
    elif isinstance(phenotypes_input, (str, Path)):
        # Galaxy passes .dat files, but they contain CSV data
        # Don't check extension - directly read as CSV
        path = Path(phenotypes_input)
        try:
            phenotypes = pd.read_csv(path)
            logging.info(f"Successfully loaded phenotypes from: {path}")
        except Exception as e:
            raise ValueError(
                f"Failed to read CSV data from '{path}'. "
                f"This tool expects CSV/tabular format. "
                f"Error: {str(e)}"
            )
    else:
        raise TypeError(
            f"Phenotypes_Code must be DataFrame or file path. "
            f"Got {type(phenotypes_input)}"
        )

    # Extract parameters
    prefix = params.get('Classification_Column_Prefix', '')
    suffix = params.get('Classification_Column_Suffix', '')
    multiple = params.get('Allow_Multiple_Phenotypes', True)
    manual_annotation = params.get('Manual_Annotation_Name', 'manual_phenotype')

    logging.info(f"Phenotypes configuration:\n{phenotypes}")

    # returned_dic is not used, but copy from original NIDAP logic
    returned_dic = assign_manual_phenotypes(
        dataframe,
        phenotypes,
        prefix=prefix,
        suffix=suffix,
        annotation=manual_annotation,
        multiple=multiple
    )

    # The dataframe changes in place

    # Print summary statistics
    phenotype_counts = dataframe[manual_annotation].value_counts()
    logging.info(f"\nPhenotype distribution:\n{phenotype_counts}")

    logging.info("\nManual Phenotyping completed successfully.")

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for dataframe output
        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = dataframe
        
        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info("Manual Phenotyping analysis completed successfully.")
        return saved_files
    else:
        # Return the DataFrame directly for in-memory workflows
        logging.info("Returning DataFrame for in-memory use")
        return dataframe


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python manual_phenotyping_template.py <params.json> [output_dir]",
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
