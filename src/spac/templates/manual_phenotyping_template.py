#!/usr/bin/env python3
"""
Platform-agnostic Manual Phenotyping template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.phenotyping import assign_manual_phenotypes
from spac.templates.template_utils import (
    save_outputs,
    parse_params
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], pd.DataFrame]:
    """
    Execute Manual Phenotyping analysis with parameters from JSON.
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

    # Load input data - support both DataFrame and CSV file
    upstream = params['Upstream_Dataset']
    if isinstance(upstream, pd.DataFrame):
        dataframe = upstream  # Direct DataFrame pass from previous step
    else:
        dataframe = pd.read_csv(upstream)  # Read from CSV file

    # dataframe = {{{Upstream_Dataset}}}  # Already loaded above
    phenotypes = pd.read_csv(params['Phenotypes_Code'])
    prefix = params.get('Classification_Column_Prefix', '')
    suffix = params.get('Classification_Column_Suffix', '')
    multiple = params.get('Allow_Multiple_Phenotypes', True)
    manual_annotation = params.get('Manual_Annotation_Name', 'manual_phenotype')

    print(phenotypes)

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
    # --------- Original NIDAP Logic End --------- #

    # Print summary statistics
    phenotype_counts = dataframe[manual_annotation].value_counts()
    print(f"\nPhenotype distribution:")
    print(phenotype_counts)

    print("\nManual Phenotyping completed successfully.")

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "manual_phenotyped.csv")
        saved_files = save_outputs({output_file: dataframe})
        
        print(f"Manual Phenotyping completed → {saved_files[output_file]}")
        
        return saved_files
    else:
        # Return the DataFrame directly for in-memory workflows
        print("Returning DataFrame (not saving to file)")
        return dataframe


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python manual_phenotyping_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    saved_files = run_from_json(sys.argv[1])

    if isinstance(saved_files, dict):
        print("\nOutput files:")
        for filename, filepath in saved_files.items():
            print(f"  {filename}: {filepath}")