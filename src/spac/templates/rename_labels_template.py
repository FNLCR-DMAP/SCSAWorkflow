"""
Platform-agnostic Rename Labels template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Follows standardized output schema.

Usage
-----
>>> from spac.templates.rename_labels_template import run_from_json
>>> run_from_json("examples/rename_labels_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import logging
import pandas as pd
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import rename_annotations
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
    Execute Rename Labels analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Cluster_Mapping_Dictionary": "path/to/mapping.csv",
            "Source_Annotation": "original_column",
            "New_Annotation": "new_column",
            "outputs": {
                "analysis": {"type": "file", "name": "renamed_data.pickle"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the adata object
        directly for in-memory workflows. Default is True.
    output_dir : str, optional
        Override output directory from params. Default uses params value.

    Returns
    -------
    dict or AnnData
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: The processed AnnData object
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    logger.info(f"Loading upstream analysis data from {params['Upstream_Analysis']}")
    all_data = load_input(params["Upstream_Analysis"])

    # Extract parameters
    rename_list_path = params["Cluster_Mapping_Dictionary"]
    original_column = params.get("Source_Annotation", "None")
    renamed_column = params.get("New_Annotation", "None")

    # Load the mapping dictionary CSV
    logger.info(f"Loading cluster mapping dictionary from {rename_list_path}")
    rename_list = pd.read_csv(rename_list_path)

    original_column = text_to_value(original_column)
    renamed_column = text_to_value(renamed_column)

    # Create a new dictionary with the desired format
    dict_list = rename_list.to_dict('records')
    mappings = {d['Original']: d['New'] for d in dict_list}

    logger.info(f"Cluster Name Mapping: \n{mappings}")

    rename_annotations(
        all_data, 
        src_annotation=original_column,
        dest_annotation=renamed_column,
        mappings=mappings)

    logger.info(f"After Renaming Clusters: \n{all_data}")

    # Count and display occurrences of each label in the annotation
    logger.info(f'Count of cells in the output annotation:"{renamed_column}":')
    label_counts = all_data.obs[renamed_column].value_counts()
    logger.info(f"{label_counts}")

    object_to_output = all_data
    
    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Add analysis output (single file)
        if "analysis" in params["outputs"]:
            results_dict["analysis"] = object_to_output
        
        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logger.info("Rename Labels analysis completed successfully.")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        logger.info("Returning AnnData object for in-memory use")
        return object_to_output


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python rename_labels_template.py <params.json> [output_dir]",
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
        print(f"Observations columns: {list(result.obs.columns)}")
