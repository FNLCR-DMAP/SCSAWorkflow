"""
Platform-agnostic Combine Annotations template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.combine_annotations_template import run_from_json
>>> run_from_json("examples/combine_annotations_params.json")
"""
import json
import sys
import logging
from pathlib import Path
from typing import Any, Dict, Union, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import combine_annotations
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
)

logger = logging.getLogger(__name__)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], Any]:
    """
    Execute Combine Annotations analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Annotations_Names": ["annotation1", "annotation2"],
            "New_Annotation_Name": "combined_annotation",
            "Separator": "_",
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"},
                "analysis": {"type": "file", "name": "output.pickle"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the adata object
        directly for in-memory workflows. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory.

    Returns
    -------
    dict or AnnData
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {
                "dataframe": "path/to/dataframe.csv",
                "analysis": "path/to/output.pickle"
            }
        If save_to_disk=False: The processed AnnData object

    Notes
    -----
    Output Structure:
    - Analysis output is saved as a pickle file
    - DataFrame (label counts) is saved as a CSV file
    - When save_to_disk=False, the AnnData object is returned for programmatic use
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    # Ensure outputs configuration exists with standardized defaults
    if "outputs" not in params:
        params["outputs"] = {
            "dataframe": {"type": "file", "name": "dataframe.csv"},
            "analysis": {"type": "file", "name": "output.pickle"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    annotations_list = params["Annotations_Names"]
    new_annotation = params.get("New_Annotation_Name", "combined_annotation")
    separator = params.get("Separator", "_")

    combine_annotations(
        adata,
        annotations=annotations_list,
        separator=separator,
        new_annotation_name=new_annotation
    )

    logger.info(f"After combining annotations: \n{adata}")
    value_counts = adata.obs[new_annotation].value_counts(dropna=False)
    logger.info(f"Unique labels in {new_annotation}")
    logger.info(f"{value_counts}")

    # Create the frequency CSV for download
    df_counts = (
        value_counts
        .rename_axis(new_annotation)   # move index to a column name
        .reset_index(name='count')     # two columns: label | count
    )

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}

        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = df_counts

        if "analysis" in params["outputs"]:
            results_dict["analysis"] = adata

        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        logger.info("Combine Annotations analysis completed successfully.")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        logger.info("Returning AnnData object for in-memory use")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python combine_annotations_template.py <params.json> "
            "[output_dir]",
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

    result = run_from_json(
        json_path=sys.argv[1],
        output_dir=output_dir
    )

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
