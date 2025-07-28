"""
Platform-agnostic Setup Analysis template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.setup_analysis_template import run_from_json
>>> run_from_json("examples/setup_analysis_params.json")
"""

import sys
from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd
import ast
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import ingest_cells
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], Any]:
    """
    Execute Setup Analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the adata object
        directly for in-memory workflows. Default is True.

    Returns
    -------
    dict or AnnData
        If save_results=True: Dictionary of saved file paths
        If save_results=False: The processed AnnData object
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Extract parameters
    upstream_dataset = params["Upstream_Dataset"]
    feature_names = params["Features_to_Analyze"]
    regex_str = params.get("Feature_Regex", [])
    x_col = params["X_Coordinate_Column"]
    y_col = params["Y_Coordinate_Column"]
    annotation = params["Annotation_s_"]

    # Load upstream data - could be DataFrame or CSV
    if isinstance(upstream_dataset, (str, Path)):
        # Check if it's a pickle/h5ad file or CSV
        path = Path(upstream_dataset)
        if path.suffix.lower() in ['.pickle', '.pkl', '.p', '.h5ad']:
            input_dataset = load_input(upstream_dataset)
        else:
            # Assume it's a CSV file
            input_dataset = pd.read_csv(upstream_dataset)
    else:
        # Already a DataFrame
        input_dataset = upstream_dataset

    # Process annotation parameter
    if isinstance(annotation, str):
        annotation = [annotation]

    if len(annotation) == 1 and annotation[0] == "None":
        annotation = None

    if annotation and len(annotation) != 1 and "None" in annotation:
        error_msg = 'String "None" found in the annotation list'
        raise ValueError(error_msg)

    # Process coordinate columns
    x_col = text_to_value(x_col, default_none_text="None")
    y_col = text_to_value(y_col, default_none_text="None")

    # Process feature names and regex
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    if isinstance(regex_str, str):
        try:
            regex_str = ast.literal_eval(regex_str)
        except (ValueError, SyntaxError):
            regex_str = [regex_str] if regex_str else []

    # Processing two search methods
    for feature in feature_names:
        regex_str.append(f"^{feature}$")

    # Sanitizing search list
    regex_str_set = set(regex_str)
    regex_str_list = list(regex_str_set)

    # Run the ingestion
    ingested_anndata = ingest_cells(
        dataframe=input_dataset,
        regex_str=regex_str_list,
        x_col=x_col,
        y_col=y_col,
        annotation=annotation
    )

    print("Analysis Setup:")
    print(ingested_anndata)
    print("Schema:")
    print(ingested_anndata.var_names)

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'

        saved_files = save_outputs({output_file: ingested_anndata})

        logging.info(
            f"Setup Analysis completed → {saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        logging.info("Returning AnnData object (not saving to file)")
        return ingested_anndata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python setup_analysis_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    saved_files = run_from_json(sys.argv[1])

    if isinstance(saved_files, dict):
        print("\nOutput files:")
        for filename, filepath in saved_files.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned AnnData object")