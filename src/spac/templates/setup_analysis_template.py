"""
Platform-agnostic Setup Analysis template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Follows standardized output schema where analysis is saved as a file.

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
    save_results,
    parse_params,
    text_to_value
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, str], Any]:
    """
    Execute Setup Analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Dataset": "path/to/data.csv",
            "Features_to_Analyze": ["CD25", "CD3D"],
            "Feature_Regex": [],
            "X_Coordinate_Column": "X_centroid",
            "Y_Coordinate_Column": "Y_centroid",
            "Annotation_s_": ["cell_type"],
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

    # Ensure outputs configuration exists with standardized defaults
    # Analysis uses file type per standardized schema
    if "outputs" not in params:
        # Get output filename from params or use default
        output_file = params.get("Output_File", "output.pickle")
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        params["outputs"] = {
            "analysis": {"type": "file", "name": output_file}
        }

    # Extract parameters
    upstream_dataset = params["Upstream_Dataset"]
    feature_names = params["Features_to_Analyze"]
    regex_str = params.get("Feature_Regex", [])
    x_col = params["X_Coordinate_Column"]
    y_col = params["Y_Coordinate_Column"]
    annotation = params["Annotation_s_"]

    # Load upstream data - could be DataFrame or CSV
    if isinstance(upstream_dataset, (str, Path)):
        try:
            input_dataset = pd.read_csv(upstream_dataset)
            # Validate it's a proper DataFrame
            if input_dataset.empty:
                raise ValueError("CSV file is empty")
        except Exception as e:
            raise ValueError(f"Failed to read CSV from {upstream_dataset}: {e}")
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

    logging.info("Analysis Setup:")
    logging.info(f"{ingested_anndata}")
    logging.info("Schema:")
    logging.info(f"{ingested_anndata.var_names.tolist()}")

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        if "analysis" in params["outputs"]:
            results_dict["analysis"] = ingested_anndata
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        logging.info(
            f"Setup Analysis completed → {saved_files['analysis']}"
        )
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        logging.info("Returning AnnData object (not saving to file)")
        return ingested_anndata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python setup_analysis_template.py <params.json>",
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
        print(f"Shape: {result.shape}")