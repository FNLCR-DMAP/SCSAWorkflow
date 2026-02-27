"""
Platform-agnostic Nearest Neighbor Calculation template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.nearest_neighbor_calculation_template import (
...     run_from_json
... )
>>> run_from_json("examples/nearest_neighbor_calculation_params.json")
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.spatial_analysis import calculate_nearest_neighbor
from spac.templates.template_utils import (
    load_input,
    parse_params,
    save_results,
    text_to_value,
)

# Set up logging
logger = logging.getLogger(__name__)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: Union[str, Path] = None
) -> Union[Dict[str, str], Any]:
    """
    Execute Nearest Neighbor Calculation analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/input.pickle",
            "Annotation": "cell_type",
            "ImageID": "None",
            "Nearest_Neighbor_Associated_Table": "spatial_distance",
            "Verbose": true,
            "outputs": {
                "analysis": {"type": "file", "name": "output.pickle"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the adata object
        directly for in-memory workflows. Default is True.
    output_dir : str or Path, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory. All outputs will be saved relative to this directory.

    Returns
    -------
    dict or AnnData
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {"analysis": "path/to/output.pickle"}
        If save_to_disk=False: The processed AnnData object

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
    annotation = params["Annotation"]
    spatial_associated_table = "spatial"
    imageid = params.get("ImageID", "None")
    label = params.get(
        "Nearest_Neighbor_Associated_Table", "spatial_distance"
    )
    verbose = params.get("Verbose", True)

    # Convert any string "None" to actual None for Python
    imageid = text_to_value(imageid, default_none_text="None")

    logger.info(
        "Running `calculate_nearest_neighbor` with the following parameters:"
    )
    logger.info(f"  annotation: {annotation}")
    logger.info(f"  spatial_associated_table: {spatial_associated_table}")
    logger.info(f"  imageid: {imageid}")
    logger.info(f"  label: {label}")
    logger.info(f"  verbose: {verbose}")

    # Perform the nearest neighbor calculation
    calculate_nearest_neighbor(
        adata=adata,
        annotation=annotation,
        spatial_associated_table=spatial_associated_table,
        imageid=imageid,
        label=label,
        verbose=verbose
    )

    logger.info("Nearest neighbor calculation complete.")
    logger.info(f"adata.obsm keys: {list(adata.obsm.keys())}")
    if label in adata.obsm:
        logger.info(
            f"Preview of adata.obsm['{label}']:\n{adata.obsm[label].head()}"
        )

    logger.info(f"{adata}")

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

        logger.info(
            f"Nearest Neighbor Calculation completed → "
            f"{saved_files['analysis']}"
        )
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        logger.info("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python nearest_neighbor_calculation_template.py "
            "<params.json> [output_dir]",
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
