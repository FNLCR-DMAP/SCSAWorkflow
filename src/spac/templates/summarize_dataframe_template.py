"""
Platform-agnostic Summarize DataFrame template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.summarize_dataframe_template import run_from_json
>>> run_from_json("examples/summarize_dataframe_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Tuple
import pandas as pd
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import summarize_dataframe
from spac.visualization import present_summary_as_figure
from spac.templates.template_utils import (
    save_results,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
    show_plot: bool = False,
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Summarize DataFrame analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Dataset": "path/to/dataframe.csv",
            "Columns": ["col1", "col2"],
            "Print_Missing_Location": false,
            "outputs": {
                "html": {"type": "directory", "name": "html_dir"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves the HTML summary
        to a directory. If False, returns the figure and dataframe directly
        for in-memory workflows. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory. All outputs will be saved relative to this directory.
    show_plot : bool, optional
        Whether to display the plot interactively. Default is False.

    Returns
    -------
    dict or tuple
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {
                "html": ["path/to/html_dir/summary.html"]
            }
        If save_to_disk=False: Tuple of (figure, summary_dataframe)

    Notes
    -----
    Output Structure:
    - HTML is saved to a directory as specified in outputs config
    - When save_to_disk=False, returns (figure, summary_df) for programmatic use
    
    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["html"])  # List of paths to saved HTML files
    
    >>> # Get results in memory
    >>> fig, summary_df = run_from_json("params.json", save_to_disk=False)
    
    >>> # Custom output directory with interactive display
    >>> saved = run_from_json("params.json", output_dir="/custom/path", show_plot=True)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    # HTML outputs use directory type per standardized schema
    if "outputs" not in params:
        params["outputs"] = {
            "html": {"type": "directory", "name": "html_dir"}
        }

    # Load upstream data - DataFrame or CSV file
    # Corrected "Calculate_Centroids" to "Upstream_Dataset" in the blueprint
    input_path = params.get("Upstream_Dataset")
    if isinstance(input_path, pd.DataFrame):
        df = input_path  # Direct DataFrame from previous step
    elif isinstance(input_path, (str, Path)):
        # Galaxy passes .dat files, but they contain CSV data
        # Don't check extension - directly read as CSV
        path = Path(input_path)
        try:
            df = pd.read_csv(path)
            logging.info(f"Successfully loaded CSV data from: {path}")
        except Exception as e:
            raise ValueError(
                f"Failed to read CSV data from '{path}'. "
                f"This tool expects CSV/tabular format. "
                f"Error: {str(e)}"
            )
    else:
        raise TypeError(
            f"Input dataset must be DataFrame or file path. "
            f"Got {type(input_path)}"
        )

    # Extract parameters
    columns = params["Columns"]
    print_missing_location = params.get("Print_Missing_Location", False)

    # Run the analysis exactly as in NIDAP template
    summary = summarize_dataframe(
        df,
        columns=columns,
        print_nan_locations=print_missing_location
    )
    
    # Generate figure from the summary
    fig = present_summary_as_figure(summary)

    if show_plot:
        fig.show()  # Opens in an interactive Plotly window

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Check for html output - convert figure to HTML string
        if "html" in params["outputs"]:
            # Convert Plotly figure to HTML string for save_results
            html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
            results_dict["html"] = {"summary": html_content}
        
        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info("Summarize DataFrame analysis completed successfully.")
        return saved_files
    else:
        # Return the figure and summary dataframe directly for in-memory workflows
        logging.info("Returning figure and dataframe for in-memory use")
        return fig, summary


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python summarize_dataframe_template.py <params.json> [output_dir]",
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
        print("\nReturned figure and dataframe")
