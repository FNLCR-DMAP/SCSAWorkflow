"""
Platform-agnostic Summarize DataFrame template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import summarize_dataframe
from spac.visualization import present_summary_as_figure
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True,
    show_plot: bool = True
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Summarize DataFrame analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the figure and
        dataframe directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    dict or tuple
        If save_results=True: Dictionary of saved file paths
        If save_results=False: Tuple of (figure, dataframe)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    # Note: load_input is for AnnData objects; DataFrames from CSV are handled separately
    input_path = params["Calculate_Centroids"]
    if isinstance(input_path, str) and input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        # For pickle files that contain DataFrames
        df = load_input(input_path)

    # Extract parameters
    columns = params["Columns"]
    print_missing_location = params.get("Print_Missing_Location", False)

    # Run the analysis exactly as in NIDAP template
    summary = summarize_dataframe(
        df,
        columns=columns,
        print_nan_locations=print_missing_location)
    
    # Generate HTML from the summary.
    fig = present_summary_as_figure(summary)

    if show_plot:
        fig.show()  # Opens in an interactive Plotly window

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "summary.html")
        
        # Since the figure is a Plotly figure, we save it as HTML
        if not output_file.endswith('.html'):
            output_file = output_file + '.html'
        
        fig.write_html(output_file)
        saved_files = {output_file: output_file}
        
        print(f"Summarize DataFrame completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the figure and summary dataframe directly for in-memory workflows
        print("Returning figure and dataframe (not saving to file)")
        return fig, summary


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python summarize_dataframe_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure and dataframe")