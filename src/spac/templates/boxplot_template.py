"""
Platform-agnostic Boxplot template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.boxplot_template import run_from_json
>>> run_from_json("examples/boxplot_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Tuple
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import boxplot
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True,
    show_plot: bool = True
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Boxplot analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the figure and
        summary dataframe directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    dict or tuple
        If save_results=True: Dictionary of saved file paths
        If save_results=False: Tuple of (figure, summary_dataframe)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    annotation = params.get("Primary_Annotation", "None")
    second_annotation = params.get("Secondary_Annotation", "None")
    layer_to_plot = params.get("Table_to_Visualize", "Original")
    feature_to_plot = params.get("Feature_s_to_Plot", ["All"])
    log_scale = params.get("Value_Axis_Log_Scale", False)
    
    # Figure parameters
    figure_title = params.get("Figure_Title", "BoxPlot")
    figure_horizontal = params.get("Horizontal_Plot", False)
    fig_width = params.get("Figure_Width", 12)
    fig_height = params.get("Figure_Height", 8)
    fig_dpi = params.get("Figure_DPI", 300)
    font_size = params.get("Font_Size", 10)
    showfliers = params.get("Keep_Outliers", True)

    # Process parameters exactly as in NIDAP template
    if layer_to_plot == "Original":
        layer_to_plot = None
    
    if second_annotation == "None":
        second_annotation = None
    
    if annotation == "None":
        annotation = None

    if figure_horizontal:
        figure_orientation = "h"
    else:
        figure_orientation = "v"

    if any(item == "All" for item in feature_to_plot):
        logging.info("Plotting All Features")
        feature_to_plot = adata.var_names.tolist()
    else:
        feature_str = "\n".join(feature_to_plot)
        logging.info(f"Plotting Feature:\n{feature_str}")

    # Create the plot exactly as in NIDAP template
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': font_size})
    fig.set_size_inches(fig_width, fig_height)
    fig.set_dpi(fig_dpi)

    fig, ax, df = boxplot(
        adata=adata,
        ax=ax,
        layer=layer_to_plot,
        annotation=annotation,
        second_annotation=second_annotation,
        features=feature_to_plot,
        log_scale=log_scale,
        orient=figure_orientation,
        showfliers=showfliers
    )
    
    # Set the figure title
    ax.set_title(figure_title)

    # Get summary statistics of the dataset
    logging.info("Summary statistics of the dataset:")
    summary = df.describe()

    # Convert the summary to a DataFrame that includes the index as a column
    summary_df = summary.reset_index()
    logging.info(f"\n{summary_df.to_string()}")
    
    # Move the legend outside the plotting area
    # Check if a legend exists
    try:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    except Exception as e:
        logging.debug("Legend does not exist.")
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "boxplot_summary.csv")
        saved_files = save_outputs({output_file: summary_df})

        # Also save the figure if specified
        figure_file = params.get("Figure_File", None)
        if figure_file:
            saved_files.update(save_outputs({figure_file: fig}))

        logging.info(f"Boxplot completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the figure and summary dataframe for in-memory workflows
        logging.info(
            "Returning figure and summary dataframe (not saving to file)"
        )
        return fig, summary_df


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python boxplot_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    # Set up logging for CLI usage
    logging.basicConfig(level=logging.INFO)
    
    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure and summary dataframe")