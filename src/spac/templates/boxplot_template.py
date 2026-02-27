"""
Platform-agnostic Boxplot template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Follows standardized output schema where figures are saved as directories.

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
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    show_plot: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, Union[str, List[str]]], Tuple[Any, pd.DataFrame]]:
    """
    Execute Boxplot analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Primary_Annotation": "cell_type",
            "Feature_s_to_Plot": ["CD4", "CD8"],
            "outputs": {
                "figures": {"type": "directory", "name": "figures"},
                "dataframe": {"type": "file", "name": "output.csv"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves figures to a directory
        and summary statistics to a CSV file. If False, returns the figure and
        summary dataframe directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory. All outputs will be saved relative to this directory.

    Returns
    -------
    dict or tuple
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {
                "figures": ["path/to/figures/boxplot.png"],  # List of figure paths
                "DataFrame": "path/to/output.csv"           # Single file path
            }
        If save_to_disk=False: Tuple of (matplotlib.figure.Figure, pd.DataFrame)
            containing the figure object and summary statistics dataframe

    Notes
    -----
    Output Structure:
    - Figures are saved in a directory (standardized for all figure outputs)
    - Summary statistics are saved as a single CSV file
    - When save_to_disk=False, objects are returned for programmatic use
    
    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["figure"])  # List of paths to saved plots
    >>> # ['./figures/boxplot.png']
    
    >>> # Get results in memory
    >>> fig, summary_df = run_from_json("params.json", save_to_disk=False)

    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    # Figures use directory type per standardized schema
    if "outputs" not in params:
        params["outputs"] = {
            "figures": {"type": "directory", "name": "figures"},
            "dataframe": {"type": "file", "name": "output.csv"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    annotation = params.get("Primary_Annotation", "None")
    second_annotation = params.get("Secondary_Annotation", "None")
    layer_to_plot = params.get("Table_to_Visualize", "Original")
    feature_to_plot = params.get("Feature_s_to_Plot", ["All"])
    log_scale = params.get("Value_Axis_Log_Scale", False)
    
    # Extract figure parameters with defaults
    figure_title = params.get("Figure_Title", "BoxPlot")
    figure_horizontal = params.get("Horizontal_Plot", False)
    fig_width = params.get("Figure_Width", 12)
    fig_height = params.get("Figure_Height", 8)
    fig_dpi = params.get("Figure_DPI", 300)
    font_size = params.get("Font_Size", 10)
    showfliers = params.get("Keep_Outliers", True)

    # Process parameters to match expected format
    # Convert "None" strings to actual None values
    layer_to_plot = None if layer_to_plot == "Original" else layer_to_plot
    second_annotation = None if second_annotation == "None" else second_annotation
    annotation = None if annotation == "None" else annotation

    # Convert horizontal flag to orientation string
    figure_orientation = "h" if figure_horizontal else "v"

    # Handle feature selection
    if isinstance(feature_to_plot, str):
        # Convert single string to list
        feature_to_plot = [feature_to_plot]

    # Check for "All" features selection
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
    
    # Apply tight layout to prevent label cutoff
    plt.tight_layout()
    
    if show_plot:
        plt.show()

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}
        
        # Package figure in a dictionary for directory saving
        # This ensures it's saved in a directory per standardized schema
        if "figures" in params["outputs"]:
            results_dict["figures"] = {"boxplot": fig}  # Dict triggers directory save
        
        # Check for DataFrames output (case-insensitive)
        if any(k.lower() == "dataframe" for k in params["outputs"].keys()):
            results_dict["dataframe"] = summary_df
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        logging.info("Boxplot analysis completed successfully.")
        return saved_files
    else:
        # Return objects directly for in-memory workflows
        logging.info(
            "Returning figure and summary dataframe for in-memory use"
        )
        return fig, summary_df


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python boxplot_template.py <params.json> [output_dir]",
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
        fig, summary_df = result
        print("\nReturned figure and summary dataframe for in-memory use")
        print(f"Figure size: {fig.get_size_inches()}")
        print(f"Summary shape: {summary_df.shape}")
        print("\nSummary statistics preview:")
        print(summary_df.head())