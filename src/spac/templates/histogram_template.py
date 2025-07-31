"""
Platform-agnostic Histogram template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.histogram_template import run_from_json
>>> run_from_json("examples/histogram_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import histogram
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True,
    show_plot: bool = False
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Histogram analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the figure and
        dataframe directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is False.

    Returns
    -------
    dict or tuple
        If save_results=True: Dictionary of saved file paths
        If save_results=False: Tuple of (figure, dataframe)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    feature = text_to_value(params.get("Feature", "None"))
    annotation = text_to_value(params.get("Annotation", "None"))
    layer = params.get("Table_", "Original")
    group_by = params.get("Group_by", "None")
    together = params.get("Together", True)
    fig_width = params.get("Figure_Width", 8)
    fig_height = params.get("Figure_Height", 6)
    font_size = params.get("Font_Size", 12)
    fig_dpi = params.get("Figure_DPI", 300)
    legend_location = params.get("Legend_Location", "best")
    legend_in_figure = params.get("Legend_in_Figure", False)
    take_X_log = params.get("Take_X_Log", False)
    take_Y_log = params.get("Take_Y_log", False)
    multiple = params.get("Multiple", "dodge")
    shrink = params.get("Shrink_Number", 1)
    bins = params.get("Bins", "auto")
    alpha = params.get("Bin_Transparency", 0.75)
    stat = params.get("Stat", "count")
    x_rotate = params.get("X_Axis_Label_Rotation", 0)
    histplot_by = params.get("Plot_By", "Annotation")

    # Close all existing figures to prevent extra plots
    plt.close('all')
    existing_fig_nums = plt.get_fignums()

    plt.rcParams.update({'font.size': font_size})

    # Adjust feature and annotation based on histplot_by
    if histplot_by == "Annotation":
        feature = None
    else:
        annotation = None

    # If both feature and annotation are None, set default
    if feature is None and annotation is None:
        if histplot_by == "Annotation":
            if adata.obs.columns.size > 0:
                annotation = adata.obs.columns[0]
                print(
                    f'No annotation specified. Using the first annotation '
                    f'"{annotation}" as default.'
                )
            else:
                raise ValueError(
                    'No annotations available in adata.obs to plot.'
                )
        else:
            if adata.var_names.size > 0:
                feature = adata.var_names[0]
                print(
                    f'No feature specified. Using the first feature '
                    f'"{feature}" as default.'
                )
            else:
                raise ValueError(
                    'No features available in adata.var_names to plot.'
                )

    # Validate and set bins
    if feature is not None:
        bins = text_to_value(
            bins,
            default_none_text="auto",
            to_int=True,
            param_name="bins"
        )
        if bins is None:
            num_rows = adata.X.shape[0]
            bins = max(int(2 * (num_rows ** (1/3))), 1)
        elif bins <= 0:
            raise ValueError(
                f'Bins should be a positive integer. Received "{bins}"'
            )
    elif annotation is not None:
        if take_X_log:
            take_X_log = False
            print(
                "Warning: Take X log should only apply to feature. "
                "Setting Take X Log to False."
            )
        if bins != 'auto':
            bins = 'auto'
            print(
                "Warning: Bin number should only apply to feature. "
                "Setting bin number calculation to auto."
            )

    if (x_rotate < 0) or (x_rotate > 360):
        raise ValueError(
            f'The X label rotation should fall within 0 to 360 degree. '
            f'Received "{x_rotate}".'
        )

    # Initialize the x-variable before the loop
    if histplot_by == "Annotation":
        x_var = annotation
    else:
        x_var = feature

    result = histogram(
        adata=adata,
        feature=feature,
        annotation=annotation,
        layer=text_to_value(layer, "Original"),
        group_by=text_to_value(group_by),
        together=together,
        ax=None,
        x_log_scale=take_X_log,
        y_log_scale=take_Y_log,
        multiple=multiple,
        shrink=shrink,
        bins=bins,
        alpha=alpha,
        stat=stat
    )

    fig = result["fig"]
    axs = result["axs"]
    df_counts = result["df"]

    # Set figure size and dpi
    fig.set_size_inches(fig_width, fig_height)
    fig.set_dpi(fig_dpi)

    # Ensure axes is a list
    if isinstance(axs, list):
        axes = axs
    else:
        axes = [axs]

    # Close any extra figures created during the histogram call
    fig_nums_after = plt.get_fignums()
    new_fig_nums = [
        num for num in fig_nums_after if num not in existing_fig_nums
    ]
    histogram_fig_num = fig.number

    for num in new_fig_nums:
        if num != histogram_fig_num:
            plt.close(plt.figure(num))
            print(f"Closed extra figure {num}")

    # Process each axis
    for ax in axes:
        if feature:
            print(f'Plotting Feature: "{feature}"')
        if ax.get_legend() is not None:
            if legend_in_figure:
                sns.move_legend(ax, legend_location)
            else:
                sns.move_legend(
                    ax, legend_location, bbox_to_anchor=(1, 1)
                )

        # Rotate x labels
        ax.tick_params(axis='x', rotation=x_rotate)

    # Set titles based on group_by
    if text_to_value(group_by):
        if together:
            for ax in axes:
                ax.set_title(
                    f'Histogram of "{x_var}" grouped by "{group_by}"'
                )
        else:
            # compute unique groups directly from adata.obs.
            unique_groups = adata.obs[
                text_to_value(group_by)
            ].dropna().unique()
            if len(axes) != len(unique_groups):
                print(
                    "Warning: Number of axes does not match number of "
                    "groups. Titles may not correspond correctly."
                )
            for ax, grp in zip(axes, unique_groups):
                ax.set_title(
                    f'Histogram of "{x_var}" for group: "{grp}"'
                )
    else:
        for ax in axes:
            ax.set_title(f'Count plot of "{x_var}"')

    plt.tight_layout()

    print("Displaying top 10 rows of histogram dataframe:")
    print(df_counts.head(10))

    if show_plot:
        plt.show()

    plt.close('all')

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "plots.csv")
        saved_files = save_outputs({output_file: df_counts})

        print(f"Histogram completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the figure and dataframe directly for in-memory workflows
        print("Returning figure and dataframe (not saving to file)")
        return fig, df_counts


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram_template.py <params.json>")
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure and dataframe")