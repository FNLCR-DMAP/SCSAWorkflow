"""
Platform-agnostic Visualize Nearest Neighbor template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.visualize_nearest_neighbor_template import (
...     run_from_json
... )
>>> run_from_json("examples/visualize_nearest_neighbor_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Tuple, List
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import visualize_nearest_neighbor
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
    Execute Visualize Nearest Neighbor analysis with parameters from JSON.
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
        If save_results=False: Tuple of (figure(s), dataframe)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    # Use direct dictionary access for required parameters
    # Will raise KeyError if missing
    annotation = params["Annotation"]
    source_label = params["Source_Anchor_Cell_Label"]

    # Use .get() with defaults for optional parameters from JSON template
    image_id = params.get("ImageID", "None")
    method = params.get("Plot_Method", "numeric")
    plot_type = params.get("Plot_Type", "boxen")
    target_label = params.get("Target_Cell_Label", "All")
    distance_key = params.get(
        "Nearest_Neighbor_Associated_Table", "spatial_distance"
    )
    log_scale = params.get("Log_Scale", False)
    facet_plot = params.get("Facet_Plot", False)
    x_axis_title_rotation = params.get("X_Axis_Label_Rotation", 0)
    shared_x_axis_title = params.get("Shared_X_Axis_Title_", True)
    x_axis_title_fontsize = params.get("X_Axis_Title_Font_Size", "None")

    defined_color_map = text_to_value(
        params.get("Defined_Color_Mapping", "None"),
        param_name="Define Label Color Mapping"
    )
    annotation_colorscale = "rainbow"

    fig_width = params.get("Figure_Width", 12)
    fig_height = params.get("Figure_Height", 6)
    fig_dpi = params.get("Figure_DPI", 300)
    global_font_size = params.get("Font_Size", 12)
    fig_title = (
        f'Nearest Neighbor Distance Distribution\nMeasured from '
        f'"{source_label}"'
    )

    image_id = text_to_value(
        image_id,
        default_none_text="None",
        value_to_convert_to=None
    )

    # If target_label is None, it means "All distance columns"
    # If it's a comma-separated string (e.g. "Stroma,Immune"),
    # split into a list
    target_label = text_to_value(
        target_label,
        default_none_text="All",
        value_to_convert_to=None
    )

    if target_label is not None:
        distance_to_processed = [x.strip() for x in target_label.split(",")]
    else:
        distance_to_processed = None

    x_axis_title_fontsize = text_to_value(
        x_axis_title_fontsize,
        default_none_text="None",
        to_int="True"
    )

    # Configure Matplotlib font size
    plt.rcParams.update({'font.size': global_font_size})

    # If facet_plot=True but no valid stratify column => revert to
    # single figure
    if facet_plot and image_id is None:
        warning_message = (
            "Facet plotting was requested, but there is no annotation "
            "to group by. Switching to a single-figure display."
        )
        print(warning_message)
        facet_plot = False

    result_dict = visualize_nearest_neighbor(
        adata=adata,
        annotation=annotation,
        spatial_distance=distance_key,
        distance_from=source_label,
        distance_to=distance_to_processed,
        method=method,
        plot_type=plot_type,
        stratify_by=image_id,
        facet_plot=facet_plot,
        log=log_scale,
        annotation_colorscale=annotation_colorscale,
        defined_color_map=defined_color_map,
    )

    # Extract the data and figure(s)
    df_long = result_dict["data"]
    figs_out = result_dict["fig"]  # Single Figure or List of Figures
    palette_hex = result_dict["palette"]
    axes_out = result_dict["ax"]

    print("Summary statistics of the dataset:")
    print(df_long.describe())

    # Customize figure legends & X-axis rotation
    legend_labels = (
        distance_to_processed or df_long["group"].unique().tolist()
    )
    legend_labels = (
        legend_labels if distance_to_processed else sorted(legend_labels)
    )

    handles = [
        mpatches.Patch(
            facecolor=palette_hex[label],
            edgecolor='none',
            label=label
        )
        for label in legend_labels
    ]

    def _flatten_axes(ax_input):
        if isinstance(ax_input, Axes):
            return [ax_input]
        if isinstance(ax_input, (list, tuple, np.ndarray)):
            return [
                ax for ax in np.ravel(ax_input) if isinstance(ax, Axes)
            ]
        return []

    flat_axes_list = _flatten_axes(axes_out)
    shared_x_title_applied_to_fig = None

    if flat_axes_list:
        # Attach legend to the last axis
        flat_axes_list[-1].legend(
            handles=handles,
            title="Target phenotype",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )

        # X-Axis Title Handling
        current_x_label_text = ""
        if flat_axes_list[0].get_xlabel():
            current_x_label_text = flat_axes_list[0].get_xlabel()

        if not current_x_label_text:
            current_x_label_text = (
                f"Log({distance_key})" if log_scale else distance_key
            )
        if not current_x_label_text:
            current_x_label_text = "Distance"  # Ultimate fallback

        effective_fontsize = (
            x_axis_title_fontsize if x_axis_title_fontsize is not None
            else global_font_size
        )

        if (facet_plot and shared_x_axis_title and
                isinstance(figs_out, plt.Figure)):
            for ax_item in flat_axes_list:
                ax_item.set_xlabel('')

            sup_ha_align = 'center'
            if 0 < x_axis_title_rotation % 360 < 180:
                sup_ha_align = 'right'
            elif 180 < x_axis_title_rotation % 360 < 360:
                sup_ha_align = 'left'

            figs_out.supxlabel(
                current_x_label_text, y=0.02, fontsize=effective_fontsize,
                rotation=x_axis_title_rotation, ha=sup_ha_align
            )
            shared_x_title_applied_to_fig = figs_out

        else:  # Apply to individual subplot x-axis titles
            for ax_item in flat_axes_list:
                label_object = ax_item.xaxis.get_label()
                if not label_object.get_text():  # If no label, set it
                    ax_item.set_xlabel(current_x_label_text)
                    label_object = ax_item.xaxis.get_label()

                if label_object.get_text():  # Configure if actual label
                    label_object.set_rotation(x_axis_title_rotation)
                    label_object.set_fontsize(effective_fontsize)
                    ha_align_val = 'center'
                    if 0 < x_axis_title_rotation % 360 < 180:
                        ha_align_val = 'right'
                    elif 180 < x_axis_title_rotation % 360 < 360:
                        ha_align_val = 'left'
                    label_object.set_ha(ha_align_val)

    # Stratification Info
    if image_id is not None and image_id in df_long.columns:
        unique_vals = df_long[image_id].unique()
        n_unique = len(unique_vals)

        if n_unique == 0:
            print(
                f"[WARNING] The annotation '{image_id}' has 0 unique "
                f"values or is empty. No data to plot => Potential "
                f"empty plot."
            )
        elif n_unique == 1 and facet_plot:
            print(
                f"[INFO] The annotation '{image_id}' has only one unique "
                f"value ({unique_vals[0]}). Facet plot will resemble a "
                f"single plot."
            )
        elif n_unique > 1:
            print(
                f"The annotation '{image_id}' has {n_unique} unique "
                f"values: {unique_vals}"
            )

    # Figure Configuration & Display
    def _title_main(fig, title):
        """
        Sets a bold, centered main title on the figure, and
        adjusts figure size and layout accordingly.
        """
        fig.set_size_inches(fig_width, fig_height)
        fig.set_dpi(fig_dpi)
        fig.suptitle(
            title,
            fontsize=global_font_size + 4,
            weight='bold',
            x=0.5,  # center horizontally
            horizontalalignment='center'
        )

    def _label_each_figure(fig_list, categories):
        """
        Adds a title to each figure, typically used when multiple
        separate figures are returned (one per category).
        """
        for fig, cat in zip(fig_list, categories):
            if fig:
                _title_main(fig, f"{fig_title}\n{image_id}: {cat}")
                # Adjust top for the suptitle
                fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.96])
                if show_plot:
                    plt.show()

    # Determine the actual distance column name used in df_long for summary
    distance_col = (
        "log_distance" if "log_distance" in df_long.columns else "distance"
    )

    # Displaying Figures
    cat_list = []
    if image_id and (image_id in df_long.columns):
        if pd.api.types.is_categorical_dtype(df_long[image_id]):
            cat_list = list(df_long[image_id].cat.categories)
        else:
            cat_list = df_long[image_id].unique().tolist()

    # Track figures for optional saving
    figures = []

    if isinstance(figs_out, list) and not facet_plot and \
            cat_list and len(figs_out) == len(cat_list):
        # Scenario: Multiple separate figures, one per category
        # (non-faceted)
        figures = figs_out
        _label_each_figure(figs_out, cat_list)
        if show_plot:
            plt.show()
    else:
        # Scenario: Single figure (faceted) or list of figures not
        # matching categories
        figures_to_display = (
            figs_out if isinstance(figs_out, list) else [figs_out]
        )
        figures = figures_to_display
        for fig_item_to_display in figures_to_display:
            if fig_item_to_display is not None:
                _title_main(fig_item_to_display, fig_title)

                bottom_padding = 0.01
                # Make space for shared x-title
                if fig_item_to_display is shared_x_title_applied_to_fig:
                    bottom_padding = 0.01  # Adjusted from 0.05

                top_padding = 0.99  # Adjusted from 0.90

                # rect=[left, bottom, right, top]
                fig_item_to_display.tight_layout(
                    rect=[0.01, bottom_padding, 0.99, top_padding]
                )
                if show_plot:
                    plt.show()

    # summary statistics
    # 1) Per-group summary
    df_summary_group = (
        df_long
        .groupby("group")[distance_col]
        .describe()
        .reset_index()
    )

    # 2) Per-group-and-stratify, if image_id is valid
    if image_id and (image_id in df_long.columns):
        df_summary_group_strat = (
            df_long
            .groupby([image_id, "group"])[distance_col]
            .describe()
            .reset_index()
        )
    else:
        df_summary_group_strat = None

    if df_summary_group_strat is not None:
        print(f"\nSummary by group(target phenotypes) AND '{image_id}':")
        print(df_summary_group_strat)
    else:
        print("\nSummary: By group(target phenotypes) only")
        print(df_summary_group)

    # CSV Output
    final_df = (
        df_summary_group_strat if df_summary_group_strat is not None
        else df_summary_group
    )

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get(
            "Output_File", "nearest_neighbor_plots.csv"
        )
        saved_files = save_outputs({output_file: final_df})

        print(f"\nSaved summary statistics to '{output_file}'.")
        print(
            f"Visualize Nearest Neighbor completed → "
            f"{saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the figure(s) and dataframe directly for in-memory
        # workflows
        print("Returning figure(s) and dataframe (not saving to file)")
        # If single figure, return it directly; if multiple, return list
        if len(figures) == 1:
            return figures[0], final_df
        else:
            return figures, final_df


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python visualize_nearest_neighbor_template.py "
            "<params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure(s) and dataframe")
