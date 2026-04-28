"""
Platform-agnostic Histogram template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.histogram_template import run_from_json
>>> run_from_json("examples/histogram_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import histogram
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    show_plot: bool = False,
    output_dir: str = None,
) -> Union[Dict[str, Union[str, List[str]]], Tuple[Any, pd.DataFrame]]:
    """
    Execute Histogram analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Plot_By": "Annotation",
            "Annotation": "cell_type",
            ...
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"},
                "figures": {"type": "directory", "name": "figures_dir"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the figure and
        dataframe directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is False.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory.

    Returns
    -------
    dict or tuple
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: Tuple of (figure, dataframe)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")

    # Ensure outputs configuration exists with standardized defaults
    if "outputs" not in params:
        params["outputs"] = {
            "dataframe": {"type": "file", "name": "dataframe.csv"},
            "figures": {"type": "directory", "name": "figures_dir"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    feature = text_to_value(params.get("Feature", "None"))
    annotation = text_to_value(params.get("Annotation", "None"))
    layer = params.get("Table_", "Original")
    group_by = text_to_value(params.get("Group_by", "None"))
    max_groups = params.get("Max_Groups", 20)
    together = params.get("Together", True)
    fig_width = params.get("Figure_Width", "auto")
    fig_height = params.get("Figure_Height", "auto")
    font_size = params.get("Font_Size", 12)
    fig_dpi = params.get("Figure_DPI", 300)
    legend_location = params.get("Legend_Location", "best")
    legend_in_figure = params.get("Legend_in_Figure", False)
    take_X_log = params.get("Take_X_Log", False)
    take_Y_log = params.get("Take_Y_log", False)
    multiple = params.get("Multiple", "dodge")
    element = params.get("Element", "bars")
    shrink = params.get("Shrink_Number", 1)
    bins = params.get("Bins", "auto")
    alpha = params.get("Bin_Transparency", 0.75)
    stat = params.get("Stat", "count")
    x_rotate = params.get("X_Axis_Label_Rotation", 0)
    histplot_by = params.get("Plot_By", "Annotation")
    facet = params.get("Facet", False)
    facet_ncol = params.get("Facet_Ncol", "auto")

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
                logger.info(
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
                logger.info(
                    f'No feature specified. Using the first feature '
                    f'"{feature}" as default.'
                )
            else:
                raise ValueError(
                    'No features available in adata.var_names to plot.'
                )

    # Bins use a strict template contract in feature mode:
    # "auto" or a positive integer. Loose null-like values are intentionally
    # not treated as aliases here.
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
            logger.warning(
                "Take X log should only apply to feature. "
                "Setting Take X Log to False."
            )
        if bins != 'auto':
            bins = 'auto'
            logger.warning(
                "Bin number should only apply to feature. "
                "Setting bin number calculation to auto."
            )


    if group_by and together:
        multiple = str(multiple).strip().lower()
    element = str(element).strip().lower()
    stat = str(stat).strip().lower()

    # Figure_Width and Figure_Height use "auto" for template defaults.
    # In facet mode, it is forwarded as None to derive layout geometry.
    # In non-facet mode, it falls back to 8x6 inches.
    fig_width = text_to_value(
        fig_width,
        default_none_text="auto",
        value_to_convert_to=None if facet else 8,
        to_float=True,
        param_name="Figure_Width"
    )
    fig_height = text_to_value(
        fig_height,
        default_none_text="auto",
        value_to_convert_to=None if facet else 6,
        to_float=True,
        param_name="Figure_Height"
    )
    if fig_width is not None and fig_height is not None:
        if fig_width <= 0 or fig_height <= 0:
            raise ValueError(
                f'Figure_Width/Height should be a positive number.'
                f'Received "{fig_width}"/"{fig_height}".'
            )
    if fig_dpi <= 0:
        raise ValueError(
            f'Figure_DPI should be a positive number. Received "{fig_dpi}".'
        )

    # Validate x-axis label rotation
    if (x_rotate < 0) or (x_rotate > 360):
        raise ValueError(
            f'The X label rotation should fall within 0 to 360 degree. '
            f'Received "{x_rotate}".'
        )

    # Max_Groups applies only when Group_by is set.
    # It accepts a positive integer or "unlimited".
    # Missing values default to 20.
    if group_by:
        parsed_max_groups = max_groups
        if parsed_max_groups != "unlimited":
            parsed_max_groups = text_to_value(
                parsed_max_groups,
                value_to_convert_to=20,
                to_int=True,
                param_name="Max_Groups",
            )
            if parsed_max_groups <= 0:
                raise ValueError(
                    f'Max_Groups should be a positive integer or "unlimited". '
                    f'Received "{parsed_max_groups}".'
                )

    # Facet requires Group_by and forbids Together=True.
    # Facet_Ncol accepts "auto" or a positive integer.
    if facet:
        if group_by is None:
            raise ValueError(
                'Facet is True but Group_by is not specified. '
                'Please specify Group_by when using Facet.'
            )
        if together:
            raise ValueError(
                'Together and Facet cannot both be True. Please set one to False.'
            )
        facet_ncol = text_to_value(
            facet_ncol,
            default_none_text="auto",
            to_int=True,
            param_name="Facet_Ncol"
        )
        if facet_ncol is not None:
            if facet_ncol <= 0:
                raise ValueError(
                    f'Facet_Ncol must be a positive integer or "auto". '
                    f'Received "{facet_ncol}".'
                )

    # Initialize the x-variable before the loop
    if histplot_by == "Annotation":
        x_var = annotation
    else:
        x_var = feature

    # Assemble validated histogram kwargs right before the plotting call.
    hist_kwargs = dict(
        element=element,
        shrink=shrink,
        bins=bins,
        alpha=alpha,
        stat=stat,
    )
    if group_by and together:
        hist_kwargs["multiple"] = multiple
    if group_by:
        hist_kwargs["max_groups"] = parsed_max_groups
    if facet:
        hist_kwargs["facet_ncol"] = facet_ncol
        hist_kwargs["facet_fig_width"] = fig_width
        hist_kwargs["facet_fig_height"] = fig_height
        hist_kwargs["facet_tick_rotation"] = x_rotate

    result = histogram(
        adata=adata,
        feature=feature,
        annotation=annotation,
        layer=text_to_value(layer, "Original"),
        group_by=group_by,
        together=together,
        ax=None,
        x_log_scale=take_X_log,
        y_log_scale=take_Y_log,
        facet=facet,
        **hist_kwargs,
    )

    fig = result["fig"]
    axs = result["axs"]
    df_counts = result["df"]

    # Set figure size and dpi
    if fig_width is not None and fig_height is not None:
        fig.set_size_inches(fig_width, fig_height)
        logger.info(f"Set figure size to {fig_width}x{fig_height} inches.")
    fig.set_dpi(fig_dpi)
    logger.info(f"Set figure DPI to {fig_dpi}.")

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
            logger.debug(f"Closed extra figure {num}")

    # Process each axis
    for ax in axes:
        if feature:
            logger.info(f'Plotting Feature: "{feature}"')
        if ax.get_legend() is not None:
            if legend_in_figure:
                sns.move_legend(ax, legend_location)
            else:
                sns.move_legend(
                    ax, legend_location, bbox_to_anchor=(1, 1)
                )

        # Rotate x labels
        ax.tick_params(axis='x', rotation=x_rotate)
        if x_rotate:
            for label in ax.get_xticklabels():
                label.set_rotation_mode('anchor')
                label.set_horizontalalignment('right')

    # Set titles based on group_by and facet
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
                logger.warning(
                    "Number of axes does not match number of "
                    "groups. Titles may not correspond correctly."
                )
            if facet:
                fig.suptitle(
                    f'Histogram of "{x_var}" faceted by "{group_by}"'
                )
                ax_title_prefix = f'Group'
            else:
                ax_title_prefix = f'Histogram of "{x_var}" for group'
            for ax, grp in zip(axes, unique_groups):
                ax.set_title(
                    f'{ax_title_prefix}: "{grp}"'
                )
    else:
        for ax in axes:
            ax.set_title(f'Count plot of "{x_var}"')

    # Adjust layout to prevent title overlap
    if facet:
        rows = len({round(ax.get_position().y0, 3) for ax in axes})
        fig.tight_layout(
            rect=[
                min(0.030, 0.02 + 0.0025 * rows),
                max(0.022, 0.036 - 0.003 * rows),
                min(0.992, 0.98 + 0.0025 * rows),
                max(0.969, 0.975 - 0.001 * rows),
            ],
            pad=max(0.35, 0.6 - 0.05 * rows),
            h_pad=max(0.2, 0.43 - 0.04 * rows) * 6,
            w_pad=max(0.2, 0.43 - 0.04 * rows) * 6,
        )
    else:
        fig.tight_layout()

    logger.info("Displaying top 10 rows of histogram dataframe:")
    print(df_counts.head(10))

    if show_plot:
        plt.show()

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}

        # Check for dataframe output
        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = df_counts

        # Check for figures output
        if "figures" in params["outputs"]:
            results_dict["figures"] = {"histogram": fig}

        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        plt.close('all')

        logger.info("Histogram analysis completed successfully.")
        return saved_files
    else:
        # Return the figure and dataframe directly for in-memory workflows
        logger.info("Returning figure and dataframe for in-memory use")
        return fig, df_counts


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python histogram_template.py <params.json> [output_dir]",
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
        print("\nReturned figure and dataframe")
