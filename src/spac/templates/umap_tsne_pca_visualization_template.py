"""
Platform-agnostic UMAP\\tSNE\\PCA Visualization template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

Usage
-----
>>> from spac.templates.umap_tsne_pca_template import run_from_json
>>> run_from_json("examples/umap_tsne_pca_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional, List
import matplotlib.pyplot as plt
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import dimensionality_reduction_plot
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
) -> Union[Dict[str, Union[str, List[str]]], plt.Figure]:
    """
    Execute UMAP\\tSNE\\PCA Visualization analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Color_By": "Annotation",
            "Annotation_to_Highlight": "cell_type",
            "Dimension_Reduction_Method": "umap",
            ...
            "outputs": {
                "figures": {"type": "directory", "name": "figures_dir"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the figure
        directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory.

    Returns
    -------
    dict or Figure
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: The matplotlib figure
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
            "figures": {"type": "directory", "name": "figures_dir"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    annotation = params.get("Annotation_to_Highlight", "None")
    feature = params.get("Feature_to_Highlight", "None")
    layer = params.get("Table", "Original")
    method = params.get("Dimension_Reduction_Method", "umap")
    fig_width = params.get("Figure_Width", 12)
    fig_height = params.get("Figure_Height", 12)
    font_size = params.get("Font_Size", 12)
    fig_dpi = params.get("Figure_DPI", 300)
    legend_location = params.get("Legend_Location", "best")
    legend_label_size = params.get("Legend_Font_Size", 16)
    legend_marker_scale = params.get("Legend_Marker_Size", 5.0)
    color_by = params.get("Color_By", "Annotation")
    point_size = params.get("Dot_Size", 1)
    v_min = params.get("Value_Min", "None")
    v_max = params.get("Value_Max", "None")

    feature = text_to_value(feature)
    annotation = text_to_value(annotation)

    if color_by == "Annotation":
        feature = None
    else:
        annotation = None

    # Store the original value of layer
    layer_input = layer

    layer = text_to_value(layer, default_none_text="Original")

    vmin = text_to_value(
        v_min,
        default_none_text="None",
        value_to_convert_to=None,
        to_float=True,
        param_name="Value Min"
    )

    vmax = text_to_value(
        v_max,
        default_none_text="None",
        value_to_convert_to=None,
        to_float=True,
        param_name="Value Max"
    )

    plt.rcParams.update({'font.size': font_size})

    fig, ax = dimensionality_reduction_plot(
        adata=adata,
        method=method,
        annotation=annotation,
        feature=feature,
        layer=layer,
        point_size=point_size,
        vmin=vmin,
        vmax=vmax
    )

    if color_by == "Annotation":
        title = annotation
    else:
        title = f'Table:"{layer_input}" \n Feature:"{feature}"'
    ax.set_title(title)

    fig = ax.get_figure()

    fig.set_size_inches(
        fig_width,
        fig_height
    )
    fig.set_dpi(fig_dpi)

    legend = ax.get_legend()
    has_legend = legend is not None

    if has_legend:
        ax.legend(
            loc=legend_location,
            bbox_to_anchor=(1, 0.5),
            fontsize=legend_label_size,
            markerscale=legend_marker_scale
        )

    plt.tight_layout()
    
    if show_plot:
        plt.show()

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}

        # Check for figures output
        if "figures" in params["outputs"]:
            results_dict["figures"] = {f"{method}_plot": fig}

        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        plt.close(fig)

        logger.info(
            f"{method.upper()} Visualization completed successfully."
        )
        return saved_files
    else:
        # Return the figure directly for in-memory workflows
        logger.info("Returning figure for in-memory use")
        return fig


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python umap_tsne_pca_template.py <params.json> [output_dir]",
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
        print("\nReturned figure")
