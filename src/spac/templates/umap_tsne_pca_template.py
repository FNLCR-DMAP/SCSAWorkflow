"""
Platform-agnostic UMAP\\tSNE\\PCA Visualization template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.umap_tsne_pca_template import run_from_json
>>> run_from_json("examples/umap_tsne_pca_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import dimensionality_reduction_plot
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
    Execute UMAP\\tSNE\\PCA Visualization analysis with parameters from JSON.
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

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "plots.png")
        # Ensure proper extension
        if not output_file.endswith(('.png', '.pdf', '.svg')):
            output_file = output_file + '.png'
        
        fig.savefig(output_file, dpi=fig_dpi, bbox_inches='tight')
        saved_files = {output_file: output_file}
        
        print(
            f"UMAP\\tSNE\\PCA Visualization completed → "
            f"{saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the figure and dataframe directly for in-memory workflows
        print("Returning figure (not saving to file)")
        # Note: This template doesn't produce a dataframe, just a figure
        return fig, None


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python umap_tsne_pca_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure")