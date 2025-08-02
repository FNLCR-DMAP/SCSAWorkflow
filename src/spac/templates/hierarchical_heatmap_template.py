"""
Platform-agnostic Hierarchical Heatmap template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.hierarchical_heatmap_template import run_from_json
>>> run_from_json("examples/hierarchical_heatmap_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import hierarchical_heatmap
from spac.utils import check_feature
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
    Execute Hierarchical Heatmap analysis with parameters from JSON.
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
    annotation = params["Annotation"]
    layer_to_plot = params.get("Table_to_Visualize", "Original")
    features = params.get("Feature_s_", ["All"])
    standard_scale = params.get("Standard_Scale_", "None")
    z_score = params.get("Z_Score", "None")
    cluster_feature = params.get("Feature_Dendrogram", True)
    cluster_annotations = params.get("Annotation_Dendrogram", True)
    Figure_Title = params.get("Figure_Title", "Hierarchical Heatmap")
    fig_width = params.get("Figure_Width", 8)
    fig_height = params.get("Figure_Height", 8)
    fig_dpi = params.get("Figure_DPI", 300)
    font_size = params.get("Font_Size", 10)
    matrix_ratio = params.get("Matrix_Plot_Ratio", 0.8)
    swap_axes = params.get("Swap_Axes", False)
    rotate_label = params.get("Rotate_Label_", False)
    r_h_axis_dengrogram = params.get(
        "Horizontal_Dendrogram_Display_Ratio", 0.2
    )
    r_v_axis_dengrogram = params.get(
        "Vertical_Dendrogram_Display_Ratio", 0.2
    )
    v_min = params.get("Value_Min", "None")
    v_max = params.get("Value_Max", "None")
    color_map = params.get("Color_Map", 'seismic')

    # Use check_feature to validate features
    if len(features) == 1 and features[0] == "All":
        features = None
    else:
        check_feature(adata, features)

    if not swap_axes:
        features = None

    # Use text_to_value for parameter conversions
    standard_scale = text_to_value(
        standard_scale, to_int=True, param_name='Standard Scale'
    )
    layer_to_plot = text_to_value(
        layer_to_plot, default_none_text="Original"
    )
    z_score = text_to_value(z_score, param_name='Z Score')
    vmin = text_to_value(
        v_min, default_none_text="none", to_float=True, 
        param_name="Value Min"
    )
    vmax = text_to_value(
        v_max, default_none_text="none", to_float=True, 
        param_name="Value Max"
    )

    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': font_size})
    fig.set_size_inches(fig_width, fig_height)
    fig.set_dpi(fig_dpi)

    mean_intensity, clustergrid, dendrogram_data = hierarchical_heatmap(
        adata,
        annotation=annotation,
        features=features,
        layer=layer_to_plot,
        cluster_feature=cluster_feature,
        cluster_annotations=cluster_annotations,
        standard_scale=standard_scale,
        z_score=z_score,
        swap_axes=swap_axes,
        rotate_label=rotate_label,
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(r_h_axis_dengrogram, r_v_axis_dengrogram),
        vmin=vmin,
        vmax=vmax,
        cmap=color_map
    )
    print("Printing mean intensity data.")
    print(mean_intensity)
    print()
    print("Printing dendrogram data.")
    for data in dendrogram_data:
        print(data)
        print(dendrogram_data[data])

    # Ensure the mean_intensity index matches phenograph clusters
    row_clusters = adata.obs[annotation].astype(str).unique()
    mean_intensity[annotation] = mean_intensity.index.astype(str)

    # Reorder columns to move 'clusters' to the first position
    cols = mean_intensity.columns.tolist()
    cols = [annotation] + [col for col in cols if col != annotation]
    mean_intensity = mean_intensity[cols]

    # Show the modified plot
    clustergrid.ax_heatmap.set_title(Figure_Title)
    clustergrid.height = fig_height * matrix_ratio
    clustergrid.width = fig_width * matrix_ratio
    plt.close(1)

    if show_plot:
        plt.show()

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "plots.csv")
        saved_files = save_outputs({output_file: mean_intensity})

        print(
            f"Hierarchical Heatmap completed → "
            f"{saved_files[output_file]}"
        )
        return saved_files
    else:
        # Return the figure and dataframe directly for in-memory workflows
        print("Returning figure and dataframe (not saving to file)")
        return clustergrid.fig, mean_intensity


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python hierarchical_heatmap_template.py <params.json>",
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