"""
Platform-agnostic Sankey Plot template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.sankey_plot_template import run_from_json
>>> run_from_json("examples/sankey_plot_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import sankey_plot
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
) -> Union[Dict[str, str], None]:
    """
    Execute Sankey Plot analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns None
        since this template creates multiple plot files. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    dict or None
        If save_results=True: Dictionary of saved file paths
        If save_results=False: None (plots are displayed but not saved)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    annotation_columns = [
        params.get("Source_Annotation_Name", "None"),
        params.get("Target_Annotation_Name", "None")
    ]
    dpi = params.get("Figure_DPI", 300)
    width_num = params.get("Figure_Width_inch", 6)
    scale = dpi / 96
    width_in_pixels = width_num / scale * dpi

    height_num = params.get("Figure_Height_inch", 6)
    height_in_pixels = height_num / scale * dpi

    # sort_asscend = True   # unused variable
    source_color_map = params.get("Source_Annotation_Color_Map", "tab20")
    target_color_map = params.get("Target_Annotation_Color_Map", "tab20b")

    sankey_font = params.get("Font_Size", 12)

    target_annotation = text_to_value(annotation_columns[1])
    source_annotation = text_to_value(annotation_columns[0])

    fig = sankey_plot(
            adata=adata,
            source_annotation=source_annotation,
            target_annotation=target_annotation,
            source_color_map=source_color_map,
            target_color_map=target_color_map,
            sankey_font=sankey_font
        )

    # Customize the Sankey diagram layout
    fig.update_layout(
        width=width_in_pixels,  # Specify the width in pixels
        height=height_in_pixels   # Specify the height in pixels
    )

    # Show the plot with the specified display options
    print(fig)

    # Use output prefix to avoid conflicts
    output_prefix = params.get("Output_File", "sankey")
    image_path = f"{output_prefix}_diagram.png"

    pio.write_image(
        fig,
        image_path,
        width=width_in_pixels,  # Specify the width in pixels
        height=height_in_pixels,
        engine='kaleido',  # Use the 'kaleido' engine for high DPI images
        scale=scale
    )

    img = plt.imread(image_path)
    static, axs = plt.subplots(1, 1, figsize=(width_num, height_num), dpi=dpi) 

    # Load and display the image using Matplotlib
    axs.imshow(img)
    axs.axis('off')
    if show_plot:
        plt.show()

    if show_plot:
        fig.show()

    # Handle saving if requested
    if save_results:
        saved_files = {}
        output_prefix = params.get("Output_File", "sankey")
        
        # Save the static plot
        static_file = f"{output_prefix}_static.png"
        static.savefig(static_file, dpi=dpi, bbox_inches='tight')
        saved_files[static_file] = static_file
        
        # Save the interactive plot
        interactive_file = f"{output_prefix}_interactive.html"
        pio.write_html(fig, interactive_file)
        saved_files[interactive_file] = interactive_file
        
        # Save the intermediate PNG that was created
        saved_files[image_path] = image_path
        
        # Close figures after saving
        plt.close(static)
            
        print(f"Sankey Plot completed → {list(saved_files.keys())}")
        return saved_files
    
    return None


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python sankey_plot_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nPlots displayed (not saved)")