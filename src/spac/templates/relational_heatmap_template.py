"""
Platform-agnostic Relational Heatmap template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.relational_heatmap_template import run_from_json
>>> run_from_json("examples/relational_heatmap_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import relational_heatmap
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Relational Heatmap analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the figure and
        dataframe directly for in-memory workflows. Default is True.

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
    annotation_columns = [
        params.get("Source_Annotation_Name", "None"), 
        params.get("Target_Annotation_Name", "None")
    ]
    dpi = params.get("Figure_DPI", 300)
    width_in = params.get("Figure_Width_inch", 8)
    height_in = params.get("Figure_Height_inch", 10)
    width_px = width_in * 96
    print(width_px)
    height_px = height_in * 96
    print(height_px)
    
    scale = dpi / 96

    font_size = params.get("Font_Size", 8)
    colormap = params.get("Colormap", "darkmint")

    source_annotation = text_to_value(annotation_columns[0])

    target_annotation = text_to_value(annotation_columns[1]) 

    result_dict = relational_heatmap(
        adata=adata,
        source_annotation=source_annotation,
        target_annotation=target_annotation,
        color_map=colormap,
        font_size=font_size
    )
    
    # extract results from function return
    rhmap_file_name = result_dict['file_name']
    rhmap_data = result_dict['data']
    fig = result_dict['figure']

    # Generate temporary image name for Plotly export
    import tempfile
    tmp_image_name = tempfile.mktemp(suffix='.png')
    
    pio.write_image(
        fig,
        tmp_image_name,
        width=width_px,  # Specify the width in pixels
        height=height_px,
        engine='kaleido',  # Use the 'kaleido' engine for high DPI images
        scale=scale
    )

    img = plt.imread(tmp_image_name)
    static, axs = plt.subplots(
        1, 1, figsize=(width_in, height_in), dpi=dpi
    )

    # Load and display the image using Matplotlib
    axs.imshow(img)
    axs.axis('off')
    
    # Display the matplotlib static figure
    plt.show()

    # Clean up temp file
    import os
    if os.path.exists(tmp_image_name):
        os.remove(tmp_image_name)

    # Display the Plotly interactive figure
    fig.show()

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", rhmap_file_name)
        if not output_file.endswith('.csv'):
            output_file = output_file + '.csv'
        
        saved_files = save_outputs({output_file: rhmap_data})
        
        # Also save the static plot
        plot_file = output_file.replace('.csv', '.png')
        static.savefig(plot_file, dpi=dpi, bbox_inches='tight')
        saved_files[plot_file] = plot_file
        
        print(
            f"Relational Heatmap completed → {list(saved_files.keys())}"
        )
        return saved_files
    else:
        # Return the figure and dataframe directly for in-memory workflows
        print("Returning figure and dataframe (not saving to file)")
        return static, rhmap_data


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python relational_heatmap_template.py <params.json>",
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