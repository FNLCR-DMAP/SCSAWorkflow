"""
Platform-agnostic Interactive Spatial Plot template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.interactive_spatial_plot_template import run_from_json
>>> run_from_json("examples/interactive_spatial_plot_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
import pandas as pd
import plotly.io as pio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import SPAC functions from NIDAP template
from spac.visualization import interactive_spatial_plot
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Optional[Dict[str, str]]:
    """
    Execute Interactive Spatial Plot analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. Default is True.

    Returns
    -------
    dict or None
        If save_results=True: Dictionary of saved file paths
        If save_results=False: None (plots are shown interactively)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    color_by = params["Color_By"]
    annotations = params.get("Annotation_s_to_Highlight", [""])
    feature = params.get("Feature_to_Highlight", "None")
    layer = params.get("Table", "Original")

    dot_size = params.get("Dot_Size", 1.5)
    dot_transparency = params.get("Dot_Transparency", 0.75)
    color_map = params.get("Feature_Color_Scale", "balance")
    desired_width_in = params.get("Figure_Width", 6)
    desired_height_in = params.get("Figure_Height", 4)
    dpi = params.get("Figure_DPI", 200)
    Font_size = params.get("Font_Size", 12)
    stratify_by = text_to_value(
        params.get("Stratify_By", "None"),
        param_name="Stratify By"
    )  
    
    defined_color_map = text_to_value(
        params.get("Define_Label_Color_Mapping", "None"),
        param_name="Define Label Color Mapping"
    )

    cmin = params.get("Lower_Colorbar_Bound", 999)
    cmax = params.get("Upper_Colorbar_Bound", -999)

    flip_y = params.get("Flip_Vertical_Axis", False)

    feature = text_to_value(feature)
    if color_by == "Annotation":
        feature = None
        if len(annotations) == 0:
            raise ValueError(
                'Please set at least one value in the '
                '"Annotation(s) to Highlight" parameter'
            )
    else:
        annotations = None
        if feature is None:
            raise ValueError('Please set the "Feature to Highlight" parameter.')

    layer = text_to_value(layer, "Original")

    result_list = interactive_spatial_plot(
        adata=adata,
        annotations=annotations,
        feature=feature,
        layer=layer,
        dot_size=dot_size,
        dot_transparency=dot_transparency,
        feature_colorscale=color_map,
        figure_width=desired_width_in,
        figure_height=desired_height_in,
        figure_dpi=dpi,
        font_size=Font_size,
        stratify_by=stratify_by,
        defined_color_map=defined_color_map,
        reverse_y_axis=flip_y,
        cmin=cmin,
        cmax=cmax
    )

    # Handle results based on save_results flag
    if save_results:
        saved_files = {}
        output_prefix = params.get("Output_File", "interactive_plot")
        
        for result in result_list:
            image_name = result['image_name']
            image_object = result['image_object']
            
            # Show the plot (as in NIDAP template)
            image_object.show()
            
            # Convert to HTML
            html_content = pio.to_html(image_object, full_html=True)
            
            # Save HTML file
            html_filename = f"{output_prefix}_{image_name}.html"
            with open(html_filename, 'w') as file:
                file.write(html_content)
            
            saved_files[html_filename] = html_filename
        
        print(
            f"Interactive Spatial Plot completed → "
            f"{list(saved_files.keys())}"
        )
        return saved_files
    else:
        # Just show the plots without saving
        for result in result_list:
            result['image_object'].show()
        
        return None


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python interactive_spatial_plot_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nDisplayed interactive plots")