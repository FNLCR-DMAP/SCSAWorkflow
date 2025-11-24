"""
Platform-agnostic Interactive Spatial Plot template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Follows standardized output schema where HTML files are saved as a directory.

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
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, Union[str, List[str]]], None]:
    """
    Execute Interactive Spatial Plot analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Color_By": "Annotation",
            "Annotation_s_to_Highlight": ["renamed_phenotypes"],
            "outputs": {
                "html": {"type": "directory", "name": "html_dir"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns None as plots are 
        shown interactively. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory. All outputs will be saved relative to this directory.

    Returns
    -------
    dict or None
        If save_to_disk=True: Dictionary of saved file paths with structure:
            {"html": ["path/to/html_dir/plot1.html", ...]}
        If save_to_disk=False: None (plots are shown interactively)

    Notes
    -----
    Output Structure:
    - HTML files are saved in a directory (standardized for HTML outputs)
    - When save_to_disk=False, plots are shown interactively
    
    Examples
    --------
    >>> # Save results to disk
    >>> saved_files = run_from_json("params.json")
    >>> print(saved_files["html"])  # List of HTML file paths
    >>> # ['./html_dir/plot_1.html', './html_dir/plot_2.html']
    
    >>> # Display plots interactively without saving
    >>> run_from_json("params.json", save_to_disk=False)
    
    >>> # Custom output directory
    >>> saved = run_from_json("params.json", output_dir="/custom/path")
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    # HTML uses directory type per standardized schema
    if "outputs" not in params:
        params["outputs"] = {
            "html": {"type": "directory", "name": "html_dir"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
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

    # Process parameters
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

    # Execute the interactive spatial plot
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

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare HTML outputs as a dictionary for directory saving
        html_dict = {}
        
        for result in result_list:
            image_name = result['image_name']
            image_object = result['image_object']
            
            # Show the plot (as in NIDAP template)
            image_object.show()
            
            # Convert to HTML
            html_content = pio.to_html(image_object, full_html=True)
            
            # Add to dictionary with appropriate name
            html_dict[image_name] = html_content
        
        # Prepare results dictionary based on outputs config
        results_dict = {}
        if "html" in params["outputs"]:
            results_dict["html"] = html_dict
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        print(
            f"Interactive Spatial Plot completed → "
            f"{saved_files.get('html', [])}"
        )
        return saved_files
    else:
        # Just show the plots without saving
        for result in result_list:
            result['image_object'].show()
        
        print("Displayed interactive plots without saving")
        return None


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python interactive_spatial_plot_template.py <params.json> [output_dir]",
            file=sys.stderr
        )
        sys.exit(1)

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
        print("\nDisplayed interactive plots")
