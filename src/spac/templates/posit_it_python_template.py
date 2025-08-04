"""
Platform-agnostic Post-It-Python template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.posit_it_python_template import run_from_json
>>> run_from_json("examples/posit_it_python_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.templates.template_utils import (
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True,
    show_plot: bool = False
) -> Union[Dict[str, str], Any]:
    """
    Execute Post-It-Python analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the figure
        directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is False.

    Returns
    -------
    dict or figure
        If save_results=True: Dictionary of saved file paths
        If save_results=False: The matplotlib figure object
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Extract parameters using .get() with defaults from JSON template
    text = params.get("Label", "Post-It")
    text_color = params.get("Label_font_color", "Black")
    text_size = params.get("Label_font_size", "80")
    text_fontface = params.get("Label_font_type", "normal")
    text_fontfamily = params.get("Label_font_family", "Arial")
    bold = params.get("Label_Bold", "False")

    # background params
    fill_color = params.get("Background_fill_color", "Yellow1")
    fill_alpha = params.get("Background_fill_opacity", "10")
    border_alpha = 1
    border_width = 0

    # image params
    image_width = params.get("Page_width", "18")
    image_height = params.get("Page_height", "6")
    image_resolution = params.get("Page_DPI", "300")

    # output value
    tag = "CCBR"

    # Convert string parameters to appropriate types
    text_size = text_to_value(
        text_size,
        to_int=True,
        param_name="Label_font_size"
    )
    
    bold = text_to_value(bold) == "True"
    
    fill_alpha = text_to_value(
        fill_alpha,
        to_float=True,
        param_name="Background_fill_opacity"
    )
    
    image_width = text_to_value(
        image_width,
        to_float=True,
        param_name="Page_width"
    )
    
    image_height = text_to_value(
        image_height,
        to_float=True,
        param_name="Page_height"
    )
    
    image_resolution = text_to_value(
        image_resolution,
        to_int=True,
        param_name="Page_DPI"
    )

    # RUN ====   

    # paints    
    paints = {
        'White': '#FFFFFF',
        'LightGrey': '#D3D3D3',
        'Grey': '#999999',
        'Black': '#000000',
        'Red1': '#F44E3B',
        'Red2': '#D33115',
        'Red3': '#9F0500',
        'Orange1': '#FE9200',
        'Orange2': '#E27300',
        'Orange3': '#C45100',
        'Yellow1': '#FCDC00',
        'Yellow2': '#FCC400',
        'Yellow3': '#FB9E00',
        'YellowGreen1': '#DBDF00',
        'YellowGreen2': '#B0BC00',
        'Yellowgreen3': '#808900',
        'Green1': '#A4DD00',
        'Green2': '#68BC00',
        'Green3': '#194D33',
        'Teal1': '#68CCCA',
        'Teal2': '#16A5A5',
        'Teal3': '#0C797D',
        'Blue1': '#73D8FF',
        'Blue2': '#009CE0',
        'Blue3': '#0062B1',
        'Purple1': '#AEA1FF',
        'Purple2': '#7B64FF',
        'Purple3': '#653294',
        'Magenta1': '#FDA1FF',
        'Magenta2': '#FA28FF',
        'Magenta3': '#AB149E'
    }   

    # image: png
    fig = plt.figure(
        figsize=(image_width, image_height), 
        dpi=image_resolution
    )
    fig.patch.set_facecolor(paints[fill_color])
    fig.patch.set_alpha(fill_alpha/100)
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(text_size)
            item.set_fontfamily(text_fontfamily)
            item.set_fontstyle(text_fontface)
            if bold:
                item.set_fontweight('bold')

    # plt.set_facecolor(paints[fill_color]+hex_fill)    

    fig.text(
        0.5, 0.5, text, 
        fontsize=text_size, 
        color=paints[text_color], 
        ha='center', 
        va='center', 
        fontfamily=text_fontfamily,
        fontstyle=text_fontface, 
        fontweight='bold' if bold else 'normal'
    )
    
    if show_plot:
        plt.show()
    
    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "graphicsFile.png")
        # Ensure .png extension
        if not output_file.endswith('.png'):
            output_file = output_file + '.png'
        
        fig.savefig(
            output_file, 
            format='png', 
            transparent=True, 
            bbox_inches='tight'
        )
        plt.close(fig)
        
        saved_files = {output_file: output_file}
        
        print(f"Post-It-Python completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the figure object directly for in-memory workflows
        print("Returning figure object (not saving to file)")
        return fig


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python posit_it_python_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned figure object")