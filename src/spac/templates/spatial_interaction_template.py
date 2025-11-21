"""
Platform-agnostic Spatial Interaction template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Follows standardized output schema where figures are saved as directories.

Usage
-----
>>> from spac.templates.spatial_interaction_template import run_from_json
>>> run_from_json("examples/spatial_interaction_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.spatial_analysis import spatial_interaction
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
) -> Union[Dict[str, Union[str, List[str]]], Tuple[List[Any], Dict[str, pd.DataFrame]]]:
    """
    Execute Spatial Interaction analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Annotation": "cell_type",
            "Spatial_Analysis_Method": "Neighborhood Enrichment",
            "outputs": {
                "figures": {"type": "directory", "name": "figures_dir"},
                "dataframe": {"type": "file", "name": "dataframe.csv"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If True, saves figures to a directory
        and matrices to CSV files using centralized save_results. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    output_dir : str or Path, optional
        Base directory for outputs. If None, uses params['Output_Directory'] or '.'

    Returns
    -------
    dict or tuple
        If save_to_disk=True: Dictionary mapping output types to saved file paths
        If save_to_disk=False: Tuple of (figures_list, matrices_dict) for in-memory use
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    annotation = params["Annotation"]
    analysis_method = params["Spatial_Analysis_Method"]
    # Two analysis methods available:
    # 1. "Neighborhood Enrichment": Calculates how often pairs of cell types
    #    are neighbors compared to random chance. Positive scores indicate
    #    attraction/co-location, negative scores indicate avoidance.
    #    Output: z-scores (can be positive or negative)
    #    Files: neighborhood_enrichment_{identifier}.csv
    # 2. "Cluster Interaction Matrix": Counts the number of edges/connections
    #    between different cell types in the spatial neighborhood graph.
    #    Shows absolute interaction frequencies rather than enrichment.
    #    Output: raw counts (always positive integers)
    #    Files: cluster_interaction_matrix_{identifier}.csv
    # Both methods produce the same data structure, just different values
    stratify_by = params.get("Stratify_By", ["None"])
    seed = params.get("Seed", "None")
    coord_type = params.get("Coordinate_Type", "None")
    n_rings = 1
    n_neighs = params.get("K_Nearest_Neighbors", 6)
    radius = params.get("Radius", "None")
    image_width = params.get("Figure_Width", 15)
    image_height = params.get("Figure_Height", 12)
    dpi = params.get("Figure_DPI", 200)
    font_size = params.get("Font_Size", 12)
    color_bar_range = params.get("Color_Bar_Range", "Automatic")

    def save_matrix(matrix):
        for file_name in matrix:
            data_df = matrix[file_name]
            print("\n")
            print(file_name)
            print(data_df)
            # In SPAC, collect matrices for later saving instead of 
            # direct file write. Store them with proper extension if missing.
            if not file_name.endswith('.csv'):
                file_name = f"{file_name}.csv"
            matrices[file_name] = data_df

    def update_nidap_display(
        axs,
        image_width,
        image_height,
        dpi,
        font_size
    ):
        # NIDAP display logic is different than the generic python
        # image output. For example, a 12in*8in image with font 12
        # should properly display all text in generic Image
        # But in nidap code workbook resizing, the text will be reduced. 
        # This function is to adjust the image sizing and font sizing
        # to fit the NIDAP display
        # Get the figure associated with the axes
        fig = axs.get_figure()
        
        # Set figure size and DPI
        fig.set_size_inches(image_width, image_height)
        fig.set_dpi(dpi)
        
        # Customize font sizes
        axs.title.set_fontsize(font_size)  # Title font size
        axs.xaxis.label.set_fontsize(font_size)  # X-axis label font size
        axs.yaxis.label.set_fontsize(font_size)  # Y-axis label font size
        axs.tick_params(axis='both', labelsize=font_size)  # Tick labels
        # Return the updated figure and axes for chaining or further use
        # Note: This adjustment was specific to NIDAP display resizing
        # behavior and may not be necessary in other environments
        return fig, axs

    for i, item in enumerate(stratify_by):
        item_is_none = text_to_value(item)
        if item_is_none is None and i == 0:
            stratify_by = item_is_none
        elif item_is_none is None and i != 0:
            raise ValueError(
                'Found string "None" in the stratify by list that is '
                'not the first entry.\n'
                'Please remove the "None" to proceed with the list of '
                'stratify by options, \n'
                'or move the "None" to start of the list to disable '
                'stratification. Thank you.')

    seed = text_to_value(seed, to_int=True)
    radius = text_to_value(radius, to_float=True)
    coord_type = text_to_value(coord_type)
    color_bar_range = text_to_value(
        color_bar_range,
        "Automatic",
        to_float=True)
    
    if color_bar_range is not None:
        cmap = "seismic"
        vmin = -abs(color_bar_range)
        vmax = abs(color_bar_range)
    else:
        cmap = "seismic"
        vmin = vmax = color_bar_range

    plt.rcParams['font.size'] = font_size

    result_dictionary = spatial_interaction(
        adata=adata,
        annotation=annotation,
        analysis_method=analysis_method,
        stratify_by=stratify_by,
        return_matrix=True,
        seed=seed,
        coord_type=coord_type,
        n_rings=n_rings,
        n_neighs=n_neighs,
        radius=radius,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        figsize=(image_width, image_height),
        dpi=dpi
    )

    # Track figures and matrices for optional saving
    figures = []
    matrices = {}

    if not stratify_by:
        axs = result_dictionary['Ax']
        fig, axs = update_nidap_display(
            axs=axs,
            image_width=image_width,
            image_height=image_height,
            dpi=dpi,
            font_size=font_size
        )
        figures.append(fig)
        if show_plot:
            plt.show()

        matrix = result_dictionary['Matrix']['annotation']
        save_matrix(matrix)
    else:
        plt.close(1)
        axs_dict = result_dictionary['Ax']
        for key in axs_dict:            
            axs = axs_dict[key]
            fig, axs = update_nidap_display(
                axs=axs,
                image_width=image_width,
                image_height=image_height,
                dpi=dpi,
                font_size=font_size
            )
            figures.append(fig)
            if show_plot:
                plt.show()

        matrix_dict = result_dictionary['Matrix']
        for identifier in matrix_dict:
            matrix = matrix_dict[identifier]
            save_matrix(matrix)      

    # Handle saving if requested (separate from NIDAP logic)
    if save_to_disk:
        # Ensure outputs configuration exists
        if "outputs" not in params:
            # Provide default outputs config if not present
            params["outputs"] = {
                "figures": {"type": "directory", "name": "figures"},
                "dataframes": {"type": "directory", "name": "matrices"}
            }
        
        # Prepare results dictionary
        results_dict = {}
        
        # Package figures in a dictionary for directory saving
        if figures:
            # Store figures with meaningful names  
            figures_dict = {}
            for i, fig in enumerate(figures):
                # Extract title if available for better naming
                try:
                    ax = fig.axes[0] if fig.axes else None
                    title = ax.get_title() if ax and ax.get_title() else f"interaction_plot_{i+1}"
                    # Clean title for filename
                    title = title.replace(" ", "_").replace("/", "_").replace(":", "")
                    figures_dict[f"{title}.png"] = fig
                except:
                    figures_dict[f"interaction_plot_{i+1}.png"] = fig
            
            results_dict["figures"] = figures_dict
        
        # Add matrices (already have .csv extension added)
        if matrices:
            results_dict["dataframes"] = matrices
        
        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        # Close figures after saving to free memory
        for fig in figures:
            plt.close(fig)
            
        print(
            f"Spatial Interaction completed -> "
            f"{list(saved_files.keys())}"
        )
        return saved_files
    else:
        # Return objects directly for in-memory workflows
        return figures, matrices


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python spatial_interaction_template.py <params.json> [output_dir]",
            file=sys.stderr
        )
        sys.exit(1)

    # Get output directory if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run analysis
    result = run_from_json(
        json_path=sys.argv[1],
        output_dir=output_dir
    )

    # Display results based on return type
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
        figures_list, matrices_dict = result
        print("\nReturned figures and matrices for in-memory use")
        print(f"Number of figures: {len(figures_list)}")
        print(f"Number of matrices: {len(matrices_dict)}")
