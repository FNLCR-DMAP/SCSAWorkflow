"""
Platform-agnostic Spatial Plot template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.spatial_plot_template import run_from_json
>>> run_from_json("examples/spatial_plot_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional
import matplotlib.pyplot as plt
from functools import partial

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import spatial_plot
from spac.data_utils import select_values
from spac.utils import check_annotation
from spac.templates.template_utils import (
    load_input,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True,
    show_plots: bool = True
) -> Union[Dict[str, str], None]:
    """
    Execute Spatial Plot analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns None.
        Default is True.
    show_plots : bool, optional
        Whether to display the plots. Default is True.

    Returns
    -------
    dict or None
        If save_results=True: Dictionary of saved file paths
        If save_results=False: None
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters exactly as in NIDAP template
    annotation = params["Annotation_to_Highlight"]
    feature = params["Feature_to_Highlight"]
    layer = params["Table"]
    
    alpha = params["Dot_Transparency"]
    spot_size = params["Dot_Size"]
    image_height = params["Figure_Height"]
    image_width = params["Figure_Width"]
    dpi = params["Figure_DPI"]
    font_size = params["Font_Size"]
    vmin = params["Lower_Colorbar_Bound"]
    vmax = params["Upper_Colorbar_Bound"]
    color_by = params["Color_By"]
    stratify = params["Stratify"]
    stratify_by = params["Stratify_By"]

    ##--------------- ##
    ## Error Messages ##
    ## -------------- ##
    if stratify and len(stratify_by) == 0:
        raise ValueError(
            'Please set at least one annotation in the "Stratify By" '
            'option, or set the "Stratify" to False.'
        )
    check_annotation(
        adata,
        annotations=stratify_by
    )

    # Process feature and annotation with text_to_value
    feature = text_to_value(feature)
    annotation = text_to_value(annotation)    

    if color_by == "Annotation":
        feature = None
    else:
        annotation = None

    ## --------- ##
    ## Functions ##
    ## --------- ##

    ## --------------- ##
    ## Main Code Block ##
    ## --------------- ##
    layer = text_to_value(layer, "Original")

    prefilled_spatial = partial(
        spatial_plot,
        spot_size=spot_size,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        annotation=annotation,
        feature=feature,
        layer=layer
    )

    # Track figures for optional saving
    figures = []

    if not stratify:
        plt.rcParams['font.size'] = font_size
        fig, ax = plt.subplots(
            figsize=(image_width, image_height), dpi=dpi
        )

        ax = prefilled_spatial(adata=adata, ax=ax)

        if color_by == "Annotation":
            title = f'Annotation: {annotation}'
        else:
            title = f'Table:"{layer}" \n Feature:"{feature}"'
        ax[0].set_title(title)

        figures.append(fig)
        
        if show_plots:
            plt.show()
    else:
        combined_label = "concatenated_label"

        adata.obs[combined_label] = adata.obs[stratify_by].astype(str).agg(
            '_'.join, axis=1
        )

        unique_values = adata.obs[combined_label].unique()

        print(unique_values)

        max_length = min(len(unique_values), 20)
        if len(unique_values) > 20:
            print(
                f'WARNING: There are "{len(unique_values)}" unique plots, '
                'displaying only the first 20 plots.'
            )

        for value in unique_values[:max_length]:
            filtered_adata = select_values(
                data=adata, annotation=combined_label, values=value
            )

            fig, ax = plt.subplots(
                figsize=(image_width, image_height), dpi=dpi
            )

            ax = prefilled_spatial(adata=filtered_adata, ax=ax)

            if color_by == "Annotation":
                title = f'Annotation: {annotation}'
            else:
                title = f'Table:"{layer}" \n Feature:"{feature}"'
            title = f'{title}\n Stratify by: {value}'
            ax[0].set_title(title)

            figures.append(fig)
            
            if show_plots:
                plt.show()

    # Handle saving if requested (separate from NIDAP logic)
    if save_results and figures:
        saved_files = {}
        output_prefix = params.get("Output_File", "spatial_plot")
        
        if len(figures) == 1:
            output_file = f"{output_prefix}.png"
            figures[0].savefig(output_file, dpi=dpi, bbox_inches='tight')
            saved_files[output_file] = output_file
        else:
            for i, fig in enumerate(figures):
                output_file = f"{output_prefix}_plot_{i+1}.png"
                fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
                saved_files[output_file] = output_file
        
        # Close figures after saving
        for fig in figures:
            plt.close(fig)
            
        print(f"Spatial Plot completed → {list(saved_files.keys())}")
        return saved_files

    return None


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python spatial_plot_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])
    
    if result:
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")