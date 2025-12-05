"""
Platform-agnostic Spatial Plot template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Refactored to use centralized save_results from template_utils.
Reads outputs configuration from blueprint JSON file.

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
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import spatial_plot
from spac.data_utils import select_values
from spac.utils import check_annotation
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    show_plots: bool = True,
    output_dir: str = None,
) -> Union[Dict[str, Union[str, List[str]]], List[plt.Figure]]:
    """
    Execute Spatial Plot analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary.
        Expected JSON structure:
        {
            "Upstream_Analysis": "path/to/data.pickle",
            "Stratify": true,
            "Stratify_By": ["slide_id"],
            "Color_By": "Annotation",
            ...
            "outputs": {
                "figures": {"type": "directory", "name": "figures_dir"}
            }
        }
    save_to_disk : bool, optional
        Whether to save results to disk. If False, returns the figures
        directly for in-memory workflows. Default is True.
    show_plots : bool, optional
        Whether to display the plots. Default is True.
    output_dir : str, optional
        Base directory for outputs. If None, uses params['Output_Directory']
        or current directory.

    Returns
    -------
    dict or list
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: List of matplotlib figures
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

    # Extract parameters exactly as in NIDAP template
    annotation = params.get("Annotation_to_Highlight", "None")
    feature = params.get("Feature_to_Highlight", "")
    layer = params.get("Table", "Original")
    
    alpha = params.get("Dot_Transparency", 0.5)
    spot_size = params.get("Dot_Size", 25)
    image_height = params.get("Figure_Height", 6)
    image_width = params.get("Figure_Width", 12)
    dpi = params.get("Figure_DPI", 200)
    font_size = params.get("Font_Size", 12)
    vmin = params.get("Lower_Colorbar_Bound", 999)
    vmax = params.get("Upper_Colorbar_Bound", -999)
    color_by = params.get("Color_By", "Annotation")
    stratify = params.get("Stratify", True)
    stratify_by = params.get("Stratify_By", [])

    if stratify and len(stratify_by) == 0:
        raise ValueError(
            'Please set at least one annotation in the "Stratify By" '
            'option, or set the "Stratify" to False.'
        )
    
    if stratify:
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

    # Track figures for saving
    figures_dict = {}

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

        figures_dict["spatial_plot"] = fig
        
        if show_plots:
            plt.show()
    else:
        combined_label = "concatenated_label"

        adata.obs[combined_label] = adata.obs[stratify_by].astype(str).agg(
            '_'.join, axis=1
        )

        unique_values = adata.obs[combined_label].unique()

        logger.info(f"Unique stratification values: {unique_values}")

        max_length = min(len(unique_values), 20)
        if len(unique_values) > 20:
            logger.warning(
                f'There are "{len(unique_values)}" unique plots, '
                'displaying only the first 20 plots.'
            )

        for idx, value in enumerate(unique_values[:max_length]):
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

            # Use sanitized value for figure name
            safe_value = str(value).replace('/', '_').replace('\\', '_')
            figures_dict[f"spatial_plot_{safe_value}"] = fig
            
            if show_plots:
                plt.show()

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Prepare results dictionary based on outputs config
        results_dict = {}

        # Check for figures output
        if "figures" in params["outputs"]:
            results_dict["figures"] = figures_dict

        # Use centralized save_results function
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        # Close figures after saving
        for fig in figures_dict.values():
            plt.close(fig)

        logger.info("Spatial Plot analysis completed successfully.")
        return saved_files
    else:
        # Return the figures directly for in-memory workflows
        logger.info("Returning figures for in-memory use")
        return list(figures_dict.values())


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python spatial_plot_template.py <params.json> [output_dir]",
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
        print(f"\nReturned {len(result)} figures")
