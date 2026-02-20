"""
Production version of Sankey Plot template for Galaxy.
save files only, no show() calls, no blocking operations.
"""
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Tuple
import pandas as pd
import matplotlib
# Set non-interactive backend for Galaxy
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.io as pio

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import sankey_plot
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,  # Always True for Galaxy
    output_dir: str = None,
    show_static_image: bool = False,
) -> Union[Dict[str, Union[str, List[str]]], None]:
    """
    Execute Sankey Plot analysis for Galaxy.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to parameters JSON file or dict of parameters.
    save_to_disk : bool, default True
        Whether to save results to disk. Always True for Galaxy.
    output_dir : str, optional
        Output directory. If None, read from params.
    show_static_image : bool, default False
        When True, generate a static PNG placeholder figure.
        When False (default), only produce interactive HTML output.
        Disabled by default because Plotly HTML-to-PNG conversion
        hangs inside the Galaxy container environment.
    """
    # Parse parameters from JSON
    params = parse_params(json_path)
    print(f"Loaded parameters for {params.get('Source_Annotation_Name')} -> {params.get('Target_Annotation_Name')}")

    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    if "outputs" not in params:
        params["outputs"] = {
            "figures": {"type": "directory", "name": "figures_dir"},
            "html": {"type": "directory", "name": "html_dir"}
        }

    # Load the upstream analysis data
    print("Loading upstream analysis data...")
    adata = load_input(params["Upstream_Analysis"])
    print(f"Data loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Extract parameters
    annotation_columns = [
        params.get("Source_Annotation_Name", "None"),
        params.get("Target_Annotation_Name", "None")
    ]
    
    # Parse numeric parameters with error handling
    try:
        dpi = float(params.get("Figure_DPI", 300))
    except (ValueError, TypeError):
        dpi = 300
        print(f"Warning: Invalid DPI value, using default {dpi}")
    
    width_num = float(params.get("Figure_Width_inch", 6))
    height_num = float(params.get("Figure_Height_inch", 6))
    
    source_color_map = params.get("Source_Annotation_Color_Map", "tab20")
    target_color_map = params.get("Target_Annotation_Color_Map", "tab20b")
    
    try:
        sankey_font = float(params.get("Font_Size", 12))
    except (ValueError, TypeError):
        sankey_font = 12
        print(f"Warning: Invalid font size, using default {sankey_font}")

    target_annotation = text_to_value(annotation_columns[1])
    source_annotation = text_to_value(annotation_columns[0])
    
    print(f"Creating Sankey plot: {source_annotation} -> {target_annotation}")

    # Execute the sankey plot
    fig = sankey_plot(
        adata=adata,
        source_annotation=source_annotation,
        target_annotation=target_annotation,
        source_color_map=source_color_map,
        target_color_map=target_color_map,
        sankey_font=sankey_font
    )

    # Customize the Sankey diagram layout
    width_in_pixels = width_num * dpi
    height_in_pixels = height_num * dpi
    
    fig.update_layout(
        width=width_in_pixels,
        height=height_in_pixels
    )
    
    print("Sankey plot generated")

    # IMPORTANT: No show() calls — causes hang in Galaxy
    # plt.show() - REMOVED
    # fig.show() - REMOVED

    # Handle saving — always save to disk for Galaxy
    if save_to_disk:
        # Prepare results dictionary
        results_dict = {}

        # Save Plotly HTML (the actual interactive Sankey diagram)
        if "html" in params["outputs"]:
            html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
            results_dict["html"] = {"sankey_plot": html_content}
            print("Plotly HTML prepared for saving")

        if show_static_image:
            # Generate a static matplotlib placeholder figure.
            # Disabled by default on Galaxy because Plotly HTML-to-PNG
            # conversion hangs in the Galaxy container environment.
            # The interactive HTML is the first-class output.
            print("Creating matplotlib figure...")
            static_fig, ax = plt.subplots(
                figsize=(width_num, height_num), dpi=dpi
            )
            ax.text(
                0.5, 0.6, 'Sankey Diagram',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, fontweight='bold'
            )
            ax.text(
                0.5, 0.5,
                f'{source_annotation} → {target_annotation}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12
            )
            ax.text(
                0.5, 0.3,
                'View HTML output for interactive diagram',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, style='italic'
            )
            ax.axis('off')
            ax.add_patch(plt.Rectangle(
                (0.1, 0.2), 0.8, 0.5,
                fill=False, edgecolor='gray', linewidth=1,
                transform=ax.transAxes
            ))

            if "figures" in params["outputs"]:
                results_dict["figures"] = {"sankey_plot": static_fig}
                print("Matplotlib figure prepared for saving")

        # Use centralized save_results function
        print("Saving all results...")
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )

        if show_static_image:
            plt.close(static_fig)

        print(f"✓ Sankey Plot completed successfully")
        print(f"  Outputs saved: {list(saved_files.keys())}")

        return saved_files
    else:
        # For non-Galaxy use (testing)
        print("Returning None (display mode not supported)")
        return None


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python sankey_plot_template.py <params.json> [output_dir]",
            file=sys.stderr
        )
        sys.exit(1)

    # Get output directory if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("\n" + "="*60)
    print("SANKEY PLOT - GALAXY PRODUCTION VERSION")
    print("="*60 + "\n")
    
    try:
        result = run_from_json(
            json_path=sys.argv[1],
            output_dir=output_dir,
            save_to_disk=True  # Always save for Galaxy
        )
        
        if isinstance(result, dict):
            print("\nOutput files generated:")
            for key, paths in result.items():
                if isinstance(paths, list):
                    print(f"  {key}:")
                    for path in paths:
                        print(f"    - {path}")
                else:
                    print(f"  {key}: {paths}")
        
        print("\n✓ SUCCESS - Job completed without hanging")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
