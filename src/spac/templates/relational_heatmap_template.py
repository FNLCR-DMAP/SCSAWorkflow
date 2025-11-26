"""
Relational Heatmap template with Plotly figure export.
Generates both static PNG snapshots and interactive HTML outputs.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Tuple
import plotly.io as pio

sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import relational_heatmap
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None
) -> Union[Dict, Tuple]:
    """
    Execute Relational Heatmap analysis.
    
    Generates a relational heatmap showing relationships between
    two annotations, with outputs in three formats:
    - PNG snapshot of the Plotly figure
    - Interactive HTML version
    - CSV data matrix
    
    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_to_disk : bool, optional
        Whether to save results to disk. Default is True.
    output_dir : str, optional
        Override output directory from params
        
    Returns
    -------
    dict or tuple
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: Tuple of (plotly_fig, dataframe)
    """
    
    params = parse_params(json_path)
    
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    if "outputs" not in params:
        params["outputs"] = {
            "figures": {"type": "directory", "name": "figures_dir"},
            "html": {"type": "directory", "name": "html_dir"},
            "dataframe": {"type": "file", "name": "dataframe.csv"}
        }

    # Load data
    adata = load_input(params["Upstream_Analysis"])
    print(f"Data loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # Parameters
    source_annotation = text_to_value(
        params.get("Source_Annotation_Name", "None")
    )
    target_annotation = text_to_value(
        params.get("Target_Annotation_Name", "None")
    )
    
    dpi = float(params.get("Figure_DPI", 300))
    width_in = float(params.get("Figure_Width_inch", 8))
    height_in = float(params.get("Figure_Height_inch", 10))
    font_size = float(params.get("Font_Size", 8))
    colormap = params.get("Colormap", "darkmint")

    print(f"Creating heatmap: {source_annotation} vs {target_annotation}")

    # Run SPAC relational heatmap
    result_dict = relational_heatmap(
        adata=adata,
        source_annotation=source_annotation,
        target_annotation=target_annotation,
        color_map=colormap,
        font_size=font_size
    )
    
    rhmap_data = result_dict['data']
    plotly_fig = result_dict['figure']
    
    # Calculate scale factor for high-DPI export
    # Plotly's default is 96 DPI, so scale relative to that
    scale_factor = dpi / 96.0
    
    # Update Plotly figure dimensions and styling for HTML display
    if plotly_fig:
        plotly_fig.update_layout(
            width=width_in * 96,
            height=height_in * 96,
            font=dict(size=font_size)
        )
    
    if save_to_disk:
        # Generate PNG snapshot directly from Plotly figure
        print("Generating PNG snapshot from Plotly figure...")
        img_bytes = pio.to_image(
            plotly_fig,
            format='png',
            width=int(width_in * 96),  # Use base dimensions
            height=int(height_in * 96),
            scale=scale_factor  # Scale up for higher DPI
        )
        
        # Prepare outputs
        results_dict = {
            "figures": {"relational_heatmap": img_bytes},
            "html": {
                "relational_heatmap": pio.to_html(
                    plotly_fig,
                    full_html=True,
                    include_plotlyjs='cdn'
                )
            },
            "dataframe": rhmap_data
        }
        
        saved_files = save_results(
            results_dict,
            params,
            output_base_dir=output_dir
        )
        
        print("✓ Relational Heatmap completed")
        return saved_files
    else:
        return plotly_fig, rhmap_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python relational_heatmap_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)
    
    try:
        run_from_json(sys.argv[1], save_to_disk=True)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
