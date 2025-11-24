"""
Relational Heatmap with Plotly-matplotlib color synchronization.
Extracts actual colors from Plotly and uses them in matplotlib.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.io as pio
import plotly.express as px

sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.visualization import relational_heatmap
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def get_plotly_colorscale_as_matplotlib(plotly_colormap: str) -> mcolors.LinearSegmentedColormap:
    """
    Extract actual colors from Plotly colorscale and create matplotlib colormap.
    This ensures exact color matching between Plotly and matplotlib.
    """
    # Get Plotly's colorscale
    try:
        # Use plotly express to get the actual color sequence
        colorscale = getattr(px.colors.sequential, plotly_colormap, None)
        if colorscale is None:
            colorscale = getattr(px.colors.diverging, plotly_colormap, None)
        if colorscale is None:
            colorscale = getattr(px.colors.cyclical, plotly_colormap, None)
        
        if colorscale is None:
            # Fallback to a default
            print(f"Warning: Could not find Plotly colorscale '{plotly_colormap}', using default")
            colorscale = px.colors.sequential.Viridis
        
        # Convert to matplotlib colormap
        if isinstance(colorscale, list):
            # Create custom colormap from color list
            cmap = mcolors.LinearSegmentedColormap.from_list(
                f"plotly_{plotly_colormap}", 
                colorscale
            )
            return cmap
    except Exception as e:
        print(f"Error extracting Plotly colors: {e}")
    
    # Fallback to matplotlib's viridis
    return plt.cm.viridis


def create_matplotlib_heatmap_matching_plotly(
    data: pd.DataFrame,
    plotly_fig: Any,
    source_annotation: str,
    target_annotation: str,
    colormap_name: str,
    figsize: tuple,
    dpi: int,
    font_size: int
) -> plt.Figure:
    """
    Create matplotlib heatmap that matches Plotly's appearance.
    Extracts color information from the Plotly figure.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Get the actual colormap from Plotly
    cmap = get_plotly_colorscale_as_matplotlib(colormap_name)
    
    # Extract data range from Plotly figure if possible
    try:
        zmin = plotly_fig.data[0].zmin if hasattr(plotly_fig.data[0], 'zmin') else data.min().min()
        zmax = plotly_fig.data[0].zmax if hasattr(plotly_fig.data[0], 'zmax') else data.max().max()
    except:
        zmin, zmax = data.min().min(), data.max().max()
    
    # Create heatmap matching Plotly's style
    im = ax.imshow(
        data.values, 
        aspect='auto', 
        cmap=cmap,
        interpolation='nearest',
        vmin=zmin,
        vmax=zmax
    )
    
    # Match Plotly's tick placement
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=font_size)
    ax.set_yticklabels(data.index, fontsize=font_size)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    
    # Title matching Plotly
    ax.set_title(
        f'Relational Heatmap: {source_annotation} vs {target_annotation}',
        fontsize=font_size + 2,
        pad=20
    )
    ax.set_xlabel(target_annotation, fontsize=font_size)
    ax.set_ylabel(source_annotation, fontsize=font_size)
    
    # Add grid for clarity (like Plotly)
    ax.set_xticks(np.arange(len(data.columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(data.index) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.3)
    ax.tick_params(which='both', length=0)
    
    plt.tight_layout()
    return fig


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: str = None
) -> Union[Dict, Tuple]:
    """Execute Relational Heatmap with color-matched outputs."""
    
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
    source_annotation = text_to_value(params.get("Source_Annotation_Name", "None"))
    target_annotation = text_to_value(params.get("Target_Annotation_Name", "None"))
    
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
    
    # Update Plotly figure
    if plotly_fig:
        plotly_fig.update_layout(
            width=width_in * 96,
            height=height_in * 96,
            font=dict(size=font_size)
        )
    
    # Create matplotlib figure that matches Plotly's colors
    print("Creating color-matched matplotlib figure...")
    static_fig = create_matplotlib_heatmap_matching_plotly(
        rhmap_data,
        plotly_fig,
        source_annotation,
        target_annotation,
        colormap,
        (width_in, height_in),
        int(dpi),
        int(font_size)
    )
    
    if save_to_disk:
        results_dict = {
            "figures": {"relational_heatmap": static_fig},
            "html": {"relational_heatmap": pio.to_html(plotly_fig, full_html=True, include_plotlyjs='cdn')},
            "dataframe": rhmap_data
        }
        
        saved_files = save_results(results_dict, params, output_base_dir=output_dir)
        plt.close(static_fig)
        
        print("✓ Relational Heatmap completed")
        return saved_files
    else:
        return plotly_fig, rhmap_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python relational_heatmap_template.py <params.json>", file=sys.stderr)
        sys.exit(1)
    
    try:
        run_from_json(sys.argv[1], save_to_disk=True)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
