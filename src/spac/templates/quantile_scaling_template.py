"""
Platform-agnostic Quantile Scaling template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.quantile_scaling_template import run_from_json
>>> run_from_json("examples/quantile_scaling_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, Tuple
import pandas as pd
import plotly.graph_objects as go

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.transformations import normalize_features
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True,
    show_plot: bool = True
) -> Union[Dict[str, str], Tuple[Any, pd.DataFrame]]:
    """
    Execute Quantile Scaling analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the adata object
        and figure directly for in-memory workflows. Default is True.
    show_plot : bool, optional
        Whether to display the plot. Default is True.

    Returns
    -------
    dict or tuple
        If save_results=True: Dictionary of saved file paths
        If save_results=False: Tuple of (adata, figure)
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters using .get() with defaults from JSON template
    low_quantile = params.get("Low_Quantile", "0.02")
    high_quantile = params.get("High_Quantile", "0.98")
    interpolation = params.get("Interpolation", "nearest")
    input_layer = params.get("Table_to_Process", "Original")
    output_layer = params.get("Output_Table_Name", "normalized_feature")
    per_batch = params.get("Per_Batch", "False")
    # Annotation may be None, '', 'None', or a real name
    annotation = params.get("Annotation")

    # Convert parameters using text_to_value
    if input_layer == "Original":
        input_layer = None
    
    low_quantile = text_to_value(
        low_quantile,
        to_float=True,
        param_name='Low_Quantile'
    )
    
    high_quantile = text_to_value(
        high_quantile,
        to_float=True,
        param_name='High_Quantile'
    )
    
    # Convert "True"/"False" string to boolean (case-insensitive)
    per_batch = str(per_batch).strip().lower() == "true"
    
    # Annotation is optional - empty string or "None" becomes None
    annotation = text_to_value(annotation)
    
    # Validate annotation is provided when per_batch is True
    if per_batch and annotation is None:
        raise ValueError(
            'Parameter "Annotation" is required when "Per Batch" is set '
            'to True.'
        )
    
    # Check if output_layer already exists in adata
    print(f"Checking if output layer '{output_layer}' exists in adata "
          f"layers...")
    if output_layer in adata.layers.keys():
        raise ValueError(
            f"Output Table Name '{output_layer}' already exists, "
            f"please rename it."
        )
    else:
        print(f"Output layer '{output_layer}' does not exist. "
              f"Proceeding with normalization.")
    
    def df_as_html(
        df,
        columns_to_plot,
        font_size=12,
        column_scaler=1
    ):
        df = df.reset_index()
        df = df[columns_to_plot]
        df_str = df.astype(str)
        
        column_widths = [
            max(df_str[col].apply(len)) * font_size * column_scaler
            for col in df.columns
        ]
        column_widths[0] = 200
        
        fig_width = sum(column_widths) * 1.1
        # Create a table trace with the DataFrame data
        table_trace = go.Table(
            header=dict(values=list(df.columns),
                        font=dict(size=font_size)),
            cells=dict(values=df_str.values.T,
                       font=dict(size=font_size),
                       align='left'),
            columnwidth=column_widths
        )

        layout = go.Layout(
            autosize=True
            )

        fig = go.Figure(
            data=[table_trace],
            layout=layout
        )

        return fig

    def create_normalization_info(
        adata,
        low_quantile,
        high_quantile,
        input_layer,
        output_layer
    ):
        pre_dataframe = adata.to_df(layer=input_layer)
        quantiles = pre_dataframe.quantile([low_quantile, high_quantile])
        new_row_names = {
            high_quantile: 'quantile_high',
            low_quantile: 'quantile_low'
        }
        quantiles.index = quantiles.index.map(new_row_names)
        
        pre_info = pre_dataframe.describe()    
        pre_info = pd.concat([pre_info, quantiles])    
        pre_info = pre_info.reset_index()
        pre_info['index'] = 'Pre-Norm: ' + pre_info['index'].astype(str)
        del pre_dataframe

        post_dataframe = adata.to_df(layer=output_layer)
        post_info = post_dataframe.describe()
        post_info = post_info.reset_index()
        post_info['index'] = 'Post-Norm: ' + post_info['index'].astype(str)
        del post_dataframe

        normalization_info = pd.concat([pre_info, post_info]).transpose()
        normalization_info.columns = normalization_info.iloc[0]
        normalization_info = normalization_info.drop(
            normalization_info.index[0]
        )
        normalization_info = normalization_info.astype(float)
        normalization_info = normalization_info.round(3)
        normalization_info = normalization_info.astype(str)
    
        return normalization_info

    print(f"High quantile used: {str(high_quantile)}")
    print(f"Low quantile used: {str(low_quantile)}")

    transformed_data = normalize_features(
        adata=adata,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        interpolation=interpolation,
        input_layer=input_layer,
        output_layer=output_layer,
        per_batch=per_batch,
        annotation=annotation
    )

    print(f"Transformed data stored in layer: {output_layer}")
    dataframe = pd.DataFrame(transformed_data.layers[output_layer])
    print(dataframe.describe())

    normalization_info = create_normalization_info(
        adata,
        low_quantile,
        high_quantile,
        input_layer,
        output_layer
    )
    
    columns_to_plot = [
        'index', 'Pre-Norm: mean', 'Pre-Norm: std',
        'Pre-Norm: quantile_high', 'Pre-Norm: quantile_low',
        'Post-Norm: mean', 'Post-Norm: std', 
    ]
    
    html_plot = df_as_html(
        normalization_info,
        columns_to_plot
        )

    if show_plot:
        html_plot.show()

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "transform_output.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
    
        saved_files = save_outputs({output_file: transformed_data})
    
        print(f"Quantile Scaling completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object and figure directly for in-memory
        # workflows
        print("Returning AnnData object and figure (not saving to file)")
        return transformed_data, html_plot


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python quantile_scaling_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned AnnData object and figure")