"""
Platform-agnostic Ripley-L template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.ripley_l_template import run_from_json
>>> run_from_json("examples/ripley_l_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.spatial_analysis import ripley_l
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    text_to_value,
)


# def _prepare_ripley_uns_for_h5ad(adata) -> None:
#     """
#     Minimal fix for ripley_l results serialization.
#     Converts object columns to appropriate types for H5AD storage.
#     """
#     if "ripley_l" not in adata.uns:
#         return

#     rl = adata.uns.get("ripley_l")
#     if isinstance(rl, pd.DataFrame):
#         # Create a copy to avoid modifying the original
#         clean_df = rl.copy()

#         # Process each object column
#         for col in clean_df.columns:
#             if clean_df[col].dtype == "object":
#                 # Try to convert to numeric
#                 try:
#                     clean_df[col] = pd.to_numeric(
#                         clean_df[col], errors='raise'
#                     )
#                 except Exception:
#                     # If that fails, convert to string
#                     clean_df[col] = clean_df[col].astype(str)

#         # Replace with cleaned version
#         adata.uns["ripley_l"] = clean_df

def _prepare_ripley_uns_for_h5ad(adata) -> None:
    """
    Enhanced fix for ripley_l results serialization.
    Ensures proper data types and structure for H5AD storage.
    """
    import numpy
    if "ripley_l" not in adata.uns:
        return
    for uns_key in adata.uns.keys():
        rl = adata.uns.get(uns_key)
        print(type(rl))
        if isinstance(rl, numpy.ndarray):
            # Convert numpy arrays to lists
            continue
        output_dict = {}
        for key, value in rl.items():
            print(f"Key: {key}, Type: {type(value)}")
            if key == "ripley_l":
                for sub_key, sub_value in value.items():
                    print(sub_key, type(sub_value))
                    if isinstance(sub_value, Dict):
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            if sub_sub_key == "sims_stat" or sub_sub_key == "L_stat":
                                continue
                            else:
                                # Convert DataFrame to numpy array
                                if isinstance(sub_sub_value, pd.DataFrame):
                                    sub_sub_value = sub_sub_value.to_numpy()
                            output_dict[f"ripley_l-{key}-{sub_key}-{sub_sub_key}"] = sub_sub_value
                        continue
                    
                    output_dict[f"ripley_l-{sub_key}"] = sub_value
            else:
                if isinstance(value, pd.Series):
                    output_dict[f"ripley_l-{key}"] = value.to_numpy()
                else:
                    output_dict[f"ripley_l-{key}"] = value.to_numpy()

        print(f"Output dict keys: {list(output_dict.keys())}")
    del adata.uns["ripley_l"]
    for key, item in output_dict.items():
        print(f"Adding {key} to adata.uns, type: {type(item)}, sample: {item[:2] if isinstance(item, (list, numpy.ndarray)) else item}")
    adata.uns.update(output_dict)
   

def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Execute Ripley-L analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary

    Returns
    -------
    dict
        Dictionary of saved file paths
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    radii = params["Radii"]
    annotation = params["Annotation"]
    phenotypes = [params["Center_Phenotype"], params["Neighbor_Phenotype"]]
    regions = params.get("Stratify_By", "None")
    n_simulations = params.get("Number_of_Simulations", 1)
    area = params.get("Area", "None")
    seed = params.get("Seed", 42)
    spatial_key = params.get("Spatial_Key", "spatial")
    edge_correction = params.get("Edge_Correction", True)

    # Process parameters
    regions = text_to_value(
        regions,
        default_none_text="None"
    )

    area = text_to_value(
        area,
        default_none_text="None",
        value_to_convert_to=None,
        to_float=True,
        param_name='Area'
    )

    # Convert radii to floats
    radii = _convert_to_floats(radii)

    # Run the analysis
    ripley_l(
        adata,
        annotation=annotation,
        phenotypes=phenotypes,
        distances=radii,
        regions=regions,
        n_simulations=n_simulations,
        area=area,
        seed=seed,
        spatial_key=spatial_key,
        edge_correction=edge_correction
    )

    # Fix ripley_l results before saving
    _prepare_ripley_uns_for_h5ad(adata)
    print("Ripley-L analysis completed successfully.")

    # Save outputs
    outfile = params.get("Output_File", "transform_output.h5ad")
    print(type(outfile))
    saved_files = save_outputs({outfile: adata})
    print(saved_files)

    print(f"Ripley-L completed → {str(saved_files[outfile])}")
    print(adata)
    return saved_files


def _convert_to_floats(text_list: List[Any]) -> List[float]:
    """
    Convert list of text values to floats.
    Exact copy from NIDAP template.

    Parameters
    ----------
    text_list : list
        List of values to convert

    Returns
    -------
    list
        List of float values

    Raises
    ------
    ValueError
        If any value cannot be converted to float
    """
    float_list = []
    for value in text_list:
        try:
            float_list.append(float(value))
        except ValueError:
            msg = f"Failed to convert the radius: '{value}' to float."
            raise ValueError(msg)
    return float_list


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ripley_l_template.py <params.json>")
        sys.exit(1)

    saved_files = run_from_json(sys.argv[1])

    print("\nOutput files:")
    for filename, filepath in saved_files.items():
        print(f"  {filename}: {filepath}")
