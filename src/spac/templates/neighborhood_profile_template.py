"""
Platform-agnostic Neighborhood Profile template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.neighborhood_profile_template import run_from_json
>>> run_from_json("examples/neighborhood_profile_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union, List, Optional, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.spatial_analysis import neighborhood_profile
from spac.templates.template_utils import (
    load_input,
    save_results,
    parse_params,
    text_to_value,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_to_disk: bool = True,
    output_dir: Union[str, Path] = None
) -> Union[Dict[str, str], Dict[Tuple[str, str], pd.DataFrame]]:
    """
    Execute Neighborhood Profile analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_to_disk : bool, optional
        Whether to save results to file. If False, returns the dataframes
        directly for in-memory workflows. Default is True.
    output_dir : str or Path, optional
        Output directory for results. If None, uses params['Output_Directory'] or '.'

    Returns
    -------
    dict
        If save_to_disk=True: Dictionary of saved file paths
        If save_to_disk=False: Dictionary of (anchor, neighbor) tuples 
        to DataFrames
    """
    # Parse parameters from JSON
    params = parse_params(json_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = params.get("Output_Directory", ".")
    
    # Ensure outputs configuration exists with standardized defaults
    # Neighborhood Profile dataframes use directory type per special case in template_utils
    if "outputs" not in params:
        params["outputs"] = {
            "dataframe": {"type": "directory", "name": "dataframe_dir"}
        }

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    cell_types_annotation = params["Annotation_of_interest"]
    bins = params["Bins"]
    slide_names = params.get("Stratify_By", "None")
    normalization = None
    output_table = "neighborhood_profile"

    anchor_neighbor_list = params["Anchor_Neighbor_List"]
    anchor_neighbor_list = [
        tuple(map(str.strip, item.split(";"))) 
        for item in anchor_neighbor_list
    ]

    # Call the spatial umap calculation
    bins = [float(radius) for radius in bins]
    slide_names = text_to_value(slide_names)

    neighborhood_profile(
        adata,
        phenotypes=cell_types_annotation,
        distances=bins,
        regions=slide_names,
        spatial_key="spatial",
        normalize=normalization,
        associated_table_name=output_table
    )

    print(adata)
    print(adata.obsm[output_table].shape)
    print(adata.uns[output_table])

    dataframes, filenames = neighborhood_profiles_for_pairs(
        adata,
        cell_types_annotation,
        slide_names,
        bins,
        anchor_neighbor_list,
        output_table
    )

    # Handle results based on save_to_disk flag
    if save_to_disk:
        # Package dataframes in a dictionary for directory saving
        # This ensures they're saved in a directory per standardized schema
        results_dict = {}
        
        # Create a dictionary of dataframes with their filenames as keys
        dataframe_dict = {}
        for (anchor_label, neighbor_label), filename in zip(
            dataframes.keys(), filenames
        ):
            df = dataframes[(anchor_label, neighbor_label)]
            # Remove .csv extension as save_results will add it
            key = filename.replace('.csv', '')
            dataframe_dict[key] = df
        
        # Store in results with "dataframe" key to match outputs config
        if "dataframe" in params["outputs"]:
            results_dict["dataframe"] = dataframe_dict
        
        # Use centralized save_results function
        # All file handling and logging is now done by save_results
        saved_files = save_results(
            results=results_dict,
            params=params,
            output_base_dir=output_dir
        )
        
        print(f"Neighborhood Profile completed → {len(saved_files.get('dataframe', []))} files")
        return saved_files
    else:
        # Return the dataframes directly for in-memory workflows
        print("Returning dataframes (not saving to file)")
        return dataframes


# Global imports and functions included below

def neighborhood_profiles_for_pairs(
    adata,
    cell_types_annotation,
    slide_names,
    bins,
    anchor_neighbor_list,
    output_table
):
    """
    Compute neighborhood profiles for all anchor-neighbor pairs and return
    a tuple containing a dictionary of DataFrames and a list of filenames
    for saving.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial and phenotypic data.

    cell_types_annotation : str
        The column name in adata.obs containing the cell phenotype labels.

    slide_names : str
        The column name in adata.obs containing the slide names.

    bins : list
        List of increasing distance bins.

    anchor_neighbor_list : list of tuples
        List of (anchor_label, neighbor_label) pairs.

    output_table : str
        The key in adata.obsm containing neighborhood profile data.

    Returns
    -------
    tuple
        - A dictionary of DataFrames for each (anchor, neighbor) pair.
        - A list of filenames where each DataFrame should be saved.
    """

    dataframes = {}
    filenames = []

    # Get the array of neighbor labels
    neighbor_labels = adata.uns[output_table]["labels"]

    for anchor_label, neighbor_label in anchor_neighbor_list:
        # Create bin labels with the neighbor type
        bins_with_ranges = [
            f"{neighbor_label}_{bins[i]}-{bins[i+1]}" 
            for i in range(len(bins) - 1)
        ]

        # Find the index of the requested neighbor label
        neighbor_index = np.where(neighbor_labels == neighbor_label)[0]

        if len(neighbor_index) == 0:
            raise ValueError(
                f"Neighbor label '{neighbor_label}' not found in "
                f"{output_table} labels."
            )

        neighbor_index = neighbor_index[0]  # Extract the first index

        # Extract the neighborhood profile for the specific neighbor
        # Shape: (n_cells, n_bins)
        profile_data = adata.obsm[output_table][:, neighbor_index, :]

        # Construct DataFrame
        df = pd.DataFrame(profile_data, columns=bins_with_ranges)

        # Add cell phenotype labels and slide names
        df.insert(
            0, cell_types_annotation, 
            adata.obs[cell_types_annotation].values
        )
        if slide_names is not None:
            df.insert(0, slide_names, adata.obs[slide_names].values)

        # Filter for the anchor cell type
        filtered_df = df[df[cell_types_annotation] == anchor_label]

        # Generate a filename for saving
        filename = f"anchor_{anchor_label}_neighbor_{neighbor_label}.csv"

        # Store the DataFrame and filename
        dataframes[(anchor_label, neighbor_label)] = filtered_df
        filenames.append(filename)

    return dataframes, filenames


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python neighborhood_profile_template.py <params.json> [output_dir]",
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

    if isinstance(result, dict):
        print("\nOutput files:")
        for key, paths in result.items():
            if isinstance(paths, list):
                print(f"  {key}:")
                for path in paths[:3]:  # Show first 3 files
                    print(f"    - {path}")
                if len(paths) > 3:
                    print(f"    ... and {len(paths) - 3} more files")
            else:
                print(f"  {key}: {paths}")
    else:
        print("\nReturned dataframes for in-memory use")
