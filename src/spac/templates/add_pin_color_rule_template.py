"""
Platform-agnostic Append Pin Color Rule template converted from NIDAP.
Maintains the exact logic from the NIDAP template.

Usage
-----
>>> from spac.templates.add_pin_color_rule_template import run_from_json
>>> run_from_json("examples/add_pin_color_rule_params.json")
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from spac.data_utils import add_pin_color_rules
from spac.templates.template_utils import (
    load_input,
    save_outputs,
    parse_params,
    string_list_to_dictionary,
)


def run_from_json(
    json_path: Union[str, Path, Dict[str, Any]],
    save_results: bool = True
) -> Union[Dict[str, str], Any]:
    """
    Execute Append Pin Color Rule analysis with parameters from JSON.
    Replicates the NIDAP template functionality exactly.

    Parameters
    ----------
    json_path : str, Path, or dict
        Path to JSON file, JSON string, or parameter dictionary
    save_results : bool, optional
        Whether to save results to file. If False, returns the adata object
        directly for in-memory workflows. Default is True.

    Returns
    -------
    dict or AnnData
        If save_results=True: Dictionary of saved file paths
        If save_results=False: The processed AnnData object
    """
    # Parse parameters from JSON
    params = parse_params(json_path)

    # Load the upstream analysis data
    adata = load_input(params["Upstream_Analysis"])

    # Extract parameters
    color_dict_string_list = params.get("Label_Color_Map", [])
    color_map_name = params.get("Color_Map_Name", "_spac_colors")
    overwrite = params.get("Overwrite_Previous_Color_Map", True)

    color_dict = string_list_to_dictionary(
        color_dict_string_list,
        key_name="label",
        value_name="color"
    )

    add_pin_color_rules(
        adata,
        label_color_dict=color_dict,
        color_map_name=color_map_name,
        overwrite=overwrite
    )
    print(adata.uns[f'{color_map_name}_summary'])

    # Handle results based on save_results flag
    if save_results:
        # Save outputs
        output_file = params.get("Output_File", "color_mapped_analysis.pickle")
        # Default to pickle format if no recognized extension
        if not output_file.endswith(('.pickle', '.pkl', '.h5ad')):
            output_file = output_file + '.pickle'
        
        saved_files = save_outputs({output_file: adata})
        
        print(f"Append Pin Color Rule completed → {saved_files[output_file]}")
        return saved_files
    else:
        # Return the adata object directly for in-memory workflows
        print("Returning AnnData object (not saving to file)")
        return adata


# CLI interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python add_pin_color_rule_template.py <params.json>",
            file=sys.stderr
        )
        sys.exit(1)

    result = run_from_json(sys.argv[1])

    if isinstance(result, dict):
        print("\nOutput files:")
        for filename, filepath in result.items():
            print(f"  {filename}: {filepath}")
    else:
        print("\nReturned AnnData object")