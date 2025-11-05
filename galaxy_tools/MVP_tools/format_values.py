#!/usr/bin/env python3
"""
format_values.py - Utility to normalize Galaxy parameters JSON for template consumption.

Version 2.0 - No Cheetah needed! Processes Galaxy repeat structures directly.

Handles three main transformations:
1. Converts Galaxy repeat structures to simple lists
2. Converts boolean string values to actual Python booleans
3. Injects output configuration for template_utils (single files for boxplot MVP)

Usage:
    python format_values.py galaxy_params.json cleaned_params.json \
        --bool-values Horizontal_Plot Keep_Outliers Value_Axis_Log_Scale \
        --list-values Feature_s_to_Plot

This processes Galaxy's raw parameter output directly without needing Cheetah preprocessing.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


def normalize_boolean(value: Any) -> bool:
    """
    Convert Galaxy boolean strings to Python boolean.
    
    Parameters
    ----------
    value : Any
        Value to convert (typically "True"/"False" strings from Galaxy)
        
    Returns
    -------
    bool
        Python boolean value
    """
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ('true', 'yes', '1', 't'):
            return True
        elif value_lower in ('false', 'no', '0', 'f', 'none', ''):
            return False
    
    # Default to False for unexpected values
    return bool(value) if value else False


def extract_list_from_repeat(params: Dict[str, Any], param_name: str) -> List[str]:
    """
    Extract list values from Galaxy repeat structure.
    
    Galaxy generates repeat parameters with '_repeat' suffix:
    "Feature_s_to_Plot_repeat": [{"value": "CD3"}, {"value": "CD4"}]
    
    We extract to: ["CD3", "CD4"]
    
    Parameters
    ----------
    params : dict
        Full Galaxy parameters dictionary
    param_name : str
        Base parameter name (without _repeat suffix)
        
    Returns
    -------
    list
        Simple list of string values
    """
    # Check for the repeat version of the parameter
    repeat_key = f"{param_name}_repeat"
    
    # Also check if parameter exists without _repeat (for backward compatibility)
    if param_name in params:
        value = params[param_name]
        # If it's already a simple list, return it
        if isinstance(value, list) and (not value or not isinstance(value[0], dict)):
            return [str(v).strip() for v in value if v and str(v).strip()] or ["All"]
    
    # Process repeat structure
    if repeat_key in params:
        repeat_value = params[repeat_key]
        
        if isinstance(repeat_value, list):
            result = []
            for item in repeat_value:
                if isinstance(item, dict) and 'value' in item:
                    val = str(item['value']).strip()
                    if val:  # Skip empty values
                        result.append(val)
            
            # Return result or default to ["All"] if empty
            return result if result else ["All"]
    
    # No parameter found, return default
    return ["All"]


def inject_output_directories(cleaned: Dict[str, Any]) -> None:
    """
    Inject output configuration for template_utils.save_results().
    
    For boxplot: single figure file + single summary file (no directories)
    This matches what boxplot_template.py expects (lines 73-76).
    
    Parameters
    ----------
    cleaned : dict
        Cleaned parameters dictionary (modified in-place)
    """
    # Configure outputs for template_utils.save_results()
    # Must match the blueprint structure expected by template
    cleaned['outputs'] = {
        'figure': {'type': 'file', 'name': 'boxplot.png'},
        'summary': {'type': 'file', 'name': 'summary.csv'}
    }
    
    # Enable result saving
    cleaned['save_results'] = True


def process_galaxy_params(
    params: Dict[str, Any],
    bool_params: List[str],
    list_params: List[str]
) -> Dict[str, Any]:
    """
    Process raw Galaxy parameters to normalize booleans and extract lists from repeats.
    
    Parameters
    ----------
    params : dict
        Raw Galaxy parameters (directly from Galaxy, no Cheetah processing)
    bool_params : list
        List of parameter names that should be booleans
    list_params : list
        List of parameter names that should be extracted from repeat structures
        
    Returns
    -------
    dict
        Cleaned parameters ready for template consumption
    """
    cleaned = {}
    
    # Copy all non-repeat parameters first
    for key, value in params.items():
        # Skip repeat parameters (we'll handle them separately)
        if not key.endswith('_repeat'):
            cleaned[key] = value
    
    # Process boolean parameters
    for param_name in bool_params:
        if param_name in cleaned:
            cleaned[param_name] = normalize_boolean(cleaned[param_name])
    
    # Process list parameters (extract from repeat structures)
    for param_name in list_params:
        cleaned[param_name] = extract_list_from_repeat(params, param_name)
        
        # Remove the repeat version if it exists in cleaned
        repeat_key = f"{param_name}_repeat"
        if repeat_key in cleaned:
            del cleaned[repeat_key]
    
    # Inject output configuration (boxplot MVP uses single files)
    inject_output_directories(cleaned)
    
    return cleaned


def main():
    """Main entry point for the format_values utility."""
    parser = argparse.ArgumentParser(
        description="Normalize Galaxy parameters JSON for template consumption"
    )
    parser.add_argument(
        "input_json",
        help="Input JSON file from Galaxy (raw galaxy parameters)"
    )
    parser.add_argument(
        "output_json",
        help="Output cleaned JSON file (cleaned_params.json)"
    )
    parser.add_argument(
        "--bool-values",
        nargs="*",
        default=[],
        help="Parameter names that should be converted to booleans"
    )
    parser.add_argument(
        "--list-values",
        nargs="*",
        default=[],
        help="Parameter names that should be extracted from repeat structures"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Read input JSON
    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(input_path, 'r') as f:
            params = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.debug:
        print("=== Original Galaxy Parameters ===", file=sys.stderr)
        print(json.dumps(params, indent=2), file=sys.stderr)
        print("\nBoolean parameters to convert:", args.bool_values, file=sys.stderr)
        print("List parameters to extract from repeats:", args.list_values, file=sys.stderr)
    
    # Process parameters
    cleaned_params = process_galaxy_params(
        params,
        bool_params=args.bool_values or [],
        list_params=args.list_values or []
    )
    
    if args.debug:
        print("\n=== Cleaned Parameters ===", file=sys.stderr)
        print(json.dumps(cleaned_params, indent=2), file=sys.stderr)
        print("\n=== Output Configuration ===", file=sys.stderr)
        print(f"  save_results: {cleaned_params.get('save_results')}", file=sys.stderr)
        print(f"  outputs: {cleaned_params.get('outputs')}", file=sys.stderr)
    
    # Write output JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(cleaned_params, f, indent=2)
    
    print(f"Successfully normalized parameters to: {output_path}")
    
    # Log transformations for visibility
    for param in args.bool_values or []:
        if param in params or param in cleaned_params:
            original = params.get(param, "N/A")
            cleaned = cleaned_params.get(param, "N/A")
            if original != cleaned:
                print(f"  {param}: '{original}' → {cleaned}")
    
    for param in args.list_values or []:
        repeat_key = f"{param}_repeat"
        if repeat_key in params:
            original = params[repeat_key]
            cleaned = cleaned_params.get(param, [])
            print(f"  {param}: Extracted {len(cleaned)} values from repeat structure")
    
    # Confirm output configuration injection
    print(f"  Output configuration injected: {cleaned_params.get('outputs')}")


if __name__ == "__main__":
    main()
