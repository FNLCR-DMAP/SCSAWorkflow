#!/usr/bin/env python3
"""
format_values.py - Utility to normalize Galaxy parameters JSON for template consumption.

Version 3.0 - Clean version per supervisor guidance
              NO parameter-specific hardcoding
              Generic delimiter support via CLI flags

Handles three main transformations:
1. Converts Galaxy repeat structures to simple lists
2. Converts boolean string values to actual Python booleans
3. Supports delimited text fields (e.g., text areas with semicolons or newlines)
4. Injects output configuration for template_utils

Usage:
    # For repeat structures:
    python format_values.py galaxy_params.json cleaned_params.json \
        --bool-values Horizontal_Plot Keep_Outliers \
        --list-values Feature_s_to_Plot
    
    # For text area with newline separators:
    python format_values.py galaxy_params.json cleaned_params.json \
        --list-sep '\\n' --list-fields Anchor_Neighbor_List

This version is template-agnostic with no parameter name hardcoding.
"""

import json
import argparse
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        Simple list of string values, or empty list if no values found
    """
    repeat_key = f"{param_name}_repeat"
    
    # Check if parameter exists without _repeat (for backward compatibility)
    if param_name in params:
        value = params[param_name]
        # If it's already a simple list, return it
        if isinstance(value, list) and (not value or not isinstance(value[0], dict)):
            result = [str(v).strip() for v in value if v and str(v).strip()]
    
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
            return result

    # No parameter found, return empty list
    return []


def parse_delimited_text(text: str, separator: str = ';') -> List[str]:
    """
    Parse a delimited text string into a list of values.
    
    Parameters
    ----------
    text : str
        Delimited text string (may contain newlines for multi-line input)
    separator : str
        Separator character (default: semicolon)
        
    Returns
    -------
    list
        List of trimmed, non-empty values
    """
    if not text:
        return []
    
    # Handle newline separator for text areas
    if separator == '\n':
        # Split by newlines
        lines = text.strip().split('\n')
        # Clean up each line
        values = [line.strip() for line in lines if line.strip()]
    else:
        # Split by specified separator
        values = [s.strip() for s in text.split(separator) if s.strip()]
    
    return values

def inject_output_configuration(cleaned: Dict[str, Any], outputs_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Inject output configuration for template_utils.save_results().
    
    This is a generic method that accepts output configuration as a parameter
    rather than trying to detect the tool type.
    
    Parameters
    ----------
    cleaned : dict
        Cleaned parameters dictionary (modified in-place)
    outputs_config : dict, optional
        Output configuration to inject. If None, no outputs are configured.
        Format: {"output_name": {"type": "file", "name": "filename"}}
    """
    if outputs_config:
        cleaned['outputs'] = outputs_config
    
    # Enable result saving if outputs are configured
    if cleaned.get('outputs'):
        cleaned['save_results'] = True


def process_galaxy_params(
    params: Dict[str, Any],
    bool_params: List[str],
    list_params: List[str],
    delimited_params: Dict[str, str] = None,
    outputs_config: Optional[Dict[str, Any]] = None,
    inject_outputs: bool = False
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
    delimited_params : dict, optional
        Parameters with delimited text {param_name: separator}
        For text areas or fields that use delimiters instead of repeat structures
    outputs_config : dict, optional
        Output configuration to inject if inject_outputs is True
    inject_outputs : bool, optional
        Whether to inject output configuration (default False)
        
    Returns
    -------
    dict
        Cleaned parameters ready for template consumption
    """
    cleaned = {}
    delimited_params = delimited_params or {}
    
    # Copy all non-repeat parameters first
    for key, value in params.items():
        # Skip repeat parameters (we'll handle them separately)
        if not key.endswith('_repeat'):
            cleaned[key] = value
    
    # Process boolean parameters
    for param_name in bool_params:
        if param_name in cleaned:
            cleaned[param_name] = normalize_boolean(cleaned[param_name])
    
    # Process list parameters (extract from repeat structures or delimited text)
    for param_name in list_params:
        # Check if this is a delimited parameter (text area with separator)
        if param_name in delimited_params:
            # Handle as delimited text
            separator = delimited_params[param_name]
            if param_name in params and isinstance(params[param_name], str):
                cleaned[param_name] = parse_delimited_text(params[param_name], separator)
            else:
                cleaned[param_name] = []
        else:
            # Handle as repeat structure
            cleaned[param_name] = extract_list_from_repeat(params, param_name)
        
        # Remove the repeat version if it exists in cleaned
        repeat_key = f"{param_name}_repeat"
        if repeat_key in cleaned:
            del cleaned[repeat_key]
    
    # Only inject output configuration if requested
    # Templates should handle their own output configuration
    if inject_outputs and outputs_config:
        inject_output_configuration(cleaned, outputs_config)
    
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
        "--list-sep",
        default=None,
        help='Separator for delimited list fields (e.g., ";" or "\\n" for newlines)'
    )
    parser.add_argument(
        "--list-fields",
        nargs="*",
        default=[],
        help="Fields to parse as delimited lists using list-sep"
    )
    parser.add_argument(
        "--inject-outputs",
        action="store_true",
        help="Inject output configuration for template_utils"
    )
    parser.add_argument(
        "--outputs-config",
        help="JSON file containing output configuration"
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
    
    # Read outputs configuration if provided
    outputs_config = None
    if args.outputs_config:
        try:
            with open(args.outputs_config, 'r') as f:
                outputs_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not read outputs config: {e}", file=sys.stderr)
    
    if args.debug:
        print("=== Original Galaxy Parameters ===", file=sys.stderr)
        print(json.dumps(params, indent=2), file=sys.stderr)
        print("\nBoolean parameters to convert:", args.bool_values, file=sys.stderr)
        print("List parameters to extract from repeats:", args.list_values, file=sys.stderr)
        if args.list_sep and args.list_fields:
            print(f"Delimited fields using '{args.list_sep}' separator:", args.list_fields, file=sys.stderr)
    
    # Build delimited parameters map
    delimited_params = {}
    if args.list_sep and args.list_fields:
        # Handle escape sequences for separator
        separator = args.list_sep
        if separator == '\\n':
            separator = '\n'
        elif separator == '\\t':
            separator = '\t'
        
        for field in args.list_fields:
            delimited_params[field] = separator
    
    # Process parameters
    cleaned_params = process_galaxy_params(
        params,
        bool_params=args.bool_values or [],
        list_params=args.list_values or [],
        delimited_params=delimited_params,
        outputs_config=outputs_config,
        inject_outputs=args.inject_outputs
    )
    
    if args.debug:
        print("\n=== Cleaned Parameters ===", file=sys.stderr)
        print(json.dumps(cleaned_params, indent=2), file=sys.stderr)
    
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
                print(f"  {param}: '{original}' â†’ {cleaned}")
    
    for param in args.list_values or []:
        repeat_key = f"{param}_repeat"
        if repeat_key in params or param in delimited_params:
            cleaned = cleaned_params.get(param, [])
            if param in delimited_params:
                sep_display = repr(delimited_params[param])
                print(f"  {param}: Parsed {len(cleaned)} values using {sep_display} separator")
            elif repeat_key in params:
                print(f"  {param}: Extracted {len(cleaned)} values from repeat structure")


if __name__ == "__main__":
    main()
