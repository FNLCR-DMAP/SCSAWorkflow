#!/usr/bin/env python3
"""
format_values.py - Utility to normalize Galaxy parameters JSON for template consumption.

Version 5.0 - Added automatic flattening of Galaxy XML section-nested parameters

Handles transformations:
1. Flattens nested section parameters (e.g., {"section_name": {"param": value}} → {"param": value})
2. Converts Galaxy repeat structures to simple lists
3. Converts boolean string values to actual Python booleans
4. Supports delimited text fields (e.g., text areas with semicolons or newlines)
5. Sets parameter values via --set-param (for staged directory paths)
6. Injects output configuration for template_utils

Usage:
    python format_values.py galaxy_params.json cleaned_params.json \
        --bool-values Horizontal_Plot Keep_Outliers \
        --list-values Feature_s_to_Plot \
        --set-param CSV_Files=csv_files_staged

This version is template-agnostic with no parameter name hardcoding.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_boolean(value: Any) -> bool:
    """Convert Galaxy boolean strings to Python boolean."""
    if isinstance(value, bool):
        return value
    
    if isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ('true', 'yes', '1', 't'):
            return True
        elif value_lower in ('false', 'no', '0', 'f', 'none', ''):
            return False
    
    return bool(value) if value else False


def extract_list_from_repeat(params: Dict[str, Any], param_name: str) -> List[str]:
    """
    Extract list values from Galaxy repeat structure.
    
    Galaxy generates: "Feature_s_to_Plot_repeat": [{"value": "CD3"}, {"value": "CD4"}]
    We extract to: ["CD3", "CD4"]
    """
    repeat_key = f"{param_name}_repeat"
    
    # Check if parameter exists without _repeat (backward compatibility)
    if param_name in params:
        value = params[param_name]
        if isinstance(value, list) and (not value or not isinstance(value[0], dict)):
            return [str(v).strip() for v in value if v and str(v).strip()]
    
    # Process repeat structure
    if repeat_key in params:
        repeat_value = params[repeat_key]
        if isinstance(repeat_value, list):
            result = []
            for item in repeat_value:
                if isinstance(item, dict) and 'value' in item:
                    val = str(item['value']).strip()
                    if val:
                        result.append(val)
            return result

    return []


def parse_delimited_text(text: str, separator: str = ';') -> List[str]:
    """Parse a delimited text string into a list of values."""
    if not text:
        return []
    
    if separator == '\n':
        lines = text.strip().split('\n')
        values = [line.strip() for line in lines if line.strip()]
    else:
        values = [s.strip() for s in text.split(separator) if s.strip()]
    
    return values


def flatten_section_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested section parameters from Galaxy XML sections.
    
    Galaxy XML <section> tags create nested JSON structures:
        {"section_name": {"param1": value1, "param2": value2}}
    
    Templates expect flat parameter access:
        {"param1": value1, "param2": value2}
    
    This function promotes nested section contents to the top level,
    preserving special keys like 'outputs' that should remain nested.
    
    Parameters
    ----------
    params : dict
        Raw parameters dict potentially containing nested sections
        
    Returns
    -------
    dict
        Flattened parameters with section contents at top level
        
    Example
    -------
    Input:
        {
            "Upstream_Analysis": "data.pickle",
            "plot_by_parameters": {
                "Plot_By": "Feature",
                "Feature": "CD21"
            }
        }
    
    Output:
        {
            "Upstream_Analysis": "data.pickle",
            "Plot_By": "Feature",
            "Feature": "CD21"
        }
    """
    flattened = {}
    
    for key, value in params.items():
        if isinstance(value, dict) and key != 'outputs':
            # This is a Galaxy section - promote its contents to top level
            flattened.update(value)
        else:
            # Regular parameter or 'outputs' config - keep as-is
            flattened[key] = value
    
    return flattened


def inject_output_configuration(
    cleaned: Dict[str, Any],
    outputs_config: Optional[Dict[str, Any]] = None
) -> None:
    """Inject output configuration for template_utils.save_results()."""
    if outputs_config:
        cleaned['outputs'] = outputs_config
    
    if cleaned.get('outputs'):
        cleaned['save_results'] = True


def process_galaxy_params(
    params: Dict[str, Any],
    bool_params: List[str],
    list_params: List[str],
    delimited_params: Dict[str, str] = None,
    set_params: Dict[str, str] = None,
    outputs_config: Optional[Dict[str, Any]] = None,
    inject_outputs: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Process raw Galaxy parameters to normalize structure for template consumption.
    
    Processing steps (in order):
    1. Flatten section-nested parameters
    2. Copy non-repeat parameters
    3. Convert boolean strings to Python booleans
    4. Extract list values from repeat structures
    5. Apply parameter overrides
    6. Inject output configuration
    
    Parameters
    ----------
    params : dict
        Raw parameters from Galaxy JSON
    bool_params : list
        Parameter names to convert to booleans
    list_params : list
        Parameter names to extract from repeat structures
    delimited_params : dict, optional
        Mapping of param names to their delimiter characters
    set_params : dict, optional
        Parameter overrides (e.g., staged directory paths)
    outputs_config : dict, optional
        Output configuration to inject
    inject_outputs : bool
        Whether to inject output configuration
    debug : bool
        Enable debug output
        
    Returns
    -------
    dict
        Cleaned, flattened parameters ready for template consumption
    """
    delimited_params = delimited_params or {}
    set_params = set_params or {}
    
    # Step 1: Flatten section-nested parameters from Galaxy XML
    # This must happen first so subsequent processing sees flat keys
    params = flatten_section_params(params)
    
    if debug:
        print("=== After Flattening Sections ===", file=sys.stderr)
        print(json.dumps(params, indent=2), file=sys.stderr)
    
    # Step 2: Copy all non-repeat parameters
    cleaned = {}
    for key, value in params.items():
        if not key.endswith('_repeat'):
            cleaned[key] = value
    
    # Step 3: Process boolean parameters
    for param_name in bool_params:
        if param_name in cleaned:
            cleaned[param_name] = normalize_boolean(cleaned[param_name])
    
    # Step 4: Process list parameters
    for param_name in list_params:
        if param_name in delimited_params:
            separator = delimited_params[param_name]
            if param_name in params and isinstance(params[param_name], str):
                cleaned[param_name] = parse_delimited_text(params[param_name], separator)
            else:
                cleaned[param_name] = []
        else:
            cleaned[param_name] = extract_list_from_repeat(params, param_name)
        
        repeat_key = f"{param_name}_repeat"
        if repeat_key in cleaned:
            del cleaned[repeat_key]
    
    # Step 5: Apply parameter overrides (e.g., staged directory paths)
    for param_name, value in set_params.items():
        cleaned[param_name] = value
        if debug:
            print(f"[format_values] Set {param_name} = {value}", file=sys.stderr)
    
    # Step 6: Inject output configuration
    if inject_outputs and outputs_config:
        inject_output_configuration(cleaned, outputs_config)
    
    return cleaned


def parse_set_param_args(set_param_list: List[str]) -> Dict[str, str]:
    """Parse --set-param arguments: param_name=value"""
    result = {}
    for item in set_param_list or []:
        if '=' in item:
            param, value = item.split('=', 1)
            result[param.strip()] = value.strip()
        else:
            print(f"Warning: Invalid --set-param format: {item}", file=sys.stderr)
    return result


def main():
    """Main entry point for the format_values utility."""
    parser = argparse.ArgumentParser(
        description="Normalize Galaxy parameters JSON for template consumption"
    )
    parser.add_argument("input_json", help="Input JSON file from Galaxy")
    parser.add_argument("output_json", help="Output cleaned JSON file")
    parser.add_argument("--bool-values", nargs="*", default=[],
                        help="Parameter names to convert to booleans")
    parser.add_argument("--list-values", nargs="*", default=[],
                        help="Parameter names to extract from repeat structures")
    parser.add_argument("--list-sep", default=None,
                        help='Separator for delimited list fields')
    parser.add_argument("--list-fields", nargs="*", default=[],
                        help="Fields to parse as delimited lists")
    parser.add_argument("--set-param", nargs="*", default=[],
                        help="Set parameter values: param=value (e.g., CSV_Files=csv_files_staged)")
    parser.add_argument("--inject-outputs", action="store_true",
                        help="Inject output configuration")
    parser.add_argument("--outputs-config", help="JSON file with output config")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
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
    
    # Read outputs configuration
    outputs_config = None
    if args.outputs_config:
        try:
            with open(args.outputs_config, 'r') as f:
                outputs_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not read outputs config: {e}", file=sys.stderr)
    
    # Parse arguments
    set_params = parse_set_param_args(args.set_param)
    
    delimited_params = {}
    if args.list_sep and args.list_fields:
        separator = args.list_sep
        if separator == '\\n':
            separator = '\n'
        elif separator == '\\t':
            separator = '\t'
        for field in args.list_fields:
            delimited_params[field] = separator
    
    if args.debug:
        print("=== Original Galaxy Parameters ===", file=sys.stderr)
        print(json.dumps(params, indent=2), file=sys.stderr)
        print(f"\nBool params: {args.bool_values}", file=sys.stderr)
        print(f"List params: {args.list_values}", file=sys.stderr)
        print(f"Set params: {set_params}", file=sys.stderr)
    
    # Process parameters
    cleaned_params = process_galaxy_params(
        params,
        bool_params=args.bool_values or [],
        list_params=args.list_values or [],
        delimited_params=delimited_params,
        set_params=set_params,
        outputs_config=outputs_config,
        inject_outputs=args.inject_outputs,
        debug=args.debug
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


if __name__ == "__main__":
    main()
