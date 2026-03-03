#!/usr/bin/env python3
"""
format_values.py - Utility to normalize parameters JSON for template consumption.

Version 8.0 - Code Ocean compatible version

Handles transformations:
1. Flattens nested section parameters
2. Converts Galaxy repeat structures to simple lists
3. Converts boolean string values to actual Python booleans
4. Supports delimited text fields
5. Sets parameter values via --set-param
6. Injects output configuration for template_utils

Usage:
    python format_values.py params.json cleaned_params.json \
        --bool-values Horizontal_Plot Keep_Outliers \
        --list-values Feature_s_to_Plot
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_boolean(value: Any) -> bool:
    """Convert boolean strings to Python boolean."""
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
    """Extract list values from repeat structure or comma-separated string."""
    repeat_key = f"{param_name}_repeat"
    
    # Check if parameter exists directly
    if param_name in params:
        value = params[param_name]
        if isinstance(value, list):
            return [str(v).strip() for v in value if v and str(v).strip()]
        elif isinstance(value, str):
            # Handle comma-separated values
            return [s.strip() for s in value.split(',') if s.strip()]
    
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


def flatten_section_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested section parameters."""
    flattened = {}
    
    for key, value in params.items():
        if isinstance(value, dict) and key != 'outputs':
            flattened.update(value)
        else:
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


def process_params(
    params: Dict[str, Any],
    bool_params: List[str],
    list_params: List[str],
    set_params: Dict[str, str] = None,
    outputs_config: Optional[Dict[str, Any]] = None,
    inject_outputs: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """Process raw parameters to normalize structure for template consumption."""
    set_params = set_params or {}
    
    # Step 1: Flatten section-nested parameters
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
        cleaned[param_name] = extract_list_from_repeat(params, param_name)
        repeat_key = f"{param_name}_repeat"
        if repeat_key in cleaned:
            del cleaned[repeat_key]
    
    # Step 5: Apply parameter overrides
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Normalize parameters JSON for template consumption"
    )
    parser.add_argument("input_json", help="Input JSON file")
    parser.add_argument("output_json", help="Output cleaned JSON file")
    parser.add_argument("--bool-values", nargs="*", default=[],
                        help="Parameter names to convert to booleans")
    parser.add_argument("--list-values", nargs="*", default=[],
                        help="Parameter names to extract as lists")
    parser.add_argument("--set-param", nargs="*", default=[],
                        help="Set parameter values: param=value")
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
    
    if args.debug:
        print("=== Original Parameters ===", file=sys.stderr)
        print(json.dumps(params, indent=2), file=sys.stderr)
        print(f"\nBool params: {args.bool_values}", file=sys.stderr)
        print(f"List params: {args.list_values}", file=sys.stderr)
        print(f"Set params: {set_params}", file=sys.stderr)
    
    # Process parameters
    cleaned_params = process_params(
        params,
        bool_params=args.bool_values or [],
        list_params=args.list_values or [],
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


if __name__ == "__main__":
    main()
