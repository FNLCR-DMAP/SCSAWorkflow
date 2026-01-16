#!/usr/bin/env python3
"""
format_values.py - Normalize parameters JSON for SPAC template consumption.

Shared utility for all Code Ocean SPAC capsules.
Location: code_ocean_tools/format_values.py (root level)

Transformations:
1. Convert boolean strings ("True", "False") to Python booleans
2. Convert comma-separated strings to lists for LIST parameters  
3. Inject output configuration for template_utils.save_results()

Usage:
    python format_values.py params.json cleaned_params.json \
        --bool-values Horizontal_Plot Keep_Outliers \
        --list-values Feature_s_to_Plot \
        --inject-outputs --outputs-config outputs_config.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_boolean(value: Any) -> bool:
    """Convert boolean-like values to Python boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower().strip() in ('true', 'yes', '1', 't')
    return bool(value) if value else False


def parse_list_value(value: Any, separator: str = ',') -> List[str]:
    """
    Parse a value into a list of strings.
    
    Handles:
    - Already a list: ["a", "b"] -> ["a", "b"]
    - Comma-separated string: "a, b, c" -> ["a", "b", "c"]
    - Single value: "a" -> ["a"]
    """
    if value is None:
        return []
    
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict) and 'value' in item:
                val = str(item['value']).strip()
            elif item is not None:
                val = str(item).strip()
            else:
                continue
            if val:
                result.append(val)
        return result
    
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [s.strip() for s in value.split(separator) if s.strip()]
    
    return [str(value)]


def process_params(
    params: Dict[str, Any],
    bool_params: List[str] = None,
    list_params: List[str] = None,
    list_separator: str = ',',
    set_params: Dict[str, str] = None,
    outputs_config: Optional[Dict[str, Any]] = None,
    inject_outputs: bool = False
) -> Dict[str, Any]:
    """Process raw parameters into clean format for templates."""
    bool_params = bool_params or []
    list_params = list_params or []
    set_params = set_params or {}
    
    cleaned = dict(params)
    
    # Convert booleans
    for param_name in bool_params:
        if param_name in cleaned:
            cleaned[param_name] = normalize_boolean(cleaned[param_name])
    
    # Convert lists
    for param_name in list_params:
        if param_name in cleaned:
            cleaned[param_name] = parse_list_value(cleaned[param_name], list_separator)
    
    # Apply overrides
    for param_name, value in set_params.items():
        cleaned[param_name] = value
    
    # Inject outputs
    if inject_outputs and outputs_config:
        cleaned['outputs'] = outputs_config
        cleaned['save_results'] = True
    
    return cleaned


def parse_set_params(args: List[str]) -> Dict[str, str]:
    """Parse --set-param key=value arguments."""
    result = {}
    for item in args or []:
        if '=' in item:
            key, val = item.split('=', 1)
            result[key.strip()] = val.strip()
    return result


def main():
    parser = argparse.ArgumentParser(description="Normalize parameters JSON")
    parser.add_argument("input_json", help="Input JSON file")
    parser.add_argument("output_json", help="Output cleaned JSON file")
    parser.add_argument("--bool-values", nargs="*", default=[], help="Boolean parameters")
    parser.add_argument("--list-values", nargs="*", default=[], help="List parameters")
    parser.add_argument("--list-sep", default=",", help="List separator")
    parser.add_argument("--set-param", nargs="*", default=[], help="Override: key=value")
    parser.add_argument("--inject-outputs", action="store_true", help="Inject outputs config")
    parser.add_argument("--outputs-config", help="Outputs config JSON file")
    
    args = parser.parse_args()
    
    with open(args.input_json) as f:
        params = json.load(f)
    
    outputs_config = None
    if args.outputs_config:
        try:
            with open(args.outputs_config) as f:
                outputs_config = json.load(f)
        except Exception as e:
            print(f"Warning: {e}", file=sys.stderr)
    
    cleaned = process_params(
        params,
        bool_params=args.bool_values,
        list_params=args.list_values,
        list_separator=args.list_sep,
        set_params=parse_set_params(args.set_param),
        outputs_config=outputs_config,
        inject_outputs=args.inject_outputs
    )
    
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(cleaned, f, indent=2)
    
    print(f"Normalized: {args.input_json} -> {args.output_json}")


if __name__ == "__main__":
    main()
