#!/usr/bin/env python3
"""
format_values.py - Complete Galaxy Parameter Processor for SPAC
Standalone file with all functionality integrated
"""

import json
import argparse
import sys
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
    Extract list values from Galaxy repeat structure or text area.
    
    Galaxy generates repeat parameters with '_repeat' suffix:
        "Feature_s_to_Plot_repeat": [{"value": "CD3"}, {"value": "CD4"}]
    
    Text areas generate newline-separated strings:
        "Features_to_Analyze": "CD3\\nCD4\\nCD8"
    
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
    # Check for repeat structure first
    repeat_key = f"{param_name}_repeat"
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
    
    # Check for regular parameter (could be text area or single value)
    if param_name in params:
        value = params[param_name]
        
        # Handle newline-separated text area input
        if isinstance(value, str):
            if '\n' in value:
                # Text area with multiple lines
                return [line.strip() for line in value.split('\n') if line.strip()]
            elif ',' in value:
                # Comma-separated values
                return [v.strip() for v in value.split(',') if v.strip()]
            elif value.strip():
                # Single value
                return [value.strip()]
        elif isinstance(value, list):
            # Already a list
            return [str(v).strip() for v in value if v and str(v).strip()]
    
    # No parameter found, return empty list
    return []


def inject_output_configuration(
    cleaned: Dict[str, Any], 
    outputs_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Inject output configuration for template_utils.save_results().
    
    Parameters
    ----------
    cleaned : dict
        Cleaned parameters dictionary (modified in-place)
    outputs_config : dict, optional
        Output configuration to inject. If None, no outputs are configured.
        Format: {"output_name": {"type": "file|directory", "name": "filename"}}
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
    outputs_config: Optional[Dict[str, Any]] = None,
    inject_outputs: bool = False
) -> Dict[str, Any]:
    """
    Process raw Galaxy parameters to normalize booleans and extract lists.
    
    Parameters
    ----------
    params : dict
        Raw Galaxy parameters (directly from Galaxy)
    bool_params : list
        List of parameter names that should be booleans
    list_params : list
        List of parameter names that should be extracted as lists
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
    
    # Copy all non-repeat parameters first
    for key, value in params.items():
        # Skip repeat parameters (we'll handle them via base name)
        if not key.endswith('_repeat'):
            cleaned[key] = value
    
    # Process boolean parameters
    for param_name in bool_params:
        if param_name in cleaned:
            cleaned[param_name] = normalize_boolean(cleaned[param_name])
    
    # Process list parameters (extract from repeat structures or text areas)
    for param_name in list_params:
        cleaned[param_name] = extract_list_from_repeat(params, param_name)
        
        # Remove the repeat version if it exists in cleaned
        repeat_key = f"{param_name}_repeat"
        if repeat_key in cleaned:
            del cleaned[repeat_key]
    
    # Apply default for Features_to_Analyze if empty (special case)
    if 'Features_to_Analyze' in cleaned and not cleaned['Features_to_Analyze']:
        cleaned['Features_to_Analyze'] = ['All']
    
    # Inject output configuration if requested
    if inject_outputs and outputs_config:
        inject_output_configuration(cleaned, outputs_config)
    
    return cleaned


def load_outputs_config(config_file: str) -> Dict[str, Any]:
    """
    Load output configuration from file or blueprint.
    
    Parameters
    ----------
    config_file : str
        Path to JSON file containing outputs configuration
        
    Returns
    -------
    dict
        Output configuration
    """
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # If it's a full blueprint, extract outputs
        if 'outputs' in config:
            return config['outputs']
        else:
            # Already just the outputs config
            return config
    
    print(f"Warning: outputs config file not found: {config_file}")
    return {}


def main():
    """Main entry point for the format_values utility."""
    parser = argparse.ArgumentParser(
        description="Process Galaxy parameters JSON for SPAC template consumption"
    )
    
    parser.add_argument(
        "input_json",
        help="Input JSON file from Galaxy (raw galaxy_params.json)"
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
        help="Parameter names that should be extracted as lists (from repeats or text areas)"
    )
    
    parser.add_argument(
        "--inject-outputs",
        action="store_true",
        help="Inject output configuration into cleaned parameters"
    )
    
    parser.add_argument(
        "--outputs-config",
        help="Path to outputs configuration JSON file (from blueprint)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information"
    )
    
    args = parser.parse_args()
    
    # Load raw Galaxy parameters
    with open(args.input_json, 'r') as f:
        raw_params = json.load(f)
    
    if args.debug:
        print(f"Loaded {len(raw_params)} parameters from {args.input_json}")
        print(f"Boolean parameters: {args.bool_values}")
        print(f"List parameters: {args.list_values}")
    
    # Load outputs config if provided
    outputs_config = None
    if args.outputs_config:
        outputs_config = load_outputs_config(args.outputs_config)
        if args.debug and outputs_config:
            print(f"Loaded outputs config with {len(outputs_config)} outputs")
    
    # Process parameters
    cleaned = process_galaxy_params(
        params=raw_params,
        bool_params=args.bool_values,
        list_params=args.list_values,
        outputs_config=outputs_config,
        inject_outputs=args.inject_outputs
    )
    
    # Save cleaned parameters
    with open(args.output_json, 'w') as f:
        json.dump(cleaned, f, indent=2)
    
    if args.debug:
        print(f"\nCleaned parameters saved to: {args.output_json}")
        print(f"  - Processed {len(args.bool_values)} boolean parameters")
        print(f"  - Extracted {len(args.list_values)} list parameters")
        if args.inject_outputs and outputs_config:
            print(f"  - Injected {len(outputs_config)} output configurations")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
