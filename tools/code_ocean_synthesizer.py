#!/usr/bin/env python3
"""
Code Ocean Synthesizer - Generates Code Ocean capsule files from blueprint JSON.

Version: 8.0 
Features:
- Generates .codeocean/app-panel.json with named_parameters: true
- Generates SINGLE run.sh file (combined entry point + parameter parsing)
- NO separate main.sh - cleaner structure
- Named argument parsing: --param_name=value format
- Proper type handling: numeric values written WITHOUT quotes in JSON
- Correct output directory: passes output_dir='/results' to templates
- Automatic category creation from paramGroup
- Shared format_values.py generated once at root level
- Docker container: nciccbr/spac:v2-dev

Key Changes in v8.0:
1. Combined run.sh + main.sh into single run.sh
2. Cleaner structure - one less file per capsule
3. "Set file to run" only recognizes run.sh anyway

Type Mapping (Blueprint -> Code Ocean App Panel):
- STRING -> type: "text", value_type: "string"
- INTEGER/INT/NUMBER/FLOAT -> type: "text", value_type: "number"
- BOOLEAN -> type: "list", value_type: "boolean"
- SELECT -> type: "list", value_type: "string" (with extra_data options)
- LIST -> type: "text", value_type: "string" (comma-separated)
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import argparse
import sys
from collections import OrderedDict


class CodeOceanSynthesizer:
    """
    Synthesize Code Ocean capsule files from blueprint JSON.
    
    Generates:
    - .codeocean/app-panel.json (UI configuration with named parameters)
    - run.sh (SINGLE file: entry point + parameter parsing + execution)
    
    Note: v8.0 removes separate main.sh - everything is in run.sh
    """
    
    # Parameter types that should be written as numbers (no quotes in JSON)
    NUMERIC_TYPES = {'NUMBER', 'FLOAT', 'INTEGER', 'INT', 'INTERGER', 'Positive integer'}
    
    def __init__(self, blueprint: Dict[str, Any], docker_image: str = "nciccbr/spac:v2-dev"):
        self.blueprint = blueprint
        self.docker_image = docker_image
        self._parameter_order = []  # Track parameter order
        
    def synthesize(self) -> Dict[str, str]:
        """
        Generate all Code Ocean capsule files from blueprint.
        
        Returns
        -------
        dict
            Dictionary mapping filename to content:
            {
                ".codeocean/app-panel.json": "...",
                "run.sh": "...",
                "format_values.py": "..."
            }
            
        Note: v8.0 generates 3 files per tool (format_values.py in same directory as run.sh)
        """
        files = {}
        
        # Generate app-panel.json with named parameters
        app_panel = self._generate_app_panel()
        files[".codeocean/app-panel.json"] = json.dumps(app_panel, indent="\t")
        
        # Generate combined run.sh (entry point + parameter parsing + execution)
        files["run.sh"] = self._generate_run_sh()
        
        # Generate format_values.py (in same directory as run.sh)
        files["format_values.py"] = generate_format_values_py()
        
        return files
    
    def _make_tool_id(self, title: str) -> str:
        """Generate tool ID from title (used for template module name)."""
        clean_title = re.sub(r'\[.*?\]', '', title).strip()
        tool_id = clean_title.lower().replace(' ', '_')
        tool_id = tool_id.replace('\\', '_').replace('/', '_').replace("'", "").replace("-", "_")
        tool_id = re.sub(r'_+', '_', tool_id)
        tool_id = tool_id.strip('_')
        return tool_id
    
    def _make_category_id(self, group_name: str) -> str:
        """Convert paramGroup to category ID."""
        if not group_name:
            return "general"
        cat_id = group_name.lower().replace(' ', '_')
        cat_id = re.sub(r'[^a-z0-9_]', '', cat_id)
        return cat_id
    
    def _clean_text(self, text: str) -> str:
        """Clean text from markdown and escapes."""
        if not text:
            return ""
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = text.replace('\\n', ' ').replace('\n', ' ')
        return text.strip()
    
    def _generate_unique_id(self) -> str:
        """Generate a unique ID for app panel parameters."""
        return ''.join(c for c in str(uuid.uuid4())[:16] if c.isalnum())
    
    def _get_app_panel_type(self, param_type: str) -> Tuple[str, str]:
        """
        Map blueprint paramType to Code Ocean app panel type.
        
        Returns
        -------
        tuple
            (type, value_type) for app-panel.json
        """
        param_type = param_type.upper() if param_type else 'STRING'
        
        if param_type == 'BOOLEAN':
            return ('list', 'boolean')
        elif param_type == 'SELECT':
            return ('list', 'string')
        elif param_type in self.NUMERIC_TYPES:
            return ('text', 'number')
        else:  # STRING, TEXT, LIST, etc.
            return ('text', 'string')
    
    def _is_numeric_type(self, param_type: str) -> bool:
        """Check if parameter type should be written as number (no quotes)."""
        return param_type.upper() in self.NUMERIC_TYPES if param_type else False
    
    def _generate_app_panel(self) -> Dict[str, Any]:
        """
        Generate .codeocean/app-panel.json content with NAMED PARAMETERS.
        """
        title = self.blueprint.get('title', 'SPAC Tool')
        description = self._clean_text(self.blueprint.get('description', ''))
        
        app_panel = {
            "version": 1,
            "named_parameters": True,  # Use named parameters
            "general": {
                "title": title,
                "instructions": description[:500] if description else f"Run {title}"
            }
        }
        
        # Collect categories from paramGroup
        categories = OrderedDict()
        categories["general"] = {
            "id": "general",
            "name": "General Parameters",
            "help_text": "Basic tool parameters"
        }
        
        # First pass: collect all categories
        for param in self.blueprint.get('parameters', []):
            group = param.get('paramGroup')
            if group:
                cat_id = self._make_category_id(group)
                if cat_id not in categories:
                    categories[cat_id] = {
                        "id": cat_id,
                        "name": group,
                        "help_text": f"Configure {group.lower()}"
                    }
        
        app_panel["categories"] = list(categories.values())
        
        # Generate parameters (excluding input datasets - those are handled by data attachment)
        parameters = []
        self._parameter_order = []
        
        # Get ordered keys or use natural order
        ordered_keys = self.blueprint.get('orderedMustacheKeys', [])
        params_dict = {p['key']: p for p in self.blueprint.get('parameters', [])}
        
        # Process in order, skipping input datasets
        input_dataset_keys = {d['key'] for d in self.blueprint.get('inputDatasets', [])}
        
        if ordered_keys:
            keys_to_process = [k for k in ordered_keys if k in params_dict]
        else:
            keys_to_process = list(params_dict.keys())
        
        for key in keys_to_process:
            if key in input_dataset_keys:
                continue
                
            param = params_dict[key]
            param_type = param.get('paramType', 'STRING')
            app_type, value_type = self._get_app_panel_type(param_type)
            
            group = param.get('paramGroup')
            cat_id = self._make_category_id(group) if group else "general"
            
            app_param = {
                "id": self._generate_unique_id(),
                "category": cat_id,
                "name": param.get('displayName', key),
                "param_name": key,  # This is used for named parameter argument
                "description": self._clean_text(param.get('description', '')),
                "type": app_type,
                "value_type": value_type
            }
            
            # Add default value - numeric types as numbers, others as strings
            default_value = param.get('defaultValue')
            if default_value is not None:
                if self._is_numeric_type(param_type):
                    # Keep as number for numeric types
                    try:
                        # Try to parse as float first, then int if it's a whole number
                        num_val = float(default_value)
                        if num_val.is_integer():
                            app_param["default_value"] = int(num_val)
                        else:
                            app_param["default_value"] = num_val
                    except (ValueError, TypeError):
                        # Fallback to string if parsing fails
                        app_param["default_value"] = str(default_value)
                else:
                    app_param["default_value"] = str(default_value)
            
            # Add options for SELECT type
            if param_type == 'SELECT':
                param_values = param.get('paramValues', [])
                app_param["extra_data"] = param_values if param_values else []
            elif param_type == 'BOOLEAN':
                app_param["extra_data"] = []  # Empty for boolean dropdown
            
            parameters.append(app_param)
            self._parameter_order.append({
                'key': key,
                'param_type': param_type,
                'default': default_value
            })
        
        app_panel["parameters"] = parameters
        
        return app_panel
    
    def _generate_run_sh(self) -> str:
        """
        Generate COMBINED run.sh with all logic.
        
        v8.0: Single file approach per Conor's recommendation:
        "run.sh is the only requirement to run a capsule"
        
        Includes:
        1. Copy shared format_values.py
        2. Initialize parameters with defaults
        3. Parse named arguments (--param_name=value)
        4. Create parameters JSON
        5. Normalize with format_values.py
        6. Run SPAC template
        """
        title = self.blueprint.get('title', 'SPAC Tool')
        tool_id = self._make_tool_id(title)
        template_module = f"{tool_id}_template"
        
        # Collect parameter info
        bool_params = []
        list_params = []
        numeric_params = []
        
        for param in self.blueprint.get('parameters', []):
            param_type = param.get('paramType', 'STRING')
            param_key = param.get('key')
            
            if param_type == 'BOOLEAN':
                bool_params.append(param_key)
            elif param_type == 'LIST':
                list_params.append(param_key)
            elif self._is_numeric_type(param_type):
                numeric_params.append(param_key)
        
        # Build script
        lines = [
            '#!/usr/bin/env bash',
            'set -euo pipefail',
            '',
            f'# {title}',
            '# Generated by code_ocean_synthesizer.py v8.0 (Single run.sh)',
            '# Named argument parsing: --param_name=value format',
            '',
            f'echo "=== {title} ==="',
            ''
        ]
        
        # Initialize variables with defaults
        lines.append('# Initialize parameters with default values')
        
        for param_info in self._parameter_order:
            key = param_info['key']
            default = param_info['default']
            param_type = param_info['param_type']
            
            # Format default value
            if default is None:
                default_str = ''
            else:
                default_str = str(default)
            
            # For numeric types, don't quote
            if self._is_numeric_type(param_type):
                if default_str:
                    lines.append(f'{key}={default_str}')
                else:
                    lines.append(f'{key}=0')
            else:
                lines.append(f'{key}="{default_str}"')
        
        # Dynamic named argument parsing (Conor's approach)
        lines.extend([
            '',
            '# Print received arguments for debugging',
            'echo "Number of arguments: $#"',
            'echo "Arguments received:"',
            'for arg in "$@"; do',
            '    echo "  $arg"',
            'done',
            '',
            '# Parse named arguments dynamically (--param_name=value format)',
            '# Based on Conor\'s example: single pattern handles all --key=value arguments',
            'for arg in "$@"; do',
            '    case "$arg" in',
            '        --*=*)',
            '            key="${arg%%=*}"      # everything before \'=\'',
            '            value="${arg#*=}"     # everything after \'=\'',
            '            key="${key#--}"       # remove leading \'--\'',
            '            export "$key=$value"',
            '            ;;',
            '    esac',
            'done',
            ''
        ])
        
        # Debug: print parsed values
        lines.append('# Debug: Print parsed parameter values')
        for param_info in self._parameter_order:
            key = param_info['key']
            lines.append(f'echo "{key}: ${key}"')
        
        # Find input data
        lines.extend([
            '',
            '# Find input data',
            'INPUT=$(find -L ../data -type f \\( -name "*.pickle" -o -name "*.pkl" -o -name "*.h5ad" \\) 2>/dev/null | head -n 1)',
            'if [ -z "$INPUT" ]; then echo "ERROR: No input file found in ../data"; exit 1; fi',
            'echo "Input: $INPUT"',
            ''
        ])
        
        # Create output directories
        lines.extend([
            '# Create output directories',
            'mkdir -p /results/figures /results/jsons',
            ''
        ])
        
        # Build JSON - numeric values WITHOUT quotes
        lines.append('# Create parameters JSON')
        lines.append('cat > /results/jsons/params.json << EOF')
        lines.append('{')
        
        # First add input path (always string)
        json_lines = ['    "Upstream_Analysis": "$INPUT"']
        
        # Add each parameter with proper quoting
        for param_info in self._parameter_order:
            key = param_info['key']
            param_type = param_info['param_type']
            
            # CRITICAL: Numeric types WITHOUT quotes around value
            if self._is_numeric_type(param_type):
                json_lines.append(f'    "{key}": ${key}')
            else:
                json_lines.append(f'    "{key}": "${key}"')
        
        # Join with commas
        lines.append(',\n'.join(json_lines))
        lines.append('}')
        lines.append('EOF')
        lines.append('')
        
        # Output configuration
        outputs = self.blueprint.get('outputs', {})
        if outputs:
            outputs_json = json.dumps(outputs)
            lines.append(f"echo '{outputs_json}' > /results/jsons/outputs_config.json")
            lines.append('')
        
        # Normalize parameters with format_values.py (in same code/ directory)
        lines.append('# Normalize parameters')
        format_cmd = 'python ./format_values.py /results/jsons/params.json /results/jsons/cleaned_params.json'
        
        if bool_params:
            format_cmd += f" --bool-values {' '.join(bool_params)}"
        if list_params:
            format_cmd += f" --list-values {' '.join(list_params)}"
        if outputs:
            format_cmd += " --inject-outputs --outputs-config /results/jsons/outputs_config.json"
        
        lines.append(format_cmd)
        lines.append('')
        lines.append('echo "Normalized: /results/jsons/params.json -> /results/jsons/cleaned_params.json"')
        lines.append('')
        
        # Run SPAC template
        lines.append('# Run SPAC template')
        template_cmd = f'python -c "from spac.templates.{template_module} import run_from_json; run_from_json(\'/results/jsons/cleaned_params.json\', output_dir=\'/results\')"'
        lines.append(template_cmd)
        lines.append('')
        lines.append('echo ""')
        lines.append('echo "=== Completed. Outputs in /results/ ==="')
        
        return '\n'.join(lines)


def _sanitize_filename(title: str) -> str:
    """Sanitize title to create a valid folder name."""
    clean_title = re.sub(r'\[.*?\]', '', title).strip()
    tool_name = clean_title.lower().replace(' ', '_')
    tool_name = tool_name.replace("'", "").replace('\\', '_').replace('/', '_')
    tool_name = re.sub(r'_+', '_', tool_name)
    tool_name = tool_name.strip('_')
    return tool_name


def generate_format_values_py() -> str:
    """
    Generate the shared format_values.py utility.
    This is placed at the root level and copied by each tool's run.sh.
    """
    return '''#!/usr/bin/env python3
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
    python format_values.py params.json cleaned_params.json \\
        --bool-values Horizontal_Plot Keep_Outliers \\
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
        print(f"\\nBool params: {args.bool_values}", file=sys.stderr)
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
        print("\\n=== Cleaned Parameters ===", file=sys.stderr)
        print(json.dumps(cleaned_params, indent=2), file=sys.stderr)
    
    # Write output JSON
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(cleaned_params, f, indent=2)


if __name__ == "__main__":
    main()
'''


def process_blueprint(blueprint_path: Path, output_dir: Path, docker_image: str = "nciccbr/spac:v2-dev"):
    """Process a single blueprint to generate Code Ocean capsule files."""
    print(f"Processing: {blueprint_path.name}")
    
    with open(blueprint_path, 'r') as f:
        blueprint = json.load(f)
    
    synthesizer = CodeOceanSynthesizer(blueprint, docker_image)
    files = synthesizer.synthesize()
    
    # Create tool directory
    title = blueprint.get('title', 'tool')
    tool_name = _sanitize_filename(title)
    tool_dir = output_dir / tool_name
    
    # Write each file
    for filename, content in files.items():
        file_path = tool_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        # Make shell scripts executable
        if filename.endswith('.sh'):
            file_path.chmod(0o755)
        
        print(f"  -> Generated: {tool_name}/{filename}")
    
    # Report features
    features = ["named arguments", "single run.sh"]  # v8.0 key features
    
    # Check for numeric params
    for param in blueprint.get('parameters', []):
        param_type = param.get('paramType', '')
        if param_type in CodeOceanSynthesizer.NUMERIC_TYPES:
            features.append("numeric params")
            break
    
    if blueprint.get('outputs'):
        features.append("output config")
    
    print(f"  -> Features: {', '.join(features)}")
    
    return tool_dir


def batch_process(input_pattern: str, output_dir: str, docker_image: str = "nciccbr/spac:v2-dev"):
    """Process multiple blueprint files matching a pattern."""
    input_path = Path(input_pattern)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        blueprint_files = [input_path]
    elif input_path.is_dir():
        blueprint_files = list(input_path.glob("template_json_*.json"))
    else:
        parent = input_path.parent
        pattern = input_path.name
        blueprint_files = list(parent.glob(pattern))
    
    if not blueprint_files:
        print(f"No blueprint files found matching: {input_pattern}")
        return 1
    
    print(f"Found {len(blueprint_files)} blueprint files to process")
    print(f"Using v8.0: Named parameters + Single run.sh")
    print("=" * 60)
    
    generated_dirs = []
    
    for blueprint_file in sorted(blueprint_files):
        try:
            tool_dir = process_blueprint(blueprint_file, output_path, docker_image)
            generated_dirs.append(tool_dir)
        except Exception as e:
            print(f"  ERROR processing {blueprint_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Successfully generated {len(generated_dirs)} Code Ocean capsule(s)")
    print("")
    print("Key changes in v8.0:")
    print("  - Single run.sh file (no separate main.sh)")
    print("  - Named parameters: --param_name=value format")
    print("  - format_values.py in same directory as run.sh")
    print("")
    print("Generated structure per capsule:")
    print("  tool_name/")
    print("  ├── .codeocean/")
    print("  │   └── app-panel.json")
    print("  ├── format_values.py")
    print("  └── run.sh")
    print("")
    print("Next steps:")
    print("1. Push each tool folder to a separate Git repository")
    print("2. In Code Ocean, use 'Copy from public Git' to import")
    print("3. The app-panel.json will auto-generate the UI!")
    
    return 0


def main():
    """Main entry point for the synthesizer."""
    parser = argparse.ArgumentParser(
        description="Generate Code Ocean capsule files from SPAC blueprint JSON (v8.0 - Single run.sh)"
    )
    parser.add_argument(
        "blueprint",
        help="Path to blueprint JSON file or pattern (e.g., 'template_json_*.json')"
    )
    parser.add_argument(
        "-o", "--output",
        default="code_ocean_tools",
        help="Output directory for capsule files (default: code_ocean_tools)"
    )
    parser.add_argument(
        "--docker",
        default="nciccbr/spac:v2-dev",
        help="Docker image name (default: nciccbr/spac:v2-dev)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    return batch_process(args.blueprint, args.output, args.docker)


if __name__ == "__main__":
    sys.exit(main())
