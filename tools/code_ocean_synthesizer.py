#!/usr/bin/env python3
"""
Code Ocean Synthesizer - Generates Code Ocean capsule files from blueprint JSON files.

Version: 5.0 - Shared format_values.py Architecture
================================================================================

FOLDER STRUCTURE (Monorepo):
────────────────────────────────────────────────────────────────────────────────
code_ocean_tools/                    # Root - this is the git repo
├── format_values.py                 # SHARED - one copy for all tools
├── boxplot/
│   ├── .codeocean/
│   │   └── app-panel.json          # UI config
│   ├── run.sh                      # Entry point (copies shared file)
│   └── main.sh                     # Parameter parsing + template execution
├── histogram/
│   └── ...
└── ... (38 tools)
────────────────────────────────────────────────────────────────────────────────

HOW IT WORKS:
────────────────────────────────────────────────────────────────────────────────
1. run.sh copies ../format_values.py to working directory
2. main.sh uses local format_values.py
3. All tools share ONE format_values.py at the root

DEPLOYMENT:
────────────────────────────────────────────────────────────────────────────────
1. Generate:  python code_ocean_synthesizer.py blueprint_jsons/ -o code_ocean_tools/
2. Git push:  cd code_ocean_tools && git init && git add . && git push
3. Import:    Code Ocean → New Capsule → Copy from Git → select tool subfolder
────────────────────────────────────────────────────────────────────────────────

Author: FNLCR-DMAP Team
"""

import json
import re
import argparse
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import OrderedDict


# Shared format_values.py content - written once at root level
FORMAT_VALUES_CONTENT = '''#!/usr/bin/env python3
"""
format_values.py - Normalize parameters JSON for SPAC template consumption.

Shared utility for all Code Ocean SPAC capsules.
Location: code_ocean_tools/format_values.py (root level)

Transformations:
1. Convert boolean strings ("True", "False") to Python booleans
2. Convert comma-separated strings to lists for LIST parameters  
3. Inject output configuration for template_utils.save_results()

Usage:
    python format_values.py params.json cleaned_params.json \\
        --bool-values Horizontal_Plot Keep_Outliers \\
        --list-values Feature_s_to_Plot \\
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
'''


class CodeOceanSynthesizer:
    """
    Synthesize Code Ocean capsule files from blueprint JSON.
    
    Generates per capsule:
    - .codeocean/app-panel.json (UI configuration)
    - run.sh (entry point - copies shared format_values.py)
    - main.sh (parameter parsing + template execution)
    
    Shared at root level:
    - format_values.py (written once, used by all tools)
    """
    
    def __init__(self, blueprint: Dict[str, Any]):
        self.blueprint = blueprint
        
    def synthesize(self, output_dir: Path) -> Dict[str, Path]:
        """Generate capsule files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / ".codeocean").mkdir(exist_ok=True)
        
        files = {}
        
        # Generate app-panel.json
        app_panel_path = output_dir / ".codeocean" / "app-panel.json"
        with open(app_panel_path, 'w') as f:
            json.dump(self._generate_app_panel(), f, indent='\t')
        files["app_panel"] = app_panel_path
        
        # Generate run.sh (copies shared format_values.py)
        run_sh_path = output_dir / "run.sh"
        with open(run_sh_path, 'w') as f:
            f.write(self._generate_run_sh())
        run_sh_path.chmod(0o755)
        files["run_sh"] = run_sh_path
        
        # Generate main.sh
        main_sh_path = output_dir / "main.sh"
        with open(main_sh_path, 'w') as f:
            f.write(self._generate_main_sh())
        main_sh_path.chmod(0o755)
        files["main_sh"] = main_sh_path
        
        return files
    
    def _get_tool_name(self) -> str:
        """Generate tool name from blueprint title."""
        title = self.blueprint.get('title', 'tool')
        clean = re.sub(r'\[.*?\]', '', title).strip().lower().replace(' ', '_')
        clean = clean.replace("'", "").replace("-", "_")
        clean = re.sub(r'[^a-z0-9_]', '', clean)
        return re.sub(r'_+', '_', clean).strip('_')
    
    def _get_template_module(self) -> str:
        """Get template module name for spac.templates import."""
        return f"{self._get_tool_name()}_template"
    
    def _map_param_type(self, param_type: str, param_def: Dict) -> Dict[str, Any]:
        """Map blueprint paramType to Code Ocean app-panel type."""
        ptype = (param_type or 'STRING').upper()
        
        if ptype == 'BOOLEAN':
            return {'type': 'list', 'value_type': 'boolean', 'extra_data': []}
        if ptype == 'SELECT':
            opts = param_def.get('paramValues', [])
            return {'type': 'list', 'value_type': 'string', 
                    'extra_data': [str(o) for o in opts] if opts else []}
        if ptype in ('NUMBER', 'FLOAT', 'INTEGER', 'INT', 'INTERGER', 'Positive integer'):
            return {'type': 'text', 'value_type': 'number'}
        return {'type': 'text', 'value_type': 'string'}
    
    def _format_default(self, param_def: Dict) -> Optional[str]:
        """Format default value for app-panel.json."""
        default = param_def.get('defaultValue')
        ptype = (param_def.get('paramType') or 'STRING').upper()
        
        if default is None:
            return None
        if isinstance(default, list):
            return ', '.join(str(v) for v in default)
        if ptype == 'BOOLEAN':
            return 'True' if str(default).lower() in ('true', '1', 'yes') else 'False'
        return str(default)
    
    def _generate_app_panel(self) -> Dict[str, Any]:
        """Generate .codeocean/app-panel.json from blueprint."""
        title = self.blueprint.get('title', 'SPAC Tool')
        desc = self.blueprint.get('description', '')
        desc = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', desc).replace('\\n', ' ').strip()[:500]
        
        app_panel = {
            "version": 1,
            "named_parameters": False,
            "general": {"title": title, "instructions": desc},
            "datasets": [],
            "categories": [],
            "parameters": []
        }
        
        # Collect categories from paramGroup
        categories = OrderedDict()
        for p in self.blueprint.get('parameters', []):
            grp = p.get('paramGroup')
            if grp and grp not in categories:
                cat_id = re.sub(r'[^a-z0-9_]', '_', grp.lower()).strip('_')
                categories[grp] = {"id": cat_id, "name": grp, "help_text": f"Parameters for {grp}"}
        app_panel["categories"] = list(categories.values())
        
        # Process parameters
        ordered = self.blueprint.get('orderedMustacheKeys', [])
        pdict = {p['key']: p for p in self.blueprint.get('parameters', [])}
        if not ordered:
            ordered = list(pdict.keys())
        
        for key in ordered:
            if key not in pdict:
                continue
            pdef = pdict[key]
            tinfo = self._map_param_type(pdef.get('paramType'), pdef)
            
            grp = pdef.get('paramGroup')
            entry = {
                "id": uuid.uuid4().hex[:16],
                "name": pdef.get('displayName', key),
                "param_name": key,
                "description": pdef.get('description', ''),
                "type": tinfo['type'],
                "value_type": tinfo['value_type']
            }
            if grp and grp in categories:
                entry["category"] = categories[grp]['id']
            default = self._format_default(pdef)
            if default is not None:
                entry["default_value"] = default
            if 'extra_data' in tinfo:
                entry["extra_data"] = tinfo['extra_data']
            
            app_panel["parameters"].append(entry)
        
        return app_panel
    
    def _generate_run_sh(self) -> str:
        """Generate run.sh entry point that copies shared format_values.py."""
        return '''#!/usr/bin/env bash
set -ex

# Code Ocean Entry Point
# Copy shared format_values.py from parent directory
cp ../format_values.py . 2>/dev/null || true

# Run main script
bash main.sh "$@"
'''
    
    def _generate_main_sh(self) -> str:
        """Generate main.sh for parameter parsing and template execution."""
        tool_name = self._get_tool_name()
        template_module = self._get_template_module()
        title = self.blueprint.get('title', 'SPAC Tool')
        
        # Get parameters
        ordered = self.blueprint.get('orderedMustacheKeys', [])
        pdict = {p['key']: p for p in self.blueprint.get('parameters', [])}
        if not ordered:
            ordered = list(pdict.keys())
        param_keys = [k for k in ordered if k in pdict]
        
        # Collect bool/list params
        bool_params = [k for k in param_keys if (pdict[k].get('paramType') or '').upper() == 'BOOLEAN']
        list_params = [k for k in param_keys if (pdict[k].get('paramType') or '').upper() == 'LIST']
        
        lines = [
            '#!/usr/bin/env bash',
            'set -euo pipefail',
            '',
            f'# {title}',
            '# Generated by code_ocean_synthesizer.py v5.0',
            '',
            f'echo "=== {title} ==="',
            '',
            '# Capture App Panel arguments (positional: $1, $2, ...)',
        ]
        
        for i, key in enumerate(param_keys, 1):
            default = self._format_default(pdict[key]) or ''
            var = key.upper().replace(' ', '_')
            lines.append(f'{var}="${{{i}:-{default}}}"')
        
        lines.extend([
            '',
            '# Find input data',
            'INPUT=$(find -L ../data -type f \\( -name "*.pickle" -o -name "*.pkl" -o -name "*.h5ad" \\) 2>/dev/null | head -n 1)',
            'if [ -z "$INPUT" ]; then echo "ERROR: No input file found in ../data"; exit 1; fi',
            'echo "Input: $INPUT"',
            '',
            '# Create output directories',
            'mkdir -p /results/figures /results/jsons',
            '',
            '# Create parameters JSON',
            'cat > /results/jsons/params.json << EOF',
            '{',
            '    "Upstream_Analysis": "$INPUT",',
        ])
        
        for i, key in enumerate(param_keys):
            var = key.upper().replace(' ', '_')
            comma = ',' if i < len(param_keys) - 1 else ''
            lines.append(f'    "{key}": "${var}"{comma}')
        
        lines.extend([
            '}',
            'EOF',
            '',
        ])
        
        # Outputs config
        outputs = self.blueprint.get('outputs', {})
        if outputs:
            lines.append(f"echo '{json.dumps(outputs)}' > /results/jsons/outputs_config.json")
            lines.append('')
        
        # format_values command - uses local copy (copied by run.sh)
        fmt_cmd = 'python format_values.py /results/jsons/params.json /results/jsons/cleaned_params.json'
        if bool_params:
            fmt_cmd += f" --bool-values {' '.join(bool_params)}"
        if list_params:
            fmt_cmd += f" --list-values {' '.join(list_params)}"
        if outputs:
            fmt_cmd += " --inject-outputs --outputs-config /results/jsons/outputs_config.json"
        
        lines.extend([
            '# Normalize parameters',
            fmt_cmd,
            '',
            '# Run SPAC template',
            f'python -c "from spac.templates.{template_module} import run_from_json; run_from_json(\'/results/jsons/cleaned_params.json\')"',
            '',
            'echo ""',
            'echo "=== Completed. Outputs in /results/ ==="',
        ])
        
        return '\n'.join(lines) + '\n'


def _sanitize_name(title: str) -> str:
    """Sanitize title to folder name."""
    clean = re.sub(r'\[.*?\]', '', title).strip().lower().replace(' ', '_')
    clean = clean.replace("'", "").replace("-", "_")
    clean = re.sub(r'[^a-z0-9_]', '', clean)
    return re.sub(r'_+', '_', clean).strip('_')


def process_blueprint(bp_path: Path, out_dir: Path) -> Dict[str, Path]:
    """Process single blueprint."""
    print(f"Processing: {bp_path.name}")
    
    with open(bp_path) as f:
        blueprint = json.load(f)
    
    synth = CodeOceanSynthesizer(blueprint)
    name = _sanitize_name(blueprint.get('title', 'tool'))
    files = synth.synthesize(out_dir / name)
    
    print(f"  -> {name}/")
    
    return files


def batch_process(input_pat: str, output_dir: str) -> int:
    """Process multiple blueprints."""
    inp = Path(input_pat)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Write shared format_values.py at root ONCE
    format_values_path = out / "format_values.py"
    with open(format_values_path, 'w') as f:
        f.write(FORMAT_VALUES_CONTENT)
    format_values_path.chmod(0o755)
    print(f"Created: format_values.py (SHARED)")
    print("")
    
    if inp.is_file():
        bps = [inp]
    elif inp.is_dir():
        bps = sorted(inp.glob("template_json_*.json"))
    else:
        bps = sorted(inp.parent.glob(inp.name))
    
    if not bps:
        print(f"No blueprints found: {input_pat}")
        return 1
    
    print(f"Processing {len(bps)} blueprints...")
    print("=" * 50)
    
    ok = 0
    for bp in bps:
        try:
            process_blueprint(bp, out)
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("=" * 50)
    print(f"Generated {ok}/{len(bps)} capsules")
    print("")
    print("STRUCTURE:")
    print(f"  {out}/")
    print(f"  ├── format_values.py   <- SHARED (1 copy)")
    print(f"  ├── boxplot/")
    print(f"  │   ├── .codeocean/app-panel.json")
    print(f"  │   ├── run.sh")
    print(f"  │   └── main.sh")
    print(f"  └── ... ({ok} tools)")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate Code Ocean capsules from SPAC blueprints"
    )
    parser.add_argument("blueprint", help="Blueprint JSON file or directory")
    parser.add_argument("-o", "--output", default="code_ocean_tools", help="Output directory")
    args = parser.parse_args()
    return batch_process(args.blueprint, args.output)


if __name__ == "__main__":
    sys.exit(main())
