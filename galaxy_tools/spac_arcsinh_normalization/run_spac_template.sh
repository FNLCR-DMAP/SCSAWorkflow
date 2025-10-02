#!/usr/bin/env bash
# run_spac_template.sh - Docker version for Galaxy
# Version: 4.0.0 - Imports templates from installed SPAC package
set -euo pipefail

# Log everything to tool_stdout.txt
exec > >(tee -a tool_stdout.txt) 2>&1

PARAMS_JSON="${1:?Missing params.json path}"
TEMPLATE_NAME="${2:?Missing template name}"  # Just the name, not filename

# Use system Python inside Docker container
SPAC_PYTHON="${SPAC_PYTHON:-python3}"

echo "=== SPAC Template Wrapper v4.0 (Docker) ==="
echo "Parameters: $PARAMS_JSON"
echo "Template: $TEMPLATE_NAME"
echo "Python: $SPAC_PYTHON"
echo "Working directory: $(pwd)"

# Run template through Python interpreter
"$SPAC_PYTHON" - <<'PYTHON_RUNNER' "$PARAMS_JSON" "$TEMPLATE_NAME"
import json
import os
import sys
import copy
import importlib
import traceback
import inspect

# Get command line arguments
params_path = sys.argv[1]
template_name = sys.argv[2]  # Just the name like "boxplot", not "boxplot_template.py"

print(f"[Runner] Starting execution for template: {template_name}")
print(f"[Runner] Python version: {sys.version}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def determine_outputs_from_params(params):
    """Read outputs from params if available, otherwise use defaults"""
    if 'outputs' in params and isinstance(params['outputs'], dict):
        outputs = params['outputs']
        print(f"[Runner] Using outputs from params: {list(outputs.keys())}")
        return outputs
    
    # Fallback to defaults based on template name
    print(f"[Runner] No outputs in params, using defaults for {template_name}")
    
    if template_name in ['boxplot', 'histogram', 'violin_plot', 'scatter_plot',
                         'hierarchical_heatmap', 'relational_heatmap']:
        return {'figures': 'figure_folder', 'DataFrames': 'dataframe_folder'}
    elif template_name in ['interactive_spatial_plot', 'interactive_scatter_plot']:
        return {'html': 'html_folder'}
    elif template_name in ['analysis_to_csv', 'select_values']:
        return {'DataFrames': 'dataframe_folder'}
    elif template_name == 'setup_analysis':
        return {'analysis': 'analysis_output.pickle'}
    else:
        return {'analysis': 'transform_output.pickle'}

def inject_output_paths(params, outputs, template_name):
    """Add output paths to parameters"""
    params_exec = copy.deepcopy(params)
    
    # Remove the 'outputs' field before execution
    params_exec.pop('outputs', None)
    
    params_exec['save_results'] = True
    
    if 'analysis' in outputs:
        params_exec['output_path'] = outputs['analysis']
        params_exec['Output_Path'] = outputs['analysis']
        params_exec['Output_File'] = outputs['analysis']
    
    if 'DataFrames' in outputs:
        df_dir = outputs['DataFrames']
        params_exec['output_dir'] = df_dir
        params_exec['Export_Dir'] = df_dir
        params_exec['Output_File'] = os.path.join(df_dir, f'{template_name}_output.csv')
    
    if 'figures' in outputs:
        fig_dir = outputs['figures']
        params_exec['figure_dir'] = fig_dir
        params_exec['Figure_Dir'] = fig_dir
        params_exec['Figure_File'] = os.path.join(fig_dir, f'{template_name}.png')
    
    if 'html' in outputs:
        html_dir = outputs['html']
        params_exec['output_path'] = os.path.join(html_dir, f'{template_name}.html')
        params_exec['Output_File'] = params_exec['output_path']
    
    return params_exec

def handle_template_results(result, outputs, template_name):
    """Save any in-memory results returned by the template"""
    if result is None:
        return
    
    saved_count = 0
    
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if isinstance(result, tuple):
            for i, item in enumerate(result):
                if hasattr(item, 'savefig') and 'figures' in outputs:
                    fig_path = os.path.join(outputs['figures'], f'{template_name}_{i+1}.png')
                    item.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"[Runner] Saved figure to {fig_path}")
                    plt.close(item)
                    saved_count += 1
                elif hasattr(item, 'to_csv') and 'DataFrames' in outputs:
                    csv_path = os.path.join(outputs['DataFrames'], f'{template_name}_{i+1}.csv')
                    item.to_csv(csv_path, index=True)
                    print(f"[Runner] Saved DataFrame to {csv_path}")
                    saved_count += 1
        
        elif hasattr(result, 'to_csv') and 'DataFrames' in outputs:
            csv_path = os.path.join(outputs['DataFrames'], f'{template_name}_output.csv')
            result.to_csv(csv_path, index=True)
            print(f"[Runner] Saved DataFrame to {csv_path}")
            saved_count += 1
        
        elif hasattr(result, 'savefig') and 'figures' in outputs:
            fig_path = os.path.join(outputs['figures'], f'{template_name}.png')
            result.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"[Runner] Saved figure to {fig_path}")
            plt.close(result)
            saved_count += 1
        
        elif hasattr(result, 'write_h5ad') and 'analysis' in outputs:
            result.write_h5ad(outputs['analysis'])
            print(f"[Runner] Saved AnnData to {outputs['analysis']}")
            saved_count += 1
    
    except ImportError as e:
        print(f"[Runner] Note: Some libraries not available for result handling: {e}")
    
    if saved_count > 0:
        print(f"[Runner] Saved {saved_count} in-memory results")

def verify_outputs(outputs):
    """Verify that expected outputs were created"""
    found_outputs = False
    
    for output_type, path in outputs.items():
        if output_type == 'analysis':
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"[Runner] ✓ {output_type}: {os.path.basename(path)} ({size:,} bytes)")
                found_outputs = True
            else:
                print(f"[Runner] ✗ {output_type}: NOT FOUND at {path}")
        else:
            if os.path.exists(path) and os.path.isdir(path):
                files = os.listdir(path)
                if files:
                    total_size = sum(os.path.getsize(os.path.join(path, f)) for f in files)
                    print(f"[Runner] ✓ {output_type}: {len(files)} files ({total_size:,} bytes)")
                    for f in files[:3]:
                        size = os.path.getsize(os.path.join(path, f))
                        print(f"[Runner]   - {f} ({size:,} bytes)")
                    if len(files) > 3:
                        print(f"[Runner]   ... and {len(files)-3} more files")
                    found_outputs = True
                else:
                    print(f"[Runner] ⚠ {output_type}: directory exists but empty")
    
    if not found_outputs:
        print("[Runner] WARNING: No outputs were created!")

# ============================================================================
# PARAMETER PROCESSING
# ============================================================================

def _unsanitize(s: str) -> str:
    """Remove Galaxy's parameter sanitization tokens"""
    replacements = {
        '__ob__': '[', '__cb__': ']',
        '__oc__': '{', '__cc__': '}',
        '__dq__': '"', '__sq__': "'",
        '__gt__': '>', '__lt__': '<',
        '__cn__': '\n', '__cr__': '\r',
        '__tc__': '\t', '__pd__': '#',
        '__at__': '@', '__cm__': ',',
        '__dollar__': '$', '__us__': '_'
    }
    for token, char in replacements.items():
        s = s.replace(token, char)
    return s

def _maybe_parse(v):
    """Recursively unsanitize and parse JSON where possible"""
    if isinstance(v, str):
        u = _unsanitize(v).strip()
        if (u.startswith('[') and u.endswith(']')) or (u.startswith('{') and u.endswith('}')):
            try:
                return json.loads(u)
            except:
                return u
        return u
    elif isinstance(v, dict):
        return {k: _maybe_parse(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [_maybe_parse(item) for item in v]
    return v

def should_normalize_as_list(key, value):
    """Determine if a parameter should be normalized as a list"""
    if isinstance(value, list):
        return True
    
    if value is None or value == "":
        return False
    
    key_lower = key.lower()
    
    if 'regex' in key_lower or 'pattern' in key_lower:
        return False
    
    if any(x in key_lower for x in ['single', 'one', 'first', 'second', 'primary']):
        return False
    
    if any(x in key_lower for x in ['features', 'markers', 'phenotypes', 'annotations',
                                     'columns', 'types', 'labels', 'regions', 'radii']):
        return True
    
    if isinstance(value, str):
        if ',' in value or '\n' in value:
            return True
        if value.strip().startswith('[') and value.strip().endswith(']'):
            return True
    
    return False

def normalize_to_list(value):
    """Convert various input formats to a proper Python list"""
    if value in (None, "", "All", ["All"], "all", ["all"]):
        return ["All"]
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        s = value.strip()
        
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else [str(parsed)]
            except:
                pass
        
        if ',' in s:
            return [item.strip() for item in s.split(',') if item.strip()]
        
        if '\n' in s:
            return [item.strip() for item in s.split('\n') if item.strip()]
        
        return [s] if s else []
    
    return [value] if value is not None else []

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Load parameters
with open(params_path, 'r') as f:
    params = json.load(f)
    print(f"[Runner] Loaded {len(params)} parameters from {params_path}")

# Step 1: De-sanitize Galaxy parameters
print("[Runner] Step 1: De-sanitizing Galaxy parameters")
params = _maybe_parse(params)

# Step 2: Get outputs from params (injected by Galaxy)
print("[Runner] Step 2: Getting output structure")
outputs = determine_outputs_from_params(params)

# Create output directories
for output_type, path in outputs.items():
    if output_type != 'analysis' and path:
        os.makedirs(path, exist_ok=True)
        print(f"[Runner] Created {output_type} directory: {path}")

# Step 3: Normalize list parameters
print("[Runner] Step 3: Normalizing list parameters")
list_count = 0
for key, value in list(params.items()):
    if key != 'outputs' and should_normalize_as_list(key, value):
        original = value
        normalized = normalize_to_list(value)
        if original != normalized:
            params[key] = normalized
            list_count += 1
            print(f"[Runner] Normalized {key}: {original} -> {normalized}")

if list_count > 0:
    print(f"[Runner] Normalized {list_count} list parameters")

# Step 4: Inject output paths
print("[Runner] Step 4: Injecting output paths")
params_exec = inject_output_paths(params, outputs, template_name)

# Save parameter files
with open('params.exec.json', 'w') as f:
    json.dump(params_exec, f, indent=2)

with open('config_used.json', 'w') as f:
    params_display = {k: v for k, v in params.items()
                     if k != 'outputs' and 
                     not any(x in k.lower() for x in ['output', 'save', 'path', 'dir', 'file', 'export'])}
    json.dump(params_display, f, indent=2)

print(f"[Runner] Saved params.exec.json ({len(params_exec)} parameters)")
print(f"[Runner] Saved config_used.json ({len(params_display)} display parameters)")

# Step 5: Load template module from installed SPAC package
print(f"[Runner] Step 5: Loading template '{template_name}' from SPAC package")

try:
    # Import from spac.templates package
    module_name = f'spac.templates.{template_name}_template'
    mod = importlib.import_module(module_name)
    print(f"[Runner] Successfully imported {module_name}")
except ImportError as e:
    print(f"[Runner] ERROR: Could not import {module_name}: {e}")
    print(f"[Runner] Available modules in spac.templates:")
    try:
        import spac.templates
        import pkgutil
        for importer, modname, ispkg in pkgutil.iter_modules(spac.templates.__path__):
            print(f"[Runner]   - {modname}")
    except:
        pass
    sys.exit(1)

if not hasattr(mod, 'run_from_json'):
    print('[Runner] ERROR: Template missing run_from_json function')
    sys.exit(2)

# Step 6: Execute template
print("[Runner] Step 6: Executing template")
sig = inspect.signature(mod.run_from_json)
kwargs = {}

if 'save_results' in sig.parameters:
    kwargs['save_results'] = True
    print("[Runner] Added save_results=True to kwargs")

if 'show_plot' in sig.parameters:
    kwargs['show_plot'] = False
    print("[Runner] Added show_plot=False to kwargs")

try:
    print(f"[Runner] Calling run_from_json('params.exec.json', **{kwargs})")
    result = mod.run_from_json('params.exec.json', **kwargs)
    print(f"[Runner] Template completed successfully, returned {type(result).__name__}")
except Exception as e:
    print(f"[Runner] ERROR in template execution: {e}")
    print(f"[Runner] Error type: {type(e).__name__}")
    traceback.print_exc()
    sys.exit(1)

# Handle any returned objects
handle_template_results(result, outputs, template_name)

# Step 7: Verify outputs
print("[Runner] Step 7: Verifying outputs")
verify_outputs(outputs)

print("[Runner] === Execution completed successfully ===")

PYTHON_RUNNER

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=== Template execution failed with exit code $EXIT_CODE ==="
    echo ""
fi

exit $EXIT_CODE