#!/usr/bin/env bash
# run_spac_template.sh - SPAC wrapper with Load CSV fixes
# Version: 5.3.0 - Complete version with Load CSV handling
set -euo pipefail

PARAMS_JSON="${1:?Missing params.json path}"
TEMPLATE_BASE="${2:?Missing template base name}"

# Handle both base names and full .py filenames for backward compatibility
if [[ "$TEMPLATE_BASE" == *.py ]]; then
    TEMPLATE_PY="$TEMPLATE_BASE"
elif [[ "$TEMPLATE_BASE" == "load_csv_files_with_config" ]]; then
    TEMPLATE_PY="load_csv_files_with_config.py"
else
    TEMPLATE_PY="${TEMPLATE_BASE}_template.py"
fi

# Use SPAC Python environment
SPAC_PYTHON="${SPAC_PYTHON:-python3}"

echo "=== SPAC Template Wrapper v5.3 ==="
echo "Parameters: $PARAMS_JSON"
echo "Template base: $TEMPLATE_BASE"
echo "Template file: $TEMPLATE_PY"
echo "Python: $SPAC_PYTHON"

# Run template through Python
"$SPAC_PYTHON" - <<'PYTHON_RUNNER' "$PARAMS_JSON" "$TEMPLATE_PY" 2>&1 | tee tool_stdout.txt
import json
import os
import sys
import copy
import traceback
import inspect
import shutil

# Get arguments
params_path = sys.argv[1]
template_filename = sys.argv[2]

print(f"[Runner] Loading parameters from: {params_path}")
print(f"[Runner] Template: {template_filename}")

# Load parameters
with open(params_path, 'r') as f:
    params = json.load(f)

# ---------------------------------------------------------------------------
# De-sanitization and parsing helpers
# ---------------------------------------------------------------------------
def _unsanitize(s: str) -> str:
    """Remove Galaxy's parameter sanitization tokens"""
    if not isinstance(s, str):
        return s
    replacements = {
        '__ob__': '[', '__cb__': ']',
        '__oc__': '{', '__cc__': '}',
        '__dq__': '"', '__sq__': "'",
        '__gt__': '>', '__lt__': '<',
        '__cn__': '\n', '__cr__': '\r',
        '__tc__': '\t', '__pd__': '#',
        '__at__': '@', '__cm__': ','
    }
    for token, char in replacements.items():
        s = s.replace(token, char)
    return s

def _maybe_parse(v):
    """Recursively de-sanitize and JSON-parse strings where possible."""
    if isinstance(v, str):
        u = _unsanitize(v).strip()
        if (u.startswith('[') and u.endswith(']')) or (u.startswith('{') and u.endswith('}')):
            try:
                return json.loads(u)
            except Exception:
                return u
        return u
    elif isinstance(v, dict):
        return {k: _maybe_parse(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [_maybe_parse(item) for item in v]
    return v

# First normalize the whole params tree
params = _maybe_parse(params)

# Extract template name
template_name = os.path.basename(template_filename).replace('_template.py', '').replace('.py', '')

# Helper function to coerce singleton lists to strings for load_csv
def _coerce_singleton_paths_for_load_csv(params, template_name):
    """For load_csv templates, flatten 1-item lists to strings for path-like params."""
    if 'load_csv' not in template_name:
        return params
    for key in ('CSV_Files', 'CSV_Files_Configuration'):
        val = params.get(key)
        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], (str, bytes)):
            params[key] = val[0]
            print(f"[Runner] Coerced {key} from list -> string")
    return params

# Special handling for String_Columns in load_csv templates
if 'load_csv' in template_name and 'String_Columns' in params:
    value = params['String_Columns']
    if not isinstance(value, list):
        if value in [None, "", "[]", "__ob____cb__"]:
            params['String_Columns'] = []
        elif isinstance(value, str):
            s = value.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    params['String_Columns'] = json.loads(s)
                except:
                    params['String_Columns'] = [s] if s else []
            elif ',' in s:
                params['String_Columns'] = [item.strip() for item in s.split(',') if item.strip()]
            else:
                params['String_Columns'] = [s] if s else []
        else:
            params['String_Columns'] = []
    print(f"[Runner] Ensured String_Columns is list: {params['String_Columns']}")

# Apply the coercion for load_csv files
params = _coerce_singleton_paths_for_load_csv(params, template_name)

# CRITICAL FIX: For Load CSV Files, check if we have csv_input_dir
if 'load_csv' in template_name and 'CSV_Files' in params:
    # Check if csv_input_dir was created by Galaxy command
    if os.path.exists('csv_input_dir') and os.path.isdir('csv_input_dir'):
        params['CSV_Files'] = 'csv_input_dir'
        print("[Runner] Using csv_input_dir created by Galaxy")
    elif isinstance(params['CSV_Files'], str) and os.path.isfile(params['CSV_Files']):
        # We have a single file path, need to get its directory
        params['CSV_Files'] = os.path.dirname(params['CSV_Files'])
        print(f"[Runner] Using directory of CSV file: {params['CSV_Files']}")

# Extract outputs specification
raw_outputs = params.pop('outputs', {})
outputs = {}

if isinstance(raw_outputs, dict):
    outputs = raw_outputs
elif isinstance(raw_outputs, str):
    try:
        maybe = json.loads(_unsanitize(raw_outputs))
        if isinstance(maybe, dict):
            outputs = maybe
    except Exception:
        pass

if not isinstance(outputs, dict) or not outputs:
    print("[Runner] Warning: 'outputs' missing or not a dict; using defaults")
    if 'boxplot' in template_name or 'plot' in template_name or 'histogram' in template_name:
        outputs = {'DataFrames': 'dataframe_folder', 'figures': 'figure_folder'}
    elif 'load_csv' in template_name:
        outputs = {'DataFrames': 'dataframe_folder'}
    elif 'interactive' in template_name:
        outputs = {'html': 'html_folder'}
    else:
        outputs = {'analysis': 'transform_output.pickle'}

print(f"[Runner] Outputs -> {list(outputs.keys())}")

# Create output directories
for output_type, path in outputs.items():
    if output_type != 'analysis' and path:
        os.makedirs(path, exist_ok=True)
        print(f"[Runner] Created {output_type} directory: {path}")

# Normalize list parameters for features
feature_keys = [
    'Feature_s_to_Plot', 'Features_to_Plot', 'features',
    'Features', 'Phenotypes', 'Markers', 'Regions'
]

for key in feature_keys:
    if key in params:
        value = params[key]
        if value in (None, "", "All", ["All"], "all", ["all"]):
            params[key] = ["All"]
        elif isinstance(value, str):
            u = _unsanitize(value).strip()
            if u.startswith('[') and u.endswith(']'):
                try:
                    params[key] = json.loads(u)
                except:
                    if ',' in u:
                        params[key] = [s.strip() for s in u.split(',') if s.strip()]
                    else:
                        params[key] = [u] if u else []
            elif ',' in u:
                params[key] = [s.strip() for s in u.split(',') if s.strip()]
            elif '\n' in u:
                params[key] = [s.strip() for s in u.split('\n') if s.strip()]
            else:
                params[key] = [u] if u else []

# Add output paths to params
params['save_results'] = True

if 'analysis' in outputs:
    params['output_path'] = outputs['analysis']
    params['Output_Path'] = outputs['analysis']
    params['Output_File'] = outputs['analysis']

if 'DataFrames' in outputs:
    df_dir = outputs['DataFrames']
    params['output_dir'] = df_dir
    params['Export_Dir'] = df_dir
    params['Output_File'] = os.path.join(df_dir, f'{template_name}_output.csv')

if 'figures' in outputs:
    fig_dir = outputs['figures']
    params['figure_dir'] = fig_dir
    params['Figure_Dir'] = fig_dir
    params['Figure_File'] = os.path.join(fig_dir, f'{template_name}.png')

if 'html' in outputs:
    html_dir = outputs['html']
    params['html_dir'] = html_dir
    params['Output_File'] = os.path.join(html_dir, f'{template_name}.html')

# Save runtime parameters
with open('params.runtime.json', 'w') as f:
    json.dump(params, f, indent=2)

# Save clean params for Galaxy display
params_display = {k: v for k, v in params.items() 
                  if k not in ['Output_File', 'Figure_File', 'output_dir', 'figure_dir']}
with open('config_used.json', 'w') as f:
    json.dump(params_display, f, indent=2)

print(f"[Runner] Saved runtime parameters")

# ============================================================================
# LOAD AND EXECUTE TEMPLATE - CRITICAL: THIS MUST BE COMPLETE
# ============================================================================

# Try to import from installed package first (Docker environment)
template_module_name = template_filename.replace('.py', '')
try:
    import importlib
    mod = importlib.import_module(f'spac.templates.{template_module_name}')
    print(f"[Runner] Loaded template from package: spac.templates.{template_module_name}")
except (ImportError, ModuleNotFoundError):
    # Fallback to loading from file
    print(f"[Runner] Package import failed, trying file load")
    import importlib.util
    
    # Try standard locations
    template_paths = [
        f'/app/spac/templates/{template_filename}',
        f'/opt/spac/templates/{template_filename}',
        f'/opt/SCSAWorkflow/src/spac/templates/{template_filename}',
        template_filename  # Current directory
    ]
    
    spec = None
    for path in template_paths:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("template_mod", path)
            if spec:
                print(f"[Runner] Found template at: {path}")
                break
    
    if not spec or not spec.loader:
        print(f"[Runner] ERROR: Could not find template: {template_filename}")
        sys.exit(1)
    
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

# Verify run_from_json exists
if not hasattr(mod, 'run_from_json'):
    print('[Runner] ERROR: Template missing run_from_json function')
    sys.exit(2)

# Check function signature
sig = inspect.signature(mod.run_from_json)
kwargs = {}

if 'save_results' in sig.parameters:
    kwargs['save_results'] = True
if 'show_plot' in sig.parameters:
    kwargs['show_plot'] = False

print(f"[Runner] Executing template with kwargs: {kwargs}")

# Execute template
try:
    result = mod.run_from_json('params.runtime.json', **kwargs)
    print(f"[Runner] Template completed, returned: {type(result).__name__}")
    
    # Handle different return types
    if result is not None:
        if isinstance(result, dict):
            print(f"[Runner] Template saved files: {list(result.keys())}")
        elif isinstance(result, tuple):
            # Handle tuple returns
            saved_count = 0
            for i, item in enumerate(result):
                if hasattr(item, 'savefig') and 'figures' in outputs:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig_path = os.path.join(outputs['figures'], f'figure_{i+1}.png')
                    item.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(item)
                    saved_count += 1
                    print(f"[Runner] Saved figure to {fig_path}")
                elif hasattr(item, 'to_csv') and 'DataFrames' in outputs:
                    df_path = os.path.join(outputs['DataFrames'], f'table_{i+1}.csv')
                    item.to_csv(df_path, index=True)
                    saved_count += 1
                    print(f"[Runner] Saved DataFrame to {df_path}")
            
            if saved_count > 0:
                print(f"[Runner] Saved {saved_count} in-memory results")
        
        elif hasattr(result, 'to_csv') and 'DataFrames' in outputs:
            df_path = os.path.join(outputs['DataFrames'], 'output.csv')
            result.to_csv(df_path, index=True)
            print(f"[Runner] Saved DataFrame to {df_path}")
        
        elif hasattr(result, 'savefig') and 'figures' in outputs:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig_path = os.path.join(outputs['figures'], 'figure.png')
            result.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(result)
            print(f"[Runner] Saved figure to {fig_path}")
        
        elif hasattr(result, 'write_h5ad') and 'analysis' in outputs:
            result.write_h5ad(outputs['analysis'])
            print(f"[Runner] Saved AnnData to {outputs['analysis']}")
    
except Exception as e:
    print(f"[Runner] ERROR in template execution: {e}")
    traceback.print_exc()
    sys.exit(1)

# Verify outputs
print("[Runner] Verifying outputs...")
found_outputs = False

for output_type, path in outputs.items():
    if output_type == 'analysis':
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[Runner] ✔ {output_type}: {path} ({size:,} bytes)")
            found_outputs = True
        else:
            print(f"[Runner] ✗ {output_type}: NOT FOUND")
    else:
        if os.path.exists(path) and os.path.isdir(path):
            files = os.listdir(path)
            if files:
                print(f"[Runner] ✔ {output_type}: {len(files)} files")
                for f in files[:3]:
                    print(f"[Runner]   - {f}")
                if len(files) > 3:
                    print(f"[Runner]   ... and {len(files)-3} more")
                found_outputs = True
            else:
                print(f"[Runner] ⚠ {output_type}: directory empty")

# Check for files in working directory and move them
print("[Runner] Checking for files in working directory...")
for file in os.listdir('.'):
    if os.path.isdir(file) or file in ['params.runtime.json', 'config_used.json', 
                                         'tool_stdout.txt', 'outputs_returned.json']:
        continue
    
    if file.endswith('.csv') and 'DataFrames' in outputs:
        if not os.path.exists(os.path.join(outputs['DataFrames'], file)):
            target = os.path.join(outputs['DataFrames'], file)
            shutil.move(file, target)
            print(f"[Runner] Moved {file} to {target}")
            found_outputs = True
    elif file.endswith(('.png', '.pdf', '.jpg', '.svg')) and 'figures' in outputs:
        if not os.path.exists(os.path.join(outputs['figures'], file)):
            target = os.path.join(outputs['figures'], file)
            shutil.move(file, target)
            print(f"[Runner] Moved {file} to {target}")
            found_outputs = True

if found_outputs:
    print("[Runner] === SUCCESS ===")
else:
    print("[Runner] WARNING: No outputs created")

PYTHON_RUNNER

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Template execution failed with exit code $EXIT_CODE"
    exit 1
fi

echo "=== Execution Complete ==="
exit 0