#!/usr/bin/env bash
# run_spac_template.sh - SPAC wrapper with column index conversion
# Version: 5.4.2 - Integrated column conversion
set -euo pipefail

PARAMS_JSON="${1:?Missing params.json path}"
TEMPLATE_BASE="${2:?Missing template base name}"

# Handle both base names and full .py filenames
if [[ "$TEMPLATE_BASE" == *.py ]]; then
    TEMPLATE_PY="$TEMPLATE_BASE"
elif [[ "$TEMPLATE_BASE" == "load_csv_files_with_config" ]]; then
    TEMPLATE_PY="load_csv_files_with_config.py"
else
    TEMPLATE_PY="${TEMPLATE_BASE}_template.py"
fi

# Use SPAC Python environment
SPAC_PYTHON="${SPAC_PYTHON:-python3}"

echo "=== SPAC Template Wrapper v5.4 ==="
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
import re
import csv

# Get arguments
params_path = sys.argv[1]
template_filename = sys.argv[2]

print(f"[Runner] Loading parameters from: {params_path}")
print(f"[Runner] Template: {template_filename}")

# Load parameters
with open(params_path, 'r') as f:
    params = json.load(f)

# Extract template name
template_name = os.path.basename(template_filename).replace('_template.py', '').replace('.py', '')

# ===========================================================================
# DE-SANITIZATION AND PARSING
# ===========================================================================
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

# Normalize the whole params tree
params = _maybe_parse(params)

# ===========================================================================
# COLUMN INDEX CONVERSION - CRITICAL FOR SETUP ANALYSIS
# ===========================================================================
def should_skip_column_conversion(template_name):
    """Some templates don't need column index conversion"""
    return 'load_csv' in template_name

def read_file_headers(filepath):
    """Read column headers from various file formats"""
    try:
        import pandas as pd
        
        # Try pandas auto-detect
        try:
            df = pd.read_csv(filepath, nrows=1)
            if len(df.columns) > 1 or not df.columns[0].startswith('Unnamed'):
                columns = df.columns.tolist()
                print(f"[Runner] Pandas auto-detected delimiter, found {len(columns)} columns")
                return columns
        except:
            pass
        
        # Try common delimiters
        for sep in ['\t', ',', ';', '|', ' ']:
            try:
                df = pd.read_csv(filepath, sep=sep, nrows=1)
                if len(df.columns) > 1:
                    columns = df.columns.tolist()
                    sep_name = {'\t': 'tab', ',': 'comma', ';': 'semicolon', 
                               '|': 'pipe', ' ': 'space'}.get(sep, sep)
                    print(f"[Runner] Pandas found {sep_name}-delimited file with {len(columns)} columns")
                    return columns
            except:
                continue
    except ImportError:
        print("[Runner] pandas not available, using csv fallback")
    
    # CSV module fallback
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
            sample = f.read(8192)
            f.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters='\t,;| ')
                reader = csv.reader(f, dialect)
                header = next(reader)
                columns = [h.strip().strip('"') for h in header if h.strip()]
                if columns:
                    print(f"[Runner] csv.Sniffer detected {len(columns)} columns")
                    return columns
            except:
                f.seek(0)
                first_line = f.readline().strip()
                for sep in ['\t', ',', ';', '|']:
                    if sep in first_line:
                        columns = [h.strip().strip('"') for h in first_line.split(sep)]
                        if len(columns) > 1:
                            print(f"[Runner] Manual parsing found {len(columns)} columns")
                            return columns
    except Exception as e:
        print(f"[Runner] Failed to read headers: {e}")
    
    return None

def should_convert_param(key, value):
    """Check if parameter contains column indices"""
    if value is None or value == "" or value == [] or value == {}:
        return False
    
    key_lower = key.lower()
    
    # Skip String_Columns - it's names not indices
    if key == 'String_Columns':
        return False
    
    # Skip output/path parameters
    if any(x in key_lower for x in ['output', 'path', 'file', 'directory', 'save', 'export']):
        return False
    
    # Skip regex/pattern parameters (but we'll handle Feature_Regex specially)
    if 'regex' in key_lower or 'pattern' in key_lower:
        return False
    
    # Parameters with 'column' likely have indices
    if 'column' in key_lower or '_col' in key_lower:
        return True
    
    # Known index parameters
    if key in {'Annotation_s_', 'Features_to_Analyze', 'Features', 'Markers', 'Markers_to_Plot', 'Phenotypes'}:
        return True
    
    # Check if values look like indices
    if isinstance(value, list):
        return all(isinstance(v, int) or (isinstance(v, str) and v.strip().isdigit()) for v in value if v)
    elif isinstance(value, (int, str)):
        return isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit())
    
    return False

def convert_single_index(item, columns):
    """Convert a single column index to name"""
    if isinstance(item, str) and not item.strip().isdigit():
        return item
    
    try:
        if isinstance(item, str):
            item = int(item.strip())
        elif isinstance(item, float):
            item = int(item)
    except (ValueError, AttributeError):
        return item
    
    if isinstance(item, int):
        idx = item - 1  # Galaxy uses 1-based indexing
        if 0 <= idx < len(columns):
            return columns[idx]
        elif 0 <= item < len(columns):  # Fallback for 0-based
            print(f"[Runner] Note: Found 0-based index {item}, converting to {columns[item]}")
            return columns[item]
        else:
            print(f"[Runner] Warning: Index {item} out of range (have {len(columns)} columns)")
    
    return item

def convert_column_indices_to_names(params, template_name):
    """Convert column indices to names for templates that need it"""
    
    if should_skip_column_conversion(template_name):
        print(f"[Runner] Skipping column conversion for {template_name}")
        return params
    
    print(f"[Runner] Checking for column index conversion (template: {template_name})")
    
    # Find input file
    input_file = None
    input_keys = ['Upstream_Dataset', 'Upstream_Analysis', 'CSV_Files', 
                  'Input_File', 'Input_Dataset', 'Data_File']
    
    for key in input_keys:
        if key in params:
            value = params[key]
            if isinstance(value, list) and value:
                value = value[0]
            if value and os.path.exists(str(value)):
                input_file = str(value)
                print(f"[Runner] Found input file via {key}: {os.path.basename(input_file)}")
                break
    
    if not input_file:
        print("[Runner] No input file found for column conversion")
        return params
    
    # Read headers
    columns = read_file_headers(input_file)
    if not columns:
        print("[Runner] Could not read column headers, skipping conversion")
        return params
    
    print(f"[Runner] Successfully read {len(columns)} columns")
    if len(columns) <= 10:
        print(f"[Runner] Columns: {columns}")
    else:
        print(f"[Runner] First 10 columns: {columns[:10]}")
    
    # Convert indices to names
    converted_count = 0
    for key, value in params.items():
        # Skip non-column parameters
        if not should_convert_param(key, value):
            continue
        
        # Convert indices
        if isinstance(value, list):
            converted_items = []
            for item in value:
                converted = convert_single_index(item, columns)
                if converted is not None:
                    converted_items.append(converted)
            converted_value = converted_items
        else:
            converted_value = convert_single_index(value, columns)
        
        if value != converted_value:
            params[key] = converted_value
            converted_count += 1
            print(f"[Runner] Converted {key}: {value} -> {converted_value}")
    
    if converted_count > 0:
        print(f"[Runner] Total conversions: {converted_count} parameters")
    
    # CRITICAL: Handle Feature_Regex specially
    if 'Feature_Regex' in params:
        value = params['Feature_Regex']
        if value in [[], [""], "__ob____cb__", "[]", "", None]:
            params['Feature_Regex'] = ""
            print("[Runner] Cleared empty Feature_Regex parameter")
        elif isinstance(value, list) and value:
            params['Feature_Regex'] = "|".join(str(v) for v in value if v)
            print(f"[Runner] Joined Feature_Regex list: {params['Feature_Regex']}")
    
    return params

# ===========================================================================
# APPLY COLUMN CONVERSION
# ===========================================================================
print("[Runner] Step 1: Converting column indices to names")
params = convert_column_indices_to_names(params, template_name)

# ===========================================================================
# SPECIAL HANDLING FOR SPECIFIC TEMPLATES
# ===========================================================================

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

# Apply coercion for load_csv files
params = _coerce_singleton_paths_for_load_csv(params, template_name)

# Fix for Load CSV Files directory
if 'load_csv' in template_name and 'CSV_Files' in params:
    # Check if csv_input_dir was created by Galaxy command
    if os.path.exists('csv_input_dir') and os.path.isdir('csv_input_dir'):
        params['CSV_Files'] = 'csv_input_dir'
        print("[Runner] Using csv_input_dir created by Galaxy")
    elif isinstance(params['CSV_Files'], str) and os.path.isfile(params['CSV_Files']):
        # We have a single file path, need to get its directory
        params['CSV_Files'] = os.path.dirname(params['CSV_Files'])
        print(f"[Runner] Using directory of CSV file: {params['CSV_Files']}")

# ===========================================================================
# LIST PARAMETER NORMALIZATION 
# ===========================================================================
def should_normalize_as_list(key, value):
    """Determine if a parameter should be normalized as a list"""
    # CRITICAL: Skip outputs and other non-list parameters
    key_lower = key.lower()
    if key_lower in {'outputs', 'output', 'upstream_analysis', 'upstream_dataset', 
                      'table_to_visualize', 'figure_title', 'figure_width', 
                      'figure_height', 'figure_dpi', 'font_size'}:
        return False

    # Already a proper list?
    if isinstance(value, list):
        # Only re-process if it's a single JSON string that needs parsing
        if len(value) == 1 and isinstance(value[0], str):
            s = value[0].strip()
            return s.startswith('[') and s.endswith(']')
        return False

    # Nothing to normalize
    if value is None or value == "":
        return False

    # CRITICAL: Explicitly mark Feature_s_to_Plot as a list parameter
    if key == 'Feature_s_to_Plot' or key_lower == 'feature_s_to_plot':
        return True
    
    # Other explicit list parameters
    explicit_list_keys = {
        'features_to_analyze', 'features', 'markers', 'markers_to_plot', 
        'phenotypes', 'labels', 'annotation_s_', 'string_columns'
    }
    if key_lower in explicit_list_keys:
        return True
    
    # Skip regex parameters
    if 'regex' in key_lower or 'pattern' in key_lower:
        return False
    
    # Skip known single-value parameters
    if any(x in key_lower for x in ['single', 'one', 'first', 'second', 'primary']):
        return False
    
    # Plural forms suggest lists
    if any(x in key_lower for x in [
        'features', 'markers', 'phenotypes', 'annotations',
        'columns', 'types', 'labels', 'regions', 'radii'
    ]):
        return True
    
    # List-like syntax in string values
    if isinstance(value, str):
        s = value.strip()
        if s.startswith('[') and s.endswith(']'):
            return True
        # Only treat comma/newline as list separator if not in outputs-like params
        if 'output' not in key_lower and 'path' not in key_lower:
            if ',' in s or '\n' in s:
                return True
    
    return False

def normalize_to_list(value):
    """Convert various input formats to a proper Python list"""
    # Handle special "All" cases first
    if value in (None, "", "All", "all"):
        return ["All"]

    # If it's already a list
    if isinstance(value, list):
        # Check for already-correct lists
        if value == ["All"] or value == ["all"]:
            return ["All"]
        
        # Check if it's a single-element list with a JSON string
        if len(value) == 1 and isinstance(value[0], str):
            s = value[0].strip()
            # If the single element looks like JSON
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return parsed
                except:
                    pass
            # If single element is "All" or "all"
            elif s.lower() == "all":
                return ["All"]
        
        # Already a proper list, return as-is
        return value
    
    if isinstance(value, str):
        s = value.strip()

        # Check for "All" string
        if s.lower() == "all":
            return ["All"]
        
        # Try JSON parsing
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else [str(parsed)]
            except:
                pass
        
        # Split by comma
        if ',' in s:
            return [item.strip() for item in s.split(',') if item.strip()]
        
        # Split by newline
        if '\n' in s:
            return [item.strip() for item in s.split('\n') if item.strip()]
        
        # Single value
        return [s] if s else []
    
    return [value] if value is not None else []

# Normalize list parameters
print("[Runner] Step 2: Normalizing list parameters")
list_count = 0
for key, value in list(params.items()):
    if should_normalize_as_list(key, value):
        original = value
        normalized = normalize_to_list(value)
        if original != normalized:
            params[key] = normalized
            list_count += 1
            if len(str(normalized)) > 100:
                print(f"[Runner] Normalized {key}: {type(original).__name__} -> list of {len(normalized)} items")
            else:
                print(f"[Runner] Normalized {key}: {original} -> {normalized}")

if list_count > 0:
    print(f"[Runner] Normalized {list_count} list parameters")

# CRITICAL FIX: Handle single-element lists for coordinate columns
# These should be strings, not lists
coordinate_keys = ['X_Coordinate_Column', 'Y_Coordinate_Column', 'X_centroid', 'Y_centroid']
for key in coordinate_keys:
    if key in params:
        value = params[key]
        if isinstance(value, list) and len(value) == 1:
            params[key] = value[0]
            print(f"[Runner] Extracted single value from {key}: {value} -> {params[key]}")

# Also check for any key ending with '_Column' that has a single-element list
for key in list(params.keys()):
    if key.endswith('_Column') and isinstance(params[key], list) and len(params[key]) == 1:
        original = params[key]
        params[key] = params[key][0]
        print(f"[Runner] Extracted single value from {key}: {original} -> {params[key]}")

# ===========================================================================
# OUTPUTS HANDLING
# ===========================================================================

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
        
# CRITICAL FIX: Handle outputs if it was mistakenly normalized as a list
if isinstance(raw_outputs, list) and raw_outputs:
    # Try to reconstruct the dict from the list
    if len(raw_outputs) >= 2:
        # Assume format like ["{'DataFrames': 'dataframe_folder'", "'figures': 'figure_folder'}"]
        combined = ''.join(str(item) for item in raw_outputs)
        # Clean up the string
        combined = combined.replace("'", '"')
        try:
            outputs = json.loads(combined)
        except:
            # Try another approach - look for dict-like patterns
            try:
                dict_str = '{' + combined.split('{')[1].split('}')[0] + '}'
                outputs = json.loads(dict_str.replace("'", '"'))
            except:
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
# LOAD AND EXECUTE TEMPLATE
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
    
    # Standard locations
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
    print(f"[Runner] Error type: {type(e).__name__}")
    traceback.print_exc()
    
    # Debug help for common issues
    if "String Columns must be a *list*" in str(e):
        print("\n[Runner] DEBUG: String_Columns validation failed")
        print(f"[Runner] Current String_Columns value: {params.get('String_Columns')}")
        print(f"[Runner] Type: {type(params.get('String_Columns'))}")
    
    elif "regex pattern" in str(e).lower() or "^8$" in str(e):
        print("\n[Runner] DEBUG: This appears to be a column index issue")
        print("[Runner] Check that column indices were properly converted to names")
        print("[Runner] Current Features_to_Analyze value:", params.get('Features_to_Analyze'))
        print("[Runner] Current Feature_Regex value:", params.get('Feature_Regex'))
    
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