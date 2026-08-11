#!/usr/bin/env python3
"""Universal utilities for Galaxy-SPAC bridge - Refactored Version"""
import json
import os
import sys
import importlib
import importlib.util
import traceback
import shutil
import csv
import re
from pathlib import Path

def desanitize_galaxy_params(s):
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

def parse_json_strings(v):
    """Recursively parse JSON strings in parameters"""
    if isinstance(v, str):
        u = desanitize_galaxy_params(v).strip()
        if (u.startswith('[') and u.endswith(']')) or \
           (u.startswith('{') and u.endswith('}')):
            try:
                return json.loads(u)
            except:
                return u
        return u
    elif isinstance(v, dict):
        return {k: parse_json_strings(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [parse_json_strings(item) for item in v]
    return v

def read_file_headers(filepath):
    """Read column headers from CSV/TSV files"""
    try:
        import pandas as pd
        # Try pandas auto-detect
        df = pd.read_csv(filepath, nrows=1)
        if len(df.columns) > 1 or not df.columns[0].startswith('Unnamed'):
            return df.columns.tolist()
    except:
        pass
    
    # CSV module fallback
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters='\t,;| ')
                reader = csv.reader(f, dialect)
                header = next(reader)
                return [h.strip().strip('"') for h in header if h.strip()]
            except:
                # Manual parsing
                f.seek(0)
                first_line = f.readline().strip()
                for sep in ['\t', ',', ';', '|']:
                    if sep in first_line:
                        columns = [h.strip().strip('"') for h in first_line.split(sep)]
                        if len(columns) > 1:
                            return columns
    except Exception as e:
        print(f"[Bridge] Failed to read headers: {e}")
    return None

def convert_column_indices(params, column_params, input_file_key='Upstream_Analysis'):
    """Convert column indices to names for specified parameters"""
    if input_file_key not in params:
        return params
    
    input_file = params[input_file_key]
    if isinstance(input_file, list):
        input_file = input_file[0] if input_file else None
    
    if not input_file or not os.path.exists(str(input_file)):
        return params
    
    columns = read_file_headers(str(input_file))
    if not columns:
        return params
    
    print(f"[Bridge] Read {len(columns)} columns from input file")
    
    for key in column_params:
        if key not in params:
            continue
        value = params[key]
        
        if isinstance(value, list):
            converted = []
            for item in value:
                if isinstance(item, (int, str)) and str(item).isdigit():
                    idx = int(item) - 1  # Galaxy uses 1-based indexing
                    if 0 <= idx < len(columns):
                        converted.append(columns[idx])
                    else:
                        converted.append(item)
                else:
                    converted.append(item)
            if converted != value:
                params[key] = converted
                print(f"[Bridge] Converted {key}: {value} -> {converted}")
        
        elif isinstance(value, (int, str)) and str(value).isdigit():
            idx = int(value) - 1
            if 0 <= idx < len(columns):
                params[key] = columns[idx]
                print(f"[Bridge] Converted {key}: {value} -> {params[key]}")
    
    return params

def normalize_list_params(params, list_keys):
    """Normalize specified parameters to lists"""
    for key in list_keys:
        if key not in params:
            continue
        
        value = params[key]
        if value in (None, "", "All", ["All"]):
            params[key] = ["All"]
        elif not isinstance(value, list):
            if isinstance(value, str):
                s = value.strip()
                # Try JSON parsing
                if s.startswith('[') and s.endswith(']'):
                    try:
                        parsed = json.loads(s)
                        params[key] = parsed if isinstance(parsed, list) else [s]
                    except:
                        params[key] = [s] if s else []
                elif ',' in s:
                    params[key] = [v.strip() for v in s.split(',') if v.strip()]
                elif '\n' in s:
                    params[key] = [v.strip() for v in s.split('\n') if v.strip()]
                else:
                    params[key] = [s] if s else []
            else:
                params[key] = [value] if value else []
    
    return params

def save_in_memory_results(result, output_dirs, template_name):
    """Save in-memory results from templates"""
    saved_count = 0
    
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if isinstance(result, tuple):
            for i, item in enumerate(result):
                if hasattr(item, 'savefig') and 'figures' in output_dirs:
                    fig_path = os.path.join(output_dirs['figures'], f'figure_{i+1}.png')
                    item.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close(item)
                    saved_count += 1
                    print(f"[Bridge] Saved figure to {fig_path}")
                elif hasattr(item, 'to_csv') and 'DataFrames' in output_dirs:
                    csv_path = os.path.join(output_dirs['DataFrames'], f'table_{i+1}.csv')
                    item.to_csv(csv_path, index=True)
                    saved_count += 1
                    print(f"[Bridge] Saved DataFrame to {csv_path}")
        
        elif hasattr(result, 'to_csv') and 'DataFrames' in output_dirs:
            csv_path = os.path.join(output_dirs['DataFrames'], f'{template_name}_output.csv')
            result.to_csv(csv_path, index=True)
            saved_count += 1
        
        elif hasattr(result, 'savefig') and 'figures' in output_dirs:
            fig_path = os.path.join(output_dirs['figures'], f'{template_name}.png')
            result.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(result)
            saved_count += 1
        
        elif hasattr(result, 'write_h5ad') and 'analysis' in output_dirs:
            result.write_h5ad(output_dirs['analysis'])
            saved_count += 1
            
    except ImportError as e:
        print(f"[Bridge] Note: Some libraries not available: {e}")
    
    return saved_count

def collect_orphan_files(output_dirs):
    """Move files created in working directory to output folders"""
    moved_count = 0
    
    for file in os.listdir('.'):
        if os.path.isdir(file):
            continue
        if file in ['params.json', 'params.runtime.json', 'config_used.json', 
                    'tool_stdout.txt', 'outputs_returned.json']:
            continue
        
        if file.endswith('.csv') and 'DataFrames' in output_dirs:
            target = os.path.join(output_dirs['DataFrames'], file)
            if not os.path.exists(target):
                shutil.move(file, target)
                moved_count += 1
                print(f"[Bridge] Moved {file} to {target}")
        elif file.endswith(('.png', '.pdf', '.jpg', '.svg')) and 'figures' in output_dirs:
            target = os.path.join(output_dirs['figures'], file)
            if not os.path.exists(target):
                shutil.move(file, target)
                moved_count += 1
                print(f"[Bridge] Moved {file} to {target}")
    
    return moved_count

def load_tool_config(template_name):
    """Load tool-specific configuration if exists"""
    config = None
    try:
        # Try to import from tool_configs directory
        config_module = importlib.import_module(f'tool_configs.{template_name}_config')
        config = config_module.TOOL_CONFIG
        print(f"[Bridge] Loaded config for {template_name}")
    except ImportError:
        # No specific config, use defaults
        config = {}
    
    # Set defaults if not specified
    if 'outputs' not in config:
        # Determine outputs based on template name
        if any(x in template_name for x in ['boxplot', 'histogram', 'scatter', 'heatmap', 'violin']):
            config['outputs'] = {'DataFrames': 'dataframe_folder', 'figures': 'figure_folder'}
        elif 'interactive' in template_name:
            config['outputs'] = {'html': 'html_folder'}
        elif any(x in template_name for x in ['csv', 'dataframe', 'select', 'append']):
            config['outputs'] = {'DataFrames': 'dataframe_folder'}
        else:
            config['outputs'] = {'analysis': 'transform_output.pickle'}
    
    if 'list_params' not in config:
        # Common list parameters
        config['list_params'] = []
        common_lists = ['Features', 'Markers', 'Phenotypes', 'Annotations', 
                       'Features_to_Plot', 'Features_to_Analyze', 'Cell_Types']
        config['list_params'] = common_lists
    
    if 'column_params' not in config:
        # Parameters that might contain column indices
        config['column_params'] = []
        if any(x in template_name for x in ['setup_analysis', 'calculate_centroid']):
            config['column_params'] = ['X_Coordinate_Column', 'Y_Coordinate_Column', 
                                       'Features_to_Analyze', 'Annotation_s_']
    
    return config

def load_template_module(template_name):
    """Load the SPAC template module"""
    # Determine actual filename
    if template_name == 'load_csv_files':
        template_file = 'load_csv_files_with_config.py'
        module_name = 'load_csv_files_with_config'
    else:
        template_file = f'{template_name}_template.py'
        module_name = f'{template_name}_template'
    
    # Try package import first (Docker environment)
    try:
        mod = importlib.import_module(f'spac.templates.{module_name}')
        print(f"[Bridge] Loaded from package: spac.templates.{module_name}")
        return mod
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Try loading from standard locations
    template_paths = [
        f'/app/spac/templates/{template_file}',
        f'/opt/spac/templates/{template_file}',
        f'/opt/SCSAWorkflow/src/spac/templates/{template_file}',
        template_file  # Current directory
    ]
    
    for path in template_paths:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("template", path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                print(f"[Bridge] Loaded from file: {path}")
                return mod
    
    raise ImportError(f"Cannot find template: {template_file}")

def main():
    """Main entry point for Galaxy-SPAC bridge"""
    if len(sys.argv) != 3:
        print("Usage: tool_utils.py <params.json> <template_name>")
        sys.exit(1)
    
    params_path = sys.argv[1]
    template_name = sys.argv[2]
    
    print(f"[Bridge] SPAC Galaxy Bridge v2.0")
    print(f"[Bridge] Template: {template_name}")
    print(f"[Bridge] Parameters: {params_path}")
    
    # Load parameters
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Desanitize Galaxy parameters
    params = parse_json_strings(params)
    
    # Extract outputs specification from hidden parameter
    raw_outputs = params.pop('outputs', {})
    if isinstance(raw_outputs, str):
        try:
            raw_outputs = json.loads(desanitize_galaxy_params(raw_outputs))
        except:
            raw_outputs = {}
    
    # Load tool configuration
    config = load_tool_config(template_name)
    
    # Override outputs if specified in params
    if raw_outputs:
        config['outputs'] = raw_outputs
    
    outputs = config.get('outputs', {})
    print(f"[Bridge] Outputs: {list(outputs.keys())}")
    
    # Apply preprocessing if defined
    if 'preprocess' in config:
        params = config['preprocess'](params)
        print("[Bridge] Applied tool-specific preprocessing")
    
    # Special handling for Load CSV Files
    if template_name == 'load_csv_files':
        # Ensure String_Columns is a list
        if 'String_Columns' in params:
            value = params['String_Columns']
            if not isinstance(value, list):
                if value in [None, "", "[]"]:
                    params['String_Columns'] = []
                elif isinstance(value, str):
                    if ',' in value:
                        params['String_Columns'] = [s.strip() for s in value.split(',')]
                    else:
                        params['String_Columns'] = [value] if value else []
        
        # Handle CSV_Files directory
        if 'CSV_Files' in params:
            if os.path.exists('csv_input_dir') and os.path.isdir('csv_input_dir'):
                params['CSV_Files'] = 'csv_input_dir'
                print("[Bridge] Using csv_input_dir created by Galaxy")
            elif isinstance(params['CSV_Files'], list) and len(params['CSV_Files']) == 1:
                params['CSV_Files'] = params['CSV_Files'][0]
    
    # Convert column indices to names
    if config.get('column_params'):
        input_key = config.get('input_key', 'Upstream_Analysis')
        if 'load_csv' in template_name:
            input_key = 'CSV_Files'
        params = convert_column_indices(params, config['column_params'], input_key)
    
    # Normalize list parameters
    if config.get('list_params'):
        params = normalize_list_params(params, config['list_params'])
    
    # Handle single-element lists for coordinate columns
    for key in ['X_Coordinate_Column', 'Y_Coordinate_Column', 'X_centroid', 'Y_centroid']:
        if key in params and isinstance(params[key], list) and len(params[key]) == 1:
            params[key] = params[key][0]
            print(f"[Bridge] Extracted single value from {key}")
    
    # Create output directories
    for output_type, dirname in outputs.items():
        if output_type != 'analysis' and dirname:
            os.makedirs(dirname, exist_ok=True)
            print(f"[Bridge] Created {output_type} directory: {dirname}")
    
    # Add output paths to params
    params['save_results'] = True
    
    if 'analysis' in outputs:
        params['output_path'] = outputs['analysis']
        params['Output_Path'] = outputs['analysis']
        params['Output_File'] = outputs['analysis']
    
    if 'DataFrames' in outputs:
        params['output_dir'] = outputs['DataFrames']
        params['Export_Dir'] = outputs['DataFrames']
        params['Output_File'] = os.path.join(outputs['DataFrames'], f'{template_name}_output.csv')
    
    if 'figures' in outputs:
        params['figure_dir'] = outputs['figures']
        params['Figure_Dir'] = outputs['figures']
        params['Figure_File'] = os.path.join(outputs['figures'], f'{template_name}.png')
    
    if 'html' in outputs:
        params['html_dir'] = outputs['html']
        params['Output_File'] = os.path.join(outputs['html'], f'{template_name}.html')
    
    # Save runtime parameters
    with open('params.runtime.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    # Save display parameters
    params_display = {k: v for k, v in params.items() 
                     if not any(x in k.lower() for x in ['output', 'path', 'dir', 'file'])}
    with open('config_used.json', 'w') as f:
        json.dump(params_display, f, indent=2)
    
    # Load and execute template
    try:
        mod = load_template_module(template_name)
        
        if not hasattr(mod, 'run_from_json'):
            print("[Bridge] ERROR: Template missing run_from_json function")
            sys.exit(2)
        
        # Check function signature
        import inspect
        sig = inspect.signature(mod.run_from_json)
        kwargs = {}
        
        if 'save_results' in sig.parameters:
            kwargs['save_results'] = True
        if 'show_plot' in sig.parameters:
            kwargs['show_plot'] = False
        
        print(f"[Bridge] Executing template with kwargs: {kwargs}")
        result = mod.run_from_json('params.runtime.json', **kwargs)
        print(f"[Bridge] Template completed, returned: {type(result).__name__}")
        
        # Handle in-memory results
        if result is not None:
            if isinstance(result, dict):
                print(f"[Bridge] Template saved files: {list(result.keys())}")
            else:
                saved = save_in_memory_results(result, outputs, template_name)
                if saved > 0:
                    print(f"[Bridge] Saved {saved} in-memory results")
        
        # Collect any orphan files
        moved = collect_orphan_files(outputs)
        if moved > 0:
            print(f"[Bridge] Collected {moved} orphan files")
        
        # Apply postprocessing if defined
        if 'postprocess' in config:
            config['postprocess'](outputs)
            print("[Bridge] Applied tool-specific postprocessing")
        
    except Exception as e:
        print(f"[Bridge] ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Verify outputs
    print("[Bridge] Verifying outputs...")
    found_outputs = False
    
    for output_type, path in outputs.items():
        if output_type == 'analysis':
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"[Bridge] ✓ {output_type}: {path} ({size:,} bytes)")
                found_outputs = True
            else:
                print(f"[Bridge] ✗ {output_type}: NOT FOUND")
        else:
            if os.path.exists(path) and os.path.isdir(path):
                files = os.listdir(path)
                if files:
                    print(f"[Bridge] ✓ {output_type}: {len(files)} files")
                    found_outputs = True
    
    if found_outputs:
        print("[Bridge] === SUCCESS ===")
    else:
        print("[Bridge] WARNING: No outputs created")

if __name__ == '__main__':
    main()