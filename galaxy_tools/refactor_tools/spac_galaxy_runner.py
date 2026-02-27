#!/usr/bin/env python3
"""
spac_galaxy_runner.py - Hybrid version combining refactored structure with robust parameter handling
Incorporates critical fixes from original wrapper for parameter processing
"""

import json
import os
import sys
import subprocess
import shutil
from pathlib import Path
import re

def main():
    """Main entry point for SPAC Galaxy runner"""
    if len(sys.argv) != 3:
        print("Usage: spac_galaxy_runner.py <params.json> <template_name>")
        sys.exit(1)
    
    params_path = sys.argv[1]
    template_name = sys.argv[2]
    
    print(f"=== SPAC Galaxy Runner v2.0 (Hybrid) ===")
    print(f"Template: {template_name}")
    print(f"Parameters: {params_path}")
    
    # Load parameters
    with open(params_path) as f:
        params = json.load(f)
    
    # Extract outputs specification from environment variable
    outputs_spec_env = os.environ.get('GALAXY_OUTPUTS_SPEC', '')
    if outputs_spec_env:
        try:
            outputs = json.loads(outputs_spec_env)
        except json.JSONDecodeError:
            print(f"WARNING: Could not parse GALAXY_OUTPUTS_SPEC: {outputs_spec_env}")
            outputs = determine_default_outputs(template_name)
    else:
        # Fallback: try to get from params
        outputs = params.pop('outputs', {})
        if isinstance(outputs, str):
            try:
                outputs = json.loads(unsanitize_galaxy_params(outputs))
            except json.JSONDecodeError:
                print(f"WARNING: Could not parse outputs: {outputs}")
                outputs = determine_default_outputs(template_name)
    
    print(f"Outputs specification: {outputs}")
    
    # CRITICAL: Unsanitize and normalize parameters (from original)
    params = process_galaxy_parameters(params, template_name)
    
    # Handle multiple file inputs that were copied to directories by Galaxy
    handle_multiple_file_inputs(params)
    
    # Create output directories
    create_output_directories(outputs)
    
    # Add output paths to params - critical for templates that save results
    params['save_results'] = True
    
    if 'analysis' in outputs:
        params['output_path'] = outputs['analysis']
        params['Output_Path'] = outputs['analysis']
        params['Output_File'] = outputs['analysis']
    
    if 'DataFrames' in outputs:
        df_path = outputs['DataFrames']
        # Check if it's a single file or a directory
        if df_path.endswith('.csv') or df_path.endswith('.tsv'):
            # Single file output (like Load CSV Files)
            params['output_file'] = df_path
            params['Output_File'] = df_path
            print(f"  Set output_file to: {df_path}")
        else:
            # Directory for multiple files (like boxplot)
            params['output_dir'] = df_path
            params['Export_Dir'] = df_path
            params['Output_File'] = os.path.join(df_path, f'{template_name}_output.csv')
            print(f"  Set output_dir to: {df_path}")
    
    if 'figures' in outputs:
        fig_dir = outputs['figures']
        params['figure_dir'] = fig_dir
        params['Figure_Dir'] = fig_dir
        params['Figure_File'] = os.path.join(fig_dir, f'{template_name}.png')
        print(f"  Set figure_dir to: {fig_dir}")
    
    if 'html' in outputs:
        html_dir = outputs['html']
        params['html_dir'] = html_dir
        params['Output_File'] = os.path.join(html_dir, f'{template_name}.html')
        print(f"  Set html_dir to: {html_dir}")
    
    # Save config for debugging (without outputs key)
    with open('config_used.json', 'w') as f:
        config_data = {k: v for k, v in params.items() if k not in ['outputs']}
        json.dump(config_data, f, indent=2)
    
    # Save params for template execution
    with open('params_exec.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    # Find and execute template
    template_path = find_template(template_name)
    if not template_path:
        print(f"ERROR: Template for {template_name} not found")
        sys.exit(1)
    
    # Run template
    exit_code = execute_template(template_path, 'params_exec.json')
    if exit_code != 0:
        print(f"ERROR: Template failed with exit code {exit_code}")
        sys.exit(exit_code)
    
    # Handle output mapping for specific tools
    handle_output_mapping(template_name, outputs)
    
    # Verify outputs
    verify_outputs(outputs)
    
    # Save snapshot for debugging
    with open('params_snapshot.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    print("=== Execution Complete ===")
    sys.exit(0)

def unsanitize_galaxy_params(s: str) -> str:
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

def process_galaxy_parameters(params: dict, template_name: str) -> dict:
    """Process Galaxy parameters - unsanitize and normalize (from original wrapper)"""
    print("\n=== Processing Galaxy Parameters ===")
    
    # Step 1: Recursively unsanitize all parameters
    def recursive_unsanitize(obj):
        if isinstance(obj, str):
            unsanitized = unsanitize_galaxy_params(obj).strip()
            # Try to parse JSON strings
            if (unsanitized.startswith('[') and unsanitized.endswith(']')) or \
               (unsanitized.startswith('{') and unsanitized.endswith('}')):
                try:
                    return json.loads(unsanitized)
                except:
                    return unsanitized
            return unsanitized
        elif isinstance(obj, dict):
            return {k: recursive_unsanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_unsanitize(item) for item in obj]
        return obj
    
    params = recursive_unsanitize(params)
    
    # Step 2: Handle specific parameter normalizations
    
    # Special handling for String_Columns in load_csv templates
    if 'load_csv' in template_name and 'String_Columns' in params:
        value = params['String_Columns']
        if not isinstance(value, list):
            if value in [None, "", "[]", "__ob____cb__", []]:
                params['String_Columns'] = []
            elif isinstance(value, str):
                s = value.strip()
                if s and s != '[]':
                    if ',' in s:
                        params['String_Columns'] = [item.strip() for item in s.split(',') if item.strip()]
                    else:
                        params['String_Columns'] = [s] if s else []
                else:
                    params['String_Columns'] = []
            else:
                params['String_Columns'] = []
        print(f"  Normalized String_Columns: {params['String_Columns']}")
    
    # Handle Feature_Regex specially - MUST BE AFTER Features_to_Analyze processing
    if 'Feature_Regex' in params:
        value = params['Feature_Regex']
        if value in [[], [""], "__ob____cb__", "[]", "", None]:
            params['Feature_Regex'] = []
            print("  Cleared empty Feature_Regex parameter")
        elif isinstance(value, list) and value:
            # Join regex patterns with |
            params['Feature_Regex'] = "|".join(str(v) for v in value if v)
            print(f"  Joined Feature_Regex list: {params['Feature_Regex']}")
    
    # Handle Features_to_Analyze - split if it's a single string with spaces or commas
    if 'Features_to_Analyze' in params:
        value = params['Features_to_Analyze']
        if isinstance(value, str):
            # Check for comma-separated or space-separated features
            if ',' in value:
                params['Features_to_Analyze'] = [item.strip() for item in value.split(',') if item.strip()]
                print(f"  Split Features_to_Analyze on comma: {value} -> {params['Features_to_Analyze']}")
            elif ' ' in value:
                # This is likely multiple features in a single string
                params['Features_to_Analyze'] = [item.strip() for item in value.split() if item.strip()]
                print(f"  Split Features_to_Analyze on space: {value} -> {params['Features_to_Analyze']}")
            elif value:
                params['Features_to_Analyze'] = [value]
                print(f"  Wrapped Features_to_Analyze in list: {params['Features_to_Analyze']}")
    
    # Handle Feature_s_to_Plot for boxplot
    if 'Feature_s_to_Plot' in params:
        value = params['Feature_s_to_Plot']
        # Check if it's "All"
        if value == "All" or value == ["All"]:
            params['Feature_s_to_Plot'] = ["All"]
            print("  Set Feature_s_to_Plot to ['All']")
        elif isinstance(value, str) and value not in ["", "[]"]:
            params['Feature_s_to_Plot'] = [value]
            print(f"  Wrapped Feature_s_to_Plot in list: {params['Feature_s_to_Plot']}")
    
    # Normalize list parameters
    list_params = ['Annotation_s_', 'Features', 'Markers', 'Markers_to_Plot', 
                   'Phenotypes', 'Binary_Phenotypes', 'Features_to_Analyze']
    
    for key in list_params:
        if key in params:
            value = params[key]
            if not isinstance(value, list):
                if value in [None, ""]:
                    continue
                elif isinstance(value, str):
                    if ',' in value:
                        params[key] = [item.strip() for item in value.split(',') if item.strip()]
                        print(f"  Split {key} on comma: {params[key]}")
                    else:
                        params[key] = [value]
                        print(f"  Wrapped {key} in list: {params[key]}")
    
    # Fix single-element lists for coordinate columns
    coordinate_keys = ['X_Coordinate_Column', 'Y_Coordinate_Column', 
                       'X_centroid', 'Y_centroid', 'Primary_Annotation', 
                       'Secondary_Annotation', 'Annotation']
    
    for key in coordinate_keys:
        if key in params:
            value = params[key]
            if isinstance(value, list) and len(value) == 1:
                params[key] = value[0]
                print(f"  Extracted single value from {key}: {params[key]}")
    
    return params

def determine_default_outputs(template_name: str) -> dict:
    """Determine default outputs based on template name"""
    if 'boxplot' in template_name or 'plot' in template_name or 'histogram' in template_name:
        return {'DataFrames': 'dataframe_folder', 'figures': 'figure_folder'}
    elif 'load_csv' in template_name:
        # Load CSV Files produces a single CSV file, not a folder
        return {'DataFrames': 'combined_data.csv'}
    elif 'interactive' in template_name:
        return {'html': 'html_folder'}
    else:
        return {'analysis': 'transform_output.pickle'}

def handle_multiple_file_inputs(params):
    """
    Handle multiple file inputs that Galaxy copies to directories.
    Galaxy copies multiple files to xxx_dir directories.
    """
    print("\n=== Handling Multiple File Inputs ===")
    
    # Check for directory inputs that indicate multiple files
    for key in list(params.keys()):
        # Check if Galaxy created a _dir directory for this input
        dir_name = f"{key}_dir"
        if os.path.isdir(dir_name):
            params[key] = dir_name
            print(f"  Updated {key} -> {dir_name}")
            # List files in the directory
            files = os.listdir(dir_name)
            print(f"    Contains {len(files)} files")
            for f in files[:3]:
                print(f"      - {f}")
            if len(files) > 3:
                print(f"      ... and {len(files)-3} more")
    
    # Special case for CSV_Files (Load CSV Files tool)
    if 'CSV_Files' in params:
        # Check for csv_input_dir created by Galaxy command
        if os.path.exists('csv_input_dir') and os.path.isdir('csv_input_dir'):
            params['CSV_Files'] = 'csv_input_dir'
            print(f"  Using csv_input_dir for CSV_Files")
        elif os.path.isdir('CSV_Files_dir'):
            params['CSV_Files'] = 'CSV_Files_dir'
            print(f"  Updated CSV_Files -> CSV_Files_dir")
        elif isinstance(params['CSV_Files'], str) and os.path.isfile(params['CSV_Files']):
            # Single file - get its directory
            params['CSV_Files'] = os.path.dirname(params['CSV_Files'])
            print(f"  Using directory of CSV file: {params['CSV_Files']}")

def create_output_directories(outputs):
    """Create directories for collection outputs"""
    print("\n=== Creating Output Directories ===")
    
    for output_type, path in outputs.items():
        if path.endswith('_folder') or path.endswith('_dir'):
            # This is a directory for multiple files
            os.makedirs(path, exist_ok=True)
            print(f"  Created directory: {path}")
        else:
            # For single files, ensure parent directory exists if there is one
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
                print(f"  Created parent directory: {parent}")
            else:
                print(f"  Single file output: {path} (no directory needed)")
    
    # Add output parameters to params for templates that need them
    # This is critical for templates like boxplot that check for these
    return outputs

def find_template(template_name):
    """Find the template Python file"""
    print("\n=== Finding Template ===")
    
    # Determine template filename
    if template_name == 'load_csv_files':
        template_py = 'load_csv_files_with_config.py'
    else:
        template_py = f'{template_name}_template.py'
    
    # Search paths (adjust based on your container/environment)
    search_paths = [
        f'/opt/spac/templates/{template_py}',
        f'/app/spac/templates/{template_py}',
        f'/opt/SCSAWorkflow/src/spac/templates/{template_py}',
        f'/usr/local/lib/python3.9/site-packages/spac/templates/{template_py}',
        f'./templates/{template_py}',
        f'./{template_py}'
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"  Found: {path}")
            return path
    
    print(f"  ERROR: {template_py} not found in:")
    for path in search_paths:
        print(f"    - {path}")
    return None

def execute_template(template_path, params_file):
    """Execute the SPAC template"""
    print("\n=== Executing Template ===")
    print(f"  Command: python3 {template_path} {params_file}")
    
    # Run template and capture output
    result = subprocess.run(
        ['python3', template_path, params_file],
        capture_output=True,
        text=True
    )
    
    # Save stdout and stderr
    with open('tool_stdout.txt', 'w') as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
    
    # Display output
    if result.stdout:
        print("  Output:")
        lines = result.stdout.split('\n')
        for line in lines[:20]:  # First 20 lines
            print(f"    {line}")
        if len(lines) > 20:
            print(f"    ... ({len(lines)-20} more lines)")
    
    if result.stderr:
        print("  Errors:", file=sys.stderr)
        for line in result.stderr.split('\n'):
            if line.strip():
                print(f"    {line}", file=sys.stderr)
    
    return result.returncode

def handle_output_mapping(template_name, outputs):
    """
    Map template outputs to expected locations.
    Generic approach: find outputs based on pattern matching.
    """
    print("\n=== Output Mapping ===")
    
    for output_type, expected_path in outputs.items():
        # Skip if already exists at expected location
        if os.path.exists(expected_path):
            print(f"  {output_type}: Already at {expected_path}")
            continue
        
        # Handle single file outputs
        if expected_path.endswith('.csv') or expected_path.endswith('.tsv') or \
           expected_path.endswith('.pickle') or expected_path.endswith('.h5ad'):
            find_and_move_output(output_type, expected_path)
        
        # Handle folder outputs - check if a default folder exists
        elif expected_path.endswith('_folder') or expected_path.endswith('_dir'):
            default_folder = output_type.lower() + '_folder'
            if default_folder != expected_path and os.path.isdir(default_folder):
                print(f"  Moving {default_folder} to {expected_path}")
                shutil.move(default_folder, expected_path)

def find_and_move_output(output_type, expected_path):
    """
    Find output file based on extension and move to expected location.
    More generic approach without hardcoded paths.
    """
    ext = os.path.splitext(expected_path)[1]  # e.g., '.csv'
    basename = os.path.basename(expected_path)
    
    print(f"  Looking for {output_type} output ({ext} file)...")
    
    # Search in common output locations
    search_dirs = ['.', 'dataframe_folder', 'output', 'results']
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        if os.path.isdir(search_dir):
            # Find files with matching extension
            matches = [f for f in os.listdir(search_dir) 
                      if f.endswith(ext)]
            
            if len(matches) == 1:
                source = os.path.join(search_dir, matches[0])
                print(f"    Found: {source}")
                print(f"    Moving to: {expected_path}")
                shutil.move(source, expected_path)
                return
            elif len(matches) > 1:
                # Multiple matches - use the largest or most recent
                matches_with_size = [(f, os.path.getsize(os.path.join(search_dir, f))) 
                                   for f in matches]
                matches_with_size.sort(key=lambda x: x[1], reverse=True)
                source = os.path.join(search_dir, matches_with_size[0][0])
                print(f"    Found multiple {ext} files, using largest: {source}")
                shutil.move(source, expected_path)
                return
    
    # Also check if file exists with different name in current dir
    current_dir_matches = [f for f in os.listdir('.') 
                          if f.endswith(ext) and f != basename]
    if current_dir_matches:
        source = current_dir_matches[0]
        print(f"    Found: {source}")
        print(f"    Moving to: {expected_path}")
        shutil.move(source, expected_path)
        return
    
    print(f"    WARNING: No {ext} file found for {output_type}")

def verify_outputs(outputs):
    """Verify that expected outputs were created"""
    print("\n=== Output Verification ===")
    
    all_found = True
    for output_type, path in outputs.items():
        if os.path.exists(path):
            if os.path.isdir(path):
                files = os.listdir(path)
                total_size = sum(os.path.getsize(os.path.join(path, f)) 
                               for f in files)
                print(f"  ✔ {output_type}: {len(files)} files in {path} "
                      f"({format_size(total_size)})")
                # Show first few files
                for f in files[:3]:
                    size = os.path.getsize(os.path.join(path, f))
                    print(f"      - {f} ({format_size(size)})")
                if len(files) > 3:
                    print(f"      ... and {len(files)-3} more")
            else:
                size = os.path.getsize(path)
                print(f"  ✔ {output_type}: {path} ({format_size(size)})")
        else:
            print(f"  ✗ {output_type}: NOT FOUND at {path}")
            all_found = False
    
    if not all_found:
        print("\n  WARNING: Some outputs not found!")
        print("  Check tool_stdout.txt for errors")
        # Don't exit with error - let Galaxy handle missing outputs

def format_size(bytes):
    """Format byte size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

if __name__ == '__main__':
    main()