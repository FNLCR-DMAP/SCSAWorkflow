#!/usr/bin/env python3
"""
Generalized NIDAP to Galaxy synthesizer - Production Version v11
- No hardcoded tool-specific logic
- Blueprint-driven for all tools
- Handles multiple files/columns via blueprint flags
- FIXED: Use 'binary' instead of 'pickle' for Galaxy compatibility
- FIXED: Use 'set -eu' instead of 'set -euo pipefail' for broader shell compatibility
- FIXED: Pass outputs spec as environment variable to avoid encoding issues
- FIXED: Method signature for build_command_section
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

class GeneralizedNIDAPToGalaxySynthesizer:
    
    def __init__(self, docker_image: str = "nciccbr/spac:v1"):
        self.docker_image = docker_image
        self.galaxy_profile = "24.2"
        self.wrapper_script = Path('run_spac_template.sh')
        self.runner_script = Path('spac_galaxy_runner.py') 
    
    def slugify(self, name: str) -> str:
        """Convert name to valid Galaxy tool ID component"""
        s = re.sub(r'\[.*?\]', '', name).strip()
        s = s.lower()
        s = re.sub(r'\s+', '_', s)
        s = re.sub(r'[^a-z0-9_]+', '', s)
        s = re.sub(r'_+', '_', s)
        return s.strip('_')
    
    def escape_xml(self, text: str, is_attribute: bool = True) -> str:
        """Escape XML special characters"""
        if text is None:
            return ""
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        if is_attribute:
            text = text.replace('"', '&quot;')
            text = text.replace("'", '&apos;')
        return text
    
    def clean_description(self, description: str) -> str:
        """Clean NIDAP-specific content from descriptions"""
        if not description:
            return ""
        
        desc = str(description).replace('\r\n', '\n').replace('\r', '\n')
        desc = re.sub(r'\[DUET\s*Documentation\]\([^)]+\)', '', desc, flags=re.IGNORECASE)
        desc = re.sub(r'Please refer to\s+(?:,?\s*and\s*)+', '', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\\(?=\s*(?:\n|$))', '', desc)
        desc = re.sub(r'[ \t]{2,}', ' ', desc)
        desc = re.sub(r'\n{3,}', '\n\n', desc)
        
        return desc.strip()
    
    def determine_input_format(self, dataset: Dict, tool_name: str) -> str:
        """
        Determine the correct format for an input dataset.
        Simple mapping based on dataType field.
        Uses 'binary' instead of 'pickle' for Galaxy compatibility.
        """
        data_type = dataset.get('dataType', '').upper()
        
        # Handle comma-separated types (e.g., "CSV, Tabular")
        data_types = [dt.strip() for dt in data_type.split(',')]
        
        # Check for CSV/Tabular types
        if any(dt in ['CSV', 'TABULAR', 'TSV', 'TXT'] for dt in data_types):
            return 'csv,tabular,tsv,txt'
        
        # DataFrame types
        if any('DATAFRAME' in dt for dt in data_types):
            return 'csv,tabular,tsv,txt'
        
        # AnnData/H5AD types
        if any(dt in ['ANNDATA', 'H5AD', 'HDF5'] for dt in data_types):
            return 'h5ad,h5,hdf5'
        
        # Pickle - use 'binary' for Galaxy compatibility
        if any('PICKLE' in dt for dt in data_types):
            return 'binary'
        
        # PYTHON_TRANSFORM_INPUT - default to binary (analysis objects)
        if 'PYTHON_TRANSFORM_INPUT' in data_type:
            return 'h5ad,binary'  # Use binary instead of pickle
        
        # Default fallback
        return 'h5ad,binary'  # Use binary instead of pickle
    
    def build_inputs_section(self, blueprint: Dict, tool_name: str) -> Tuple[List[str], List[str]]:
        """Build inputs from blueprint - generalized for all tools"""
        lines = []
        multiple_file_inputs = []  # Track which inputs accept multiple files
        
        # Handle input datasets
        for dataset in blueprint.get('inputDatasets', []):
            name = dataset.get('key', 'input_data')
            label = self.escape_xml(dataset.get('displayName', 'Input Data'))
            desc = self.escape_xml(self.clean_description(dataset.get('description', '')))
            
            # Determine format - now simpler with direct dataType mapping
            formats = self.determine_input_format(dataset, tool_name)
            
            # Check if multiple files allowed (from blueprint)
            is_multiple = dataset.get('isMultiple', False)
            
            if is_multiple:
                multiple_file_inputs.append(name)
                lines.append(
                    f'        <param name="{name}" type="data" '
                    f'format="{formats}" multiple="true" '
                    f'label="{label}" help="{desc}"/>'
                )
            else:
                lines.append(
                    f'        <param name="{name}" type="data" '
                    f'format="{formats}" '
                    f'label="{label}" help="{desc}"/>'
                )
        
        # Handle explicit column definitions from 'columns' schema
        for col in blueprint.get('columns', []):
            key = col.get('key')
            if not key:
                continue
            
            label = self.escape_xml(col.get('displayName', key))
            desc = self.escape_xml(col.get('description', ''))
            # isMulti can be True, False, or None (None means False)
            is_multi = col.get('isMulti') == True
            
            # Use text inputs for column names
            if is_multi:
                lines.append(
                    f'        <param name="{key}" type="text" area="true" '
                    f'value="" optional="true" '
                    f'label="{label}" '
                    f'help="{desc} (Enter column names, one per line or comma-separated)"/>'
                )
            else:
                lines.append(
                    f'        <param name="{key}" type="text" '
                    f'value="" optional="true" '
                    f'label="{label}" '
                    f'help="{desc} (Enter column name)"/>'
                )
        
        # Handle regular parameters
        for param in blueprint.get('parameters', []):
            key = param.get('key')
            if not key:
                continue
            
            label = self.escape_xml(param.get('displayName', key))
            desc = self.escape_xml(self.clean_description(param.get('description', '')))
            param_type = param.get('paramType', 'STRING').upper()
            default = param.get('defaultValue', '')
            is_optional = param.get('isOptional', False)
            
            # Add optional attribute if needed
            optional_attr = ' optional="true"' if is_optional else ''
            
            if param_type == 'BOOLEAN':
                checked = 'true' if str(default).strip().lower() == 'true' else 'false'
                lines.append(
                    f'        <param name="{key}" type="boolean" '
                    f'truevalue="True" falsevalue="False" '
                    f'checked="{checked}" label="{label}" help="{desc}"/>'
                )
            
            elif param_type == 'INTEGER':
                lines.append(
                    f'        <param name="{key}" type="integer" '
                    f'value="{self.escape_xml(str(default))}" '
                    f'{optional_attr} label="{label}" help="{desc}"/>'
                )
            
            elif param_type in ['NUMBER', 'FLOAT']:
                lines.append(
                    f'        <param name="{key}" type="float" '
                    f'value="{self.escape_xml(str(default))}" '
                    f'{optional_attr} label="{label}" help="{desc}"/>'
                )
            
            elif param_type == 'SELECT':
                options = param.get('paramValues', [])
                lines.append(f'        <param name="{key}" type="select" {optional_attr} label="{label}" help="{desc}">')
                for opt in options:
                    selected = ' selected="true"' if str(opt) == str(default) else ''
                    opt_escaped = self.escape_xml(str(opt))
                    lines.append(f'            <option value="{opt_escaped}"{selected}>{opt_escaped}</option>')
                lines.append('        </param>')
            
            elif param_type == 'LIST':
                # Handle LIST type parameters - convert list to simple string
                if isinstance(default, list):
                    # Filter out empty strings and join
                    filtered = [str(x) for x in default if x and str(x).strip()]
                    default = ', '.join(filtered) if filtered else ''
                elif default == '[""]' or default == "['']" or default == '[]':
                    # Handle common empty list representations
                    default = ''
                lines.append(
                    f'        <param name="{key}" type="text" area="true" '
                    f'value="{self.escape_xml(str(default))}" '
                    f'{optional_attr} label="{label}" '
                    f'help="{desc} (Enter as comma-separated list or one per line)"/>'
                )
            
            else:  # STRING
                lines.append(
                    f'        <param name="{key}" type="text" '
                    f'value="{self.escape_xml(str(default))}" '
                    f'{optional_attr} label="{label}" help="{desc}"/>'
                )
        
        return lines, multiple_file_inputs
    
    def build_outputs_section(self, outputs: Dict) -> List[str]:
        """Build outputs section based on blueprint specification"""
        lines = []
        
        for output_type, output_path in outputs.items():
            
            # Determine if single file or collection
            is_collection = (output_path.endswith('_folder') or 
                           output_path.endswith('_dir'))
            
            if not is_collection:
                # Single file output
                if output_type == 'analysis':
                    if '.h5ad' in output_path:
                        fmt = 'h5ad'
                    elif '.pickle' in output_path or '.pkl' in output_path:
                        fmt = 'binary'  # Use binary instead of pickle
                    else:
                        fmt = 'binary'
                    
                    lines.append(
                        f'        <data name="analysis" format="{fmt}" '
                        f'from_work_dir="{output_path}" '
                        f'label="${{tool.name}} on ${{on_string}}: Analysis"/>'
                    )

                elif output_type == 'DataFrames' and (output_path.endswith('.csv') or output_path.endswith('.tsv')):
                    # Single DataFrame file output
                    fmt = 'csv' if output_path.endswith('.csv') else 'tabular'
                    lines.append(
                        f'        <data name="output_dataframe" format="{fmt}" '
                        f'from_work_dir="{output_path}" '
                        f'label="${{tool.name}} on ${{on_string}}: Output Data"/>'
                    ) 
                
                elif output_type == 'figure':
                    ext = output_path.split('.')[-1] if '.' in output_path else 'png'
                    lines.append(
                        f'        <data name="output_figure" format="{ext}" '
                        f'from_work_dir="{output_path}" '
                        f'label="${{tool.name}} on ${{on_string}}: Figure"/>'
                    )
                
                elif output_type == 'html':
                    lines.append(
                        f'        <data name="output_html" format="html" '
                        f'from_work_dir="{output_path}" '
                        f'label="${{tool.name}} on ${{on_string}}: HTML Report"/>'
                    )
            
            else:
                # Collection outputs
                if output_type == 'DataFrames':
                    lines.append(
                        '        <collection name="dataframes" type="list" '
                        'label="${tool.name} on ${on_string}: DataFrames">'
                    )
                    lines.append(f'            <discover_datasets pattern="(?P&lt;name&gt;.+)\.csv" directory="{output_path}" format="csv"/>')
                    lines.append(f'            <discover_datasets pattern="(?P&lt;name&gt;.+)\.tsv" directory="{output_path}" format="tabular"/>')
                    lines.append('        </collection>')
                
                elif output_type == 'figures':
                    lines.append(
                        '        <collection name="figures" type="list" '
                        'label="${tool.name} on ${on_string}: Figures">'
                    )
                    lines.append(f'            <discover_datasets pattern="(?P&lt;name&gt;.+)\.png" directory="{output_path}" format="png"/>')
                    lines.append(f'            <discover_datasets pattern="(?P&lt;name&gt;.+)\.pdf" directory="{output_path}" format="pdf"/>')
                    lines.append(f'            <discover_datasets pattern="(?P&lt;name&gt;.+)\.svg" directory="{output_path}" format="svg"/>')
                    lines.append('        </collection>')
                
                elif output_type == 'html':
                    lines.append(
                        '        <collection name="html_outputs" type="list" '
                        'label="${tool.name} on ${on_string}: HTML Reports">'
                    )
                    lines.append(f'            <discover_datasets pattern="(?P&lt;name&gt;.+)\.html" directory="{output_path}" format="html"/>')
                    lines.append('        </collection>')
        
        # Debug outputs
        lines.append('        <data name="params_snapshot" format="json" from_work_dir="params_snapshot.json" label="${tool.name} on ${on_string}: Params Snapshot"/>')
        lines.append('        <data name="config_used" format="json" from_work_dir="config_used.json" label="${tool.name} on ${on_string}: Config Used"/>')
        lines.append('        <data name="execution_log" format="txt" from_work_dir="tool_stdout.txt" label="${tool.name} on ${on_string}: Execution Log"/>')
        
        return lines
    
    def build_command_section(self, tool_name: str, blueprint: Dict, multiple_file_inputs: List[str], outputs_spec: Dict) -> str:
        """Build command section - generalized for all tools
        FIXED: Use 'set -eu' instead of 'set -euo pipefail' for broader shell compatibility
        FIXED: Pass outputs spec as environment variable to avoid encoding issues
        """
        
        # Convert outputs spec to JSON string
        outputs_json = json.dumps(outputs_spec)
        
        # Check if any inputs accept multiple files
        has_multiple_files = len(multiple_file_inputs) > 0
        
        if has_multiple_files:
            # Generate file copying logic for each multiple input
            copy_sections = []
            for input_name in multiple_file_inputs:
                # Use double curly braces to escape them in f-strings
                copy_sections.append(f'''
            ## Create directory for {input_name}
            mkdir -p {input_name}_dir &&
            
            ## Copy files to directory with original names
            #for $i, $file in enumerate(${input_name})
                cp '${{file}}' '{input_name}_dir/${{file.name}}' &&
            #end for''')
            
            copy_logic = ''.join(copy_sections)
            
            command_section = f'''        <command detect_errors="exit_code"><![CDATA[
            set -eu &&
            export GALAXY_OUTPUTS_SPEC='{outputs_json}' &&
            {copy_logic}
            
            ## Debug: Show params.json
            echo "=== params.json ===" >&2 &&
            cat "$params_json" >&2 &&
            echo "==================" >&2 &&
            
            ## Save snapshot
            cp "$params_json" params_snapshot.json &&
            
            ## Run wrapper
            bash $__tool_directory__/run_spac_template.sh "$params_json" "{tool_name}"
        ]]></command>'''
        else:
            # Standard command for single-file inputs
            command_section = f'''        <command detect_errors="exit_code"><![CDATA[
            set -eu &&
            export GALAXY_OUTPUTS_SPEC='{outputs_json}' &&
            
            ## Debug: Show params.json
            echo "=== params.json ===" >&2 &&
            cat "$params_json" >&2 &&
            echo "==================" >&2 &&
            
            ## Save snapshot
            cp "$params_json" params_snapshot.json &&
            
            ## Run wrapper
            bash $__tool_directory__/run_spac_template.sh "$params_json" "{tool_name}"
        ]]></command>'''
        
        return command_section
    
    def get_template_filename(self, title: str, tool_name: str) -> str:
        """Get the correct template filename"""
        # Check if there's a custom mapping in the blueprint
        # Otherwise use standard naming convention
        if title == 'Load CSV Files' or tool_name == 'load_csv_files':
            return 'load_csv_files_with_config.py'
        else:
            return f'{tool_name}_template.py'
    
    def generate_tool(self, json_path: Path, output_dir: Path) -> Dict:
        """Generate Galaxy tool from NIDAP JSON blueprint"""
        
        with open(json_path, 'r') as f:
            blueprint = json.load(f)
        
        title = blueprint.get('title', 'Unknown Tool')
        clean_title = re.sub(r'\[.*?\]', '', title).strip()
        
        tool_name = self.slugify(clean_title)
        tool_id = f'spac_{tool_name}'
        
        # Get outputs from blueprint
        outputs_spec = blueprint.get('outputs', {})
        if not outputs_spec:
            outputs_spec = {'analysis': 'transform_output.pickle'}
        
        # Get template filename (could be in blueprint too)
        template_filename = blueprint.get('templateFilename', 
                                        self.get_template_filename(clean_title, tool_name))
        
        # Build sections - pass tool_name and outputs_spec for context
        inputs_lines, multiple_file_inputs = self.build_inputs_section(blueprint, tool_name)
        outputs_lines = self.build_outputs_section(outputs_spec)
        command_section = self.build_command_section(tool_name, blueprint, multiple_file_inputs, outputs_spec)
        
        # Generate description
        full_desc = self.clean_description(blueprint.get('description', ''))
        short_desc = full_desc.split('\n')[0] if full_desc else ''
        if len(short_desc) > 100:
            short_desc = short_desc[:97] + '...'
        
        # Build help section
        help_sections = []
        help_sections.append(f'**{title}**\n')
        help_sections.append(f'{full_desc}\n')
        help_sections.append('This tool is part of the SPAC (SPAtial single-Cell analysis) toolkit.\n')
        
        # Add usage notes based on input types
        if blueprint.get('columns'):
            help_sections.append('**Column Parameters:** Enter column names as text. Use comma-separation or one per line for multiple columns.')
        
        if any(p.get('paramType') == 'LIST' for p in blueprint.get('parameters', [])):
            help_sections.append('**List Parameters:** Use comma-separated values or one per line.')
            help_sections.append('**Special Values:** Enter "All" to select all items.')
        
        if multiple_file_inputs:
            help_sections.append(f'**Multiple File Inputs:** This tool accepts multiple files for: {", ".join(multiple_file_inputs)}')
        
        help_text = '\n'.join(help_sections)
        
        # Generate complete XML
        xml_content = f'''<tool id="{tool_id}" name="{self.escape_xml(title)}" version="1.0.0" profile="{self.galaxy_profile}">
    <description>{self.escape_xml(short_desc, False)}</description>
    
    <requirements>
        <container type="docker">{self.docker_image}</container>
    </requirements>
    
    <environment_variables>
        <variable name="SPAC_PYTHON">python3</variable>
    </environment_variables>
    
{command_section}
    
    <configfiles>
        <inputs name="params_json" filename="params.json" data_style="paths"/>
    </configfiles>
    
    <inputs>
{chr(10).join(inputs_lines)}
    </inputs>
    
    <outputs>
{chr(10).join(outputs_lines)}
    </outputs>
    
    <help><![CDATA[
        {help_text}
    ]]></help>
    
    <citations>
        <citation type="bibtex">
@misc{{spac_toolkit,
    author = {{FNLCR DMAP Team}},
    title = {{SPAC: SPAtial single-Cell analysis}},
    year = {{2024}},
    url = {{https://github.com/FNLCR-DMAP/SCSAWorkflow}}
}}
        </citation>
    </citations>
</tool>'''
        
        # Write files
        tool_dir = output_dir / tool_id
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        xml_path = tool_dir / f'{tool_id}.xml'
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        
        # Copy wrapper script
        if self.wrapper_script.exists():
            shutil.copy2(self.wrapper_script, tool_dir / 'run_spac_template.sh')

        # Copy runner script
        if self.runner_script.exists():
            shutil.copy2(self.runner_script, tool_dir / 'spac_galaxy_runner.py')
        else:
            print(f"  Warning: spac_galaxy_runner.py not found in current directory")
        
        return {
            'tool_id': tool_id,
            'tool_name': title,
            'xml_path': xml_path,
            'tool_dir': tool_dir,
            'template': template_filename,
            'outputs': outputs_spec
        }

def main():
    parser = argparse.ArgumentParser(
        description='Convert NIDAP templates to Galaxy tools - Generalized Version'
    )
    parser.add_argument('json_input', help='JSON file or directory')
    parser.add_argument('-o', '--output-dir', default='galaxy_tools')
    parser.add_argument('--docker-image', default='nciccbr/spac:v1')
    
    args = parser.parse_args()
    
    synthesizer = GeneralizedNIDAPToGalaxySynthesizer(
        docker_image=args.docker_image
    )
    
    json_input = Path(args.json_input)
    if json_input.is_file():
        json_files = [json_input]
    elif json_input.is_dir():
        json_files = sorted(json_input.glob('*.json'))
    else:
        print(f"Error: {json_input} not found")
        return 1
    
    print(f"Processing {len(json_files)} files")
    print(f"Docker image: {args.docker_image}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    failed = []
    
    for json_file in json_files:
        print(f"\nProcessing: {json_file.name}")
        try:
            result = synthesizer.generate_tool(json_file, output_dir)
            successful.append(result)
            print(f"  ✔ Created: {result['tool_id']}")
            print(f"  Template: {result['template']}")
            print(f"  Outputs: {list(result['outputs'].keys())}")
        except Exception as e:
            failed.append(json_file.name)
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(successful)} successful, {len(failed)} failed")
    
    if successful:
        snippet_path = output_dir / 'tool_conf_snippet.xml'
        with open(snippet_path, 'w') as f:
            f.write('<section id="spac_tools" name="SPAC Tools">\n')
            for result in sorted(successful, key=lambda x: x['tool_id']):
                tool_id = result['tool_id']
                f.write(f'    <tool file="spac/{tool_id}/{tool_id}.xml"/>\n')
            f.write('</section>\n')
        
        print(f"\nGenerated tool configuration snippet: {snippet_path}")
    
    return 0 if not failed else 1

if __name__ == '__main__':
    exit(main())