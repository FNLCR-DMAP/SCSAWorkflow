#!/usr/bin/env python3
"""
Galaxy XML Synthesizer - Generates Galaxy tool XML from blueprint JSON files.

This synthesizer properly handles all parameter types:
- Numeric types (INTEGER, NUMBER, FLOAT) with min/max bounds → Galaxy numeric types
- Boolean parameters → Galaxy boolean checkboxes  
- SELECT dropdowns → Galaxy select with options
- LIST parameters → Galaxy repeat structures
- Multi-value columns (isMulti=true) → Galaxy repeat structures
- Single-value columns → Galaxy text inputs
- String parameters → Galaxy text inputs

Version: 2.2 - Fixed setup_analysis Features_to_Analyze repeat structure issue
Fixed: Multi-value columns now correctly generate repeat structures instead of textareas
Author: FNLCR-DMAP Team
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Any
import re
import argparse
import sys


class GalaxyXMLSynthesizer:
    """
    Synthesize Galaxy XML purely from blueprint JSON.
    NO special cases, NO tool-specific logic.
    """
    
    def __init__(self, blueprint: Dict[str, Any], docker_image: str = "spac:latest"):
        self.blueprint = blueprint
        self.docker_image = docker_image
        
    def synthesize(self) -> str:
        """Generate Galaxy tool XML from blueprint."""
        tool = ET.Element('tool')
        
        # Extract basic info from blueprint
        tool_id = self._make_tool_id(self.blueprint.get('title', 'tool'))
        tool.set('id', tool_id)
        tool.set('name', self.blueprint.get('title', 'Tool'))
        tool.set('version', '2.0.0')
        tool.set('profile', '24.2')
        
        # Description
        desc = ET.SubElement(tool, 'description')
        desc.text = self._clean_text(self.blueprint.get('description', ''))[:200]
        
        # Requirements
        reqs = ET.SubElement(tool, 'requirements')
        container = ET.SubElement(reqs, 'container')
        container.set('type', 'docker')
        container.text = self.docker_image
        
        # Configfiles - Galaxy serializes inputs to JSON
        configfiles = ET.SubElement(tool, 'configfiles')
        inputs_config = ET.SubElement(configfiles, 'inputs')
        inputs_config.set('name', 'params_json')
        inputs_config.set('filename', 'galaxy_params.json')
        inputs_config.set('data_style', 'paths')
        
        # Command
        self._add_command(tool, tool_id)
        
        # Inputs
        self._add_inputs(tool)
        
        # Outputs - directly from blueprint, no inference
        self._add_outputs(tool)
        
        # Help
        help_elem = ET.SubElement(tool, 'help')
        help_elem.text = f"<![CDATA[\n{self._generate_help()}\n]]>"
        
        # Citations
        citations = ET.SubElement(tool, 'citations')
        citation = ET.SubElement(citations, 'citation')
        citation.set('type', 'doi')
        citation.text = '10.1038/s41586-019-1876-x'
        
        return self._format_xml(tool)
    
    def _make_tool_id(self, title: str) -> str:
        """Generate tool ID from title."""
        # Remove brackets and clean
        clean_title = re.sub(r'\[.*?\]', '', title).strip()
        tool_id = clean_title.lower().replace(' ', '_')
        return f"spac_{tool_id}"
    
    def _clean_text(self, text: str) -> str:
        """Clean text from markdown and escapes."""
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = text.replace('\\n', ' ')
        return text.strip()
    
    def _add_command(self, tool: ET.Element, tool_id: str):
        """Add command section - FIXED to use single-line commands."""
        # Collect parameter types for format_values.py
        bool_params = []
        list_params = []
        
        for param in self.blueprint.get('parameters', []):
            param_type = param.get('paramType')
            param_key = param.get('key')
            
            if param_type == 'BOOLEAN':
                bool_params.append(param_key)
            elif param_type == 'LIST':
                list_params.append(param_key)
        
        # Columns with isMulti are also lists
        for column in self.blueprint.get('columns', []):
            if column.get('isMulti'):
                list_params.append(column.get('key'))
        
        # Build command as a SINGLE LINE
        template_name = tool_id.replace('spac_', '') + '_template.py'
        
        # Build command parts
        cmd_parts = []
        
        # Add output config creation if needed
        if 'outputs' in self.blueprint:
            # Create JSON string with proper escaping for shell
            outputs_json = json.dumps(self.blueprint['outputs'])
            cmd_parts.append(f"echo '{outputs_json}' > outputs_config.json")
        
        # Build format_values.py command
        format_cmd = "python '$__tool_directory__/format_values.py' 'galaxy_params.json' 'cleaned_params.json'"
        
        if bool_params:
            format_cmd += f" --bool-values {' '.join(bool_params)}"
        if list_params:
            format_cmd += f" --list-values {' '.join(list_params)}"
        
        # Add output injection flags if outputs exist
        if 'outputs' in self.blueprint:
            format_cmd += " --inject-outputs --outputs-config outputs_config.json"
        
        cmd_parts.append(format_cmd)
        
        # Add template command
        cmd_parts.append(f"python '$__tool_directory__/{template_name}' 'cleaned_params.json'")
        
        # Join all parts with && on a SINGLE LINE
        full_command = ' && '.join(cmd_parts)
        
        command = ET.SubElement(tool, 'command')
        command.set('detect_errors', 'exit_code')
        command.text = f"<![CDATA[\n{full_command}\n]]>"
    
    def _add_inputs(self, tool: ET.Element):
        """Add inputs section."""
        inputs = ET.SubElement(tool, 'inputs')
        
        # Input datasets
        for dataset in self.blueprint.get('inputDatasets', []):
            param = ET.SubElement(inputs, 'param')
            param.set('name', dataset['key'])
            param.set('type', 'data')
            
            # Map data types
            data_type = dataset.get('dataType', '')
            if 'DATAFRAME' in data_type.upper():
                param.set('format', 'tabular,csv,tsv,txt')
            elif 'PYTHON' in data_type.upper() or 'ANNDATA' in data_type.upper():
                param.set('format', 'h5ad,binary,pickle')
            else:
                param.set('format', 'binary')
            
            param.set('label', dataset.get('displayName', dataset['key']))
            if dataset.get('description'):
                param.set('help', dataset['description'])
        
        # Parameters
        for param_def in self.blueprint.get('parameters', []):
            self._add_parameter(inputs, param_def)
        
        # Columns - treat consistently
        for column in self.blueprint.get('columns', []):
            self._add_column(inputs, column)
    
    def _add_parameter(self, inputs: ET.Element, param_def: Dict[str, Any]):
        """Add parameter based on type with proper datatype handling."""
        param_type = param_def.get('paramType', 'STRING')
        param_key = param_def.get('key')
        display_name = param_def.get('displayName', param_key)
        description = param_def.get('description', '')
        default_value = param_def.get('defaultValue')
        
        if param_type == 'LIST':
            # Use Galaxy repeat for lists
            repeat = ET.SubElement(inputs, 'repeat')
            repeat.set('name', f"{param_key}_repeat")
            repeat.set('title', display_name)
            
            param = ET.SubElement(repeat, 'param')
            param.set('name', 'value')
            param.set('type', 'text')
            param.set('label', 'Value')
            param.set('help', description)
            
            # Add default values if specified
            if default_value and isinstance(default_value, list):
                for val in default_value:
                    param.set('value', str(val))
                    break  # Only first default for repeat structure
        
        elif param_type == 'BOOLEAN':
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            param.set('type', 'boolean')
            param.set('label', display_name)
            
            # Handle boolean defaults
            if default_value is not None:
                if str(default_value).lower() in ['true', '1', 'yes']:
                    param.set('truevalue', 'True')
                    param.set('falsevalue', 'False')
                    param.set('checked', 'true')
                else:
                    param.set('truevalue', 'True')
                    param.set('falsevalue', 'False')
                    param.set('checked', 'false')
            else:
                param.set('truevalue', 'True')
                param.set('falsevalue', 'False')
                param.set('checked', 'false')
            
            if description:
                param.set('help', description)
        
        elif param_type == 'SELECT':
            # Handle SELECT type parameters
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            param.set('type', 'select')
            param.set('label', display_name)
            
            # Add options if paramValues is specified
            param_values = param_def.get('paramValues', [])
            if param_values:
                for value in param_values:
                    option = ET.SubElement(param, 'option')
                    option.set('value', str(value))
                    if default_value and str(value) == str(default_value):
                        option.set('selected', 'true')
                    option.text = str(value)
            
            if description:
                param.set('help', description)
        
        elif param_type in ['NUMBER', 'INTEGER', 'Positive integer', 'FLOAT', 'INT']:
            # Handle numeric types with proper Galaxy types
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            
            # Map blueprint types to Galaxy types
            if param_type in ['NUMBER', 'FLOAT']:
                param.set('type', 'float')
            elif param_type in ['INTEGER', 'INT', 'Positive integer']:
                param.set('type', 'integer')
            
            param.set('label', display_name)
            
            # Handle min/max bounds
            param_min = param_def.get('paramMin')
            param_max = param_def.get('paramMax')
            
            # Special handling for "Positive integer"
            if param_type == 'Positive integer':
                param.set('min', '1')
            elif param_min is not None:
                param.set('min', str(param_min))
            
            if param_max is not None:
                param.set('max', str(param_max))
            
            # Handle optional parameters
            if param_def.get('optional'):
                param.set('optional', 'true')
            
            # Set default value
            if default_value is not None:
                param.set('value', str(default_value))
            elif not param_def.get('optional'):
                # Provide type-appropriate defaults for non-optional params
                param.set('value', '0')
            
            if description:
                param.set('help', description)
        
        else:  # STRING and other text types
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            param.set('type', 'text')
            param.set('label', display_name)
            
            # Handle optional parameters
            if param_def.get('optional'):
                param.set('optional', 'true')
            
            # Set default value
            if default_value is not None:
                param.set('value', str(default_value))
            elif not param_def.get('optional'):
                param.set('value', '')
            
            if description:
                param.set('help', description)
    
    def _add_column(self, inputs: ET.Element, column: Dict[str, Any]):
        """Add column parameter - using repeat for multi-value columns."""
        param_key = column.get('key')
        display_name = column.get('displayName', param_key)
        description = column.get('description', '')
        is_multi = column.get('isMulti', False)
        
        if is_multi:
            # Multi-value column: use Galaxy repeat structure (like LIST params)
            repeat = ET.SubElement(inputs, 'repeat')
            repeat.set('name', f"{param_key}_repeat")
            repeat.set('title', display_name)
            
            param = ET.SubElement(repeat, 'param')
            param.set('name', 'value')
            param.set('type', 'text')
            param.set('label', 'Value')
            param.set('help', description)
            
            # Default values if specified
            default_value = column.get('defaultValue', [])
            if default_value:
                if isinstance(default_value, list) and default_value:
                    # Set first value as default for the repeat template
                    param.set('value', str(default_value[0]))
                elif not isinstance(default_value, list):
                    param.set('value', str(default_value))
        else:
            # Single-value column: regular text input
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            param.set('type', 'text')
            param.set('label', display_name)
            
            if column.get('optional'):
                param.set('optional', 'true')
            
            default_value = column.get('defaultValue')
            if default_value is not None:
                param.set('value', str(default_value))
            
            if description:
                param.set('help', description)
    
    def _add_outputs(self, tool: ET.Element):
        """Add outputs section - directly from blueprint."""
        outputs_elem = ET.SubElement(tool, 'outputs')
        
        # Debug outputs - always include
        debug1 = ET.SubElement(outputs_elem, 'data')
        debug1.set('name', 'params_json_debug')
        debug1.set('format', 'json')
        debug1.set('from_work_dir', 'galaxy_params.json')
        debug1.set('label', '${tool.name} on ${on_string}: Raw Parameters')
        
        debug2 = ET.SubElement(outputs_elem, 'data')
        debug2.set('name', 'cleaned_params_debug')
        debug2.set('format', 'json')
        debug2.set('from_work_dir', 'cleaned_params.json')
        debug2.set('label', '${tool.name} on ${on_string}: Cleaned Parameters')
        
        # Main outputs from blueprint
        outputs = self.blueprint.get('outputs', {})
        
        for key, config in outputs.items():
            if isinstance(config, dict):
                output_type = config.get('type', 'file')
                output_name = config.get('name', key)
            else:
                # Simple format: assume file
                output_type = 'file'
                output_name = str(config)
            
            if output_type == 'directory':
                # Directory: use collection
                collection = ET.SubElement(outputs_elem, 'collection')
                collection.set('name', key)
                collection.set('type', 'list')
                collection.set('label', f'${{tool.name}} on ${{on_string}}: {key.title()}')
                
                discover = ET.SubElement(collection, 'discover_datasets')
                discover.set('pattern', '__name_and_ext__')
                discover.set('directory', output_name)
            else:
                # File: direct data element
                data = ET.SubElement(outputs_elem, 'data')
                data.set('name', key)
                
                # Infer format from extension
                if '.csv' in output_name:
                    data.set('format', 'csv')
                elif '.png' in output_name:
                    data.set('format', 'png')
                elif '.pickle' in output_name or '.pkl' in output_name:
                    data.set('format', 'binary')
                elif '.h5ad' in output_name:
                    data.set('format', 'h5ad')
                else:
                    data.set('format', 'auto')
                
                data.set('from_work_dir', output_name)
                data.set('label', f'${{tool.name}} on ${{on_string}}: {key.title()}')
    
    def _generate_help(self) -> str:
        """Generate help text."""
        title = self.blueprint.get('title', 'Tool')
        desc = self._clean_text(self.blueprint.get('description', ''))
        
        help_lines = [
            f"**{title}**",
            "",
            desc,
            "",
            "**Outputs:**"
        ]
        
        outputs = self.blueprint.get('outputs', {})
        for key, config in outputs.items():
            if isinstance(config, dict):
                output_type = config.get('type', 'file')
                output_name = config.get('name', key)
                help_lines.append(f"- {key}: {output_type} ({output_name})")
            else:
                help_lines.append(f"- {key}: {config}")
        
        return '\n'.join(help_lines)
    
    def _format_xml(self, elem: ET.Element) -> str:
        """Format XML with proper indentation, preserving CDATA sections."""
        # First pass: generate XML without pretty printing
        rough = ET.tostring(elem, encoding='unicode')
        
        # Use minidom for pretty printing
        dom = minidom.parseString(rough)
        pretty = dom.toprettyxml(indent="    ")
        
        # Clean up extra whitespace
        lines = [line for line in pretty.split('\n') if line.strip()]
        
        # Skip XML declaration
        if lines[0].startswith('<?xml'):
            lines = lines[1:]
        
        # Join and fix CDATA sections (minidom escapes them)
        result = '\n'.join(lines)
        result = result.replace('&lt;![CDATA[', '<![CDATA[')
        result = result.replace(']]&gt;', ']]>')
        
        # Fix HTML entities within CDATA sections
        import re
        def unescape_cdata(match):
            content = match.group(1)
            # Unescape HTML entities within CDATA
            content = content.replace('&quot;', '"')
            content = content.replace('&apos;', "'")
            content = content.replace('&lt;', '<')
            content = content.replace('&gt;', '>')
            content = content.replace('&amp;', '&')
            return '<![CDATA[\n' + content + '\n]]>'
        
        result = re.sub(r'<!\[CDATA\[(.*?)\]\]>', unescape_cdata, result, flags=re.DOTALL)
        
        # Add XML declaration
        return '<?xml version="1.0" ?>\n' + result


def process_blueprint(blueprint_path: Path, output_dir: Path, docker_image: str = "spac:mvp"):
    """Process a single blueprint to generate Galaxy XML."""
    print(f"Processing: {blueprint_path.name}")
    
    # Load blueprint
    with open(blueprint_path, 'r') as f:
        blueprint = json.load(f)
    
    # Generate XML
    synthesizer = GalaxyXMLSynthesizer(blueprint, docker_image)
    xml_content = synthesizer.synthesize()
    
    # Extract tool name from blueprint
    title = blueprint.get('title', 'tool')
    clean_title = re.sub(r'\[.*?\]', '', title).strip()
    tool_name = clean_title.lower().replace(' ', '_')
    
    # Write XML file
    xml_path = output_dir / f"{tool_name}.xml"
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    
    print(f"  -> Generated: {xml_path.name}")
    
    # Verify command format
    if '\n&&\n' in xml_content or '\n    &&\n' in xml_content:
        print(f"  WARNING: Multi-line command detected in {xml_path.name}")
    
    return xml_path


def batch_process(input_pattern: str, output_dir: str, docker_image: str = "spac:mvp"):
    """Process multiple blueprint files matching a pattern."""
    input_path = Path(input_pattern)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find matching files
    if input_path.is_file():
        blueprint_files = [input_path]
    elif input_path.is_dir():
        # Process all JSON files in directory
        blueprint_files = list(input_path.glob("template_json_*.json"))
    else:
        # Handle glob pattern
        parent = input_path.parent
        pattern = input_path.name
        blueprint_files = list(parent.glob(pattern))
    
    if not blueprint_files:
        print(f"No blueprint files found matching: {input_pattern}")
        return 1
    
    print(f"Found {len(blueprint_files)} blueprint files to process")
    print("=" * 60)
    
    generated_files = []
    for blueprint_file in sorted(blueprint_files):
        try:
            xml_file = process_blueprint(blueprint_file, output_path, docker_image)
            generated_files.append(xml_file)
        except Exception as e:
            print(f"  ERROR processing {blueprint_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Successfully generated {len(generated_files)} Galaxy XML files")
    
    return 0


def main():
    """Main entry point for the synthesizer."""
    parser = argparse.ArgumentParser(
        description="Generate Galaxy tool XML from SPAC blueprint JSON files"
    )
    parser.add_argument(
        "blueprint",
        help="Path to blueprint JSON file or pattern (e.g., 'template_json_*.json')"
    )
    parser.add_argument(
        "-o", "--output",
        default="galaxy_tools",
        help="Output directory for XML files (default: galaxy_tools)"
    )
    parser.add_argument(
        "--docker",
        default="spac:mvp",
        help="Docker image name (default: spac:mvp)"
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
