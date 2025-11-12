#!/usr/bin/env python3
"""
Generalized Galaxy XML Synthesizer
Blueprint JSON drives ALL decisions - no hardcoding, no special cases
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
        """Add command section."""
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
        
        # Build command
        template_name = tool_id.replace('spac_', '') + '_template.py'
        
        cmd_lines = [
            "python '$__tool_directory__/format_values.py'",
            "    'galaxy_params.json'",
            "    'cleaned_params.json'"
        ]
        
        if bool_params:
            cmd_lines.append(f"    --bool-values {' '.join(bool_params)}")
        if list_params:
            cmd_lines.append(f"    --list-values {' '.join(list_params)}")
        
        # Add output injection from blueprint
        if 'outputs' in self.blueprint:
            cmd_lines.append("    --inject-outputs")
            # Save outputs config to file for format_values.py to read
            cmd_lines.insert(0, f"echo '{json.dumps(self.blueprint['outputs'])}' > outputs_config.json")
            cmd_lines.append("    --outputs-config outputs_config.json")
        
        cmd_lines.extend([
            "",
            "&&",
            "",
            f"python '$__tool_directory__/{template_name}' 'cleaned_params.json'"
        ])
        
        command = ET.SubElement(tool, 'command')
        command.set('detect_errors', 'exit_code')
        command.text = f"<![CDATA[\n{chr(10).join(cmd_lines)}\n]]>"
    
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
        """Add parameter based on type."""
        param_type = param_def.get('paramType', 'STRING')
        param_key = param_def.get('key')
        display_name = param_def.get('displayName', param_key)
        description = param_def.get('description', '')
        default_value = param_def.get('defaultValue')
        
        if param_type == 'LIST':
            # Use repeat for explicit list parameters
            repeat = ET.SubElement(inputs, 'repeat')
            repeat.set('name', f"{param_key}_repeat")
            repeat.set('title', display_name)
            repeat.set('min', '0')
            if description:
                repeat.set('help', description)
            
            inner = ET.SubElement(repeat, 'param')
            inner.set('name', 'value')
            inner.set('type', 'text')
            inner.set('label', 'Value')
            if default_value not in [None, 'None']:
                inner.set('value', str(default_value))
                
        elif param_type == 'BOOLEAN':
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            param.set('type', 'boolean')
            param.set('label', display_name)
            param.set('truevalue', 'True')
            param.set('falsevalue', 'False')
            param.set('checked', 'true' if default_value else 'false')
            if description:
                param.set('help', description)
                
        elif param_type in ['NUMBER', 'INTEGER', 'FLOAT']:
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            param.set('type', 'float' if param_type in ['NUMBER', 'FLOAT'] else 'integer')
            param.set('label', display_name)
            if default_value not in [None, 'None']:
                param.set('value', str(default_value))
            else:
                param.set('optional', 'true')
            if description:
                param.set('help', description)
                
        else:  # STRING or others
            param = ET.SubElement(inputs, 'param')
            param.set('name', param_key)
            param.set('type', 'text')
            param.set('label', display_name)
            if default_value not in [None, 'None']:
                param.set('value', str(default_value))
            else:
                param.set('optional', 'true')
            if description:
                param.set('help', description)
    
    def _add_column(self, inputs: ET.Element, column: Dict[str, Any]):
        """Add column parameter - consistent handling with text area for multi."""
        param = ET.SubElement(inputs, 'param')
        param.set('name', column['key'])
        param.set('type', 'text')
        param.set('label', column.get('displayName', column['key']))
        
        if column.get('isMulti'):
            # Multi-column: use text area
            param.set('area', 'true')
            help_text = column.get('description', '')
            help_text += ' (Enter one column name per line)'
            param.set('help', help_text)
        else:
            # Single column: regular text input
            if column.get('description'):
                param.set('help', column['description'])
            default = column.get('defaultValue')
            if default:
                param.set('value', str(default))
        
        param.set('optional', 'true')
    
    def _add_outputs(self, tool: ET.Element):
        """Add outputs - DIRECTLY from blueprint, no inference."""
        outputs_elem = ET.SubElement(tool, 'outputs')
        
        # Debug outputs
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
        """Format XML with proper indentation."""
        rough = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough)
        pretty = reparsed.toprettyxml(indent="    ")
        # Remove blank lines
        lines = [line for line in pretty.split('\n') if line.strip()]
        return '\n'.join(lines)


def batch_convert(input_dir: Path, output_dir: Path, docker_image: str = "spac:latest") -> int:
    """
    Batch convert all blueprint JSON files to Galaxy XML.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing template_json_*.json files
    output_dir : Path
        Directory to save generated XML files
    docker_image : str
        Docker image name for tools
        
    Returns
    -------
    int
        Number of tools converted
    """
    # Find all blueprint JSON files
    json_files = list(input_dir.glob("template_json_*.json"))
    
    if not json_files:
        print(f"No template_json_*.json files found in {input_dir}")
        return 0
    
    print(f"Found {len(json_files)} blueprint JSON files")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    for json_file in sorted(json_files):
        try:
            # Load blueprint
            with open(json_file, 'r') as f:
                blueprint = json.load(f)
            
            # Generate XML
            synthesizer = GalaxyXMLSynthesizer(blueprint, docker_image)
            xml_content = synthesizer.synthesize()
            
            # Determine output filename
            tool_name = json_file.stem.replace('template_json_', '')
            xml_file = output_dir / f"{tool_name}.xml"
            
            # Save XML
            xml_file.write_text(xml_content)
            
            print(f"✓ {tool_name}.xml")
            converted += 1
            
        except Exception as e:
            print(f"✗ Failed to convert {json_file.name}: {e}")
    
    return converted


def main():
    """Main entry point with CLI."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate Galaxy tool XML from SPAC blueprint JSON files"
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing template_json_*.json files"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("galaxy_tools"),
        help="Output directory for Galaxy XML files (default: galaxy_tools)"
    )
    
    parser.add_argument(
        "--docker-image",
        default="spac:latest",
        help="Docker image name for tools (default: spac:latest)"
    )
    
    parser.add_argument(
        "--single",
        type=Path,
        help="Convert single JSON file instead of batch"
    )
    
    args = parser.parse_args()
    
    if args.single:
        # Single file conversion
        if not args.single.exists():
            print(f"Error: File not found: {args.single}")
            return 1
        
        with open(args.single, 'r') as f:
            blueprint = json.load(f)
        
        synthesizer = GalaxyXMLSynthesizer(blueprint, args.docker_image)
        xml_content = synthesizer.synthesize()
        
        output_file = args.output_dir / f"{args.single.stem.replace('template_json_', '')}.xml"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(xml_content)
        
        print(f"Generated: {output_file}")
        return 0
    
    else:
        # Batch conversion
        if not args.input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}")
            return 1
        
        converted = batch_convert(args.input_dir, args.output_dir, args.docker_image)
        
        print(f"\nConverted {converted} tools")
        print(f"Output directory: {args.output_dir}")
        
        return 0 if converted > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
