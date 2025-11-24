#!/usr/bin/env python3
"""
Galaxy XML Synthesizer - Generates Galaxy tool XML from blueprint JSON files.

Version: 2.4 - Integrated sanitizer and section support
Features:
- Automatic section creation from paramGroup
- Sanitizer support for special characters
- Proper type mapping (INTEGER->integer, NUMBER->float, etc.)
- Fixed commands with proper && chaining

This synthesizer properly handles all parameter types with sanitizer support:
- Numeric types with min/max bounds → Galaxy numeric types
- Boolean parameters → Galaxy boolean checkboxes  
- SELECT dropdowns → Galaxy select with options
- LIST parameters → Galaxy repeat structures with sanitizer support
- Multi-value columns → Galaxy repeat structures
- String parameters → Galaxy text inputs

Author: FNLCR-DMAP Team
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import argparse
import sys
from collections import OrderedDict


class GalaxyXMLSynthesizer:
    """
    Synthesize Galaxy XML from blueprint JSON with sanitizer and section support.
    Handles special characters in parameters and automatic section grouping.
    """
    
    # Parameters that need special sanitizer configuration
    SANITIZER_CONFIG = {
        'Anchor_Neighbor_List': {
            'allowed_chars': [';', '+', '/', '[', ']', '{', '}', '<', '>', '=', "'"],
            'description': 'Allows semicolon separator and biological nomenclature characters'
        },
        # Add other parameters that need special characters here
    }
    
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
        
        # Inputs with section support
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
    
    def _add_sanitizer(self, param: ET.Element, param_key: str, 
                      custom_config: Optional[Dict[str, Any]] = None):
        """
        Add sanitizer configuration to a parameter.
        
        Parameters
        ----------
        param : ET.Element
            The parameter element to add sanitizer to
        param_key : str
            The parameter key to check for special sanitizer needs
        custom_config : dict, optional
            Custom sanitizer configuration from blueprint
        """
        # Check if this parameter needs special sanitizer configuration
        config = custom_config or self.SANITIZER_CONFIG.get(param_key)
        
        if config:
            sanitizer = ET.SubElement(param, 'sanitizer')
            valid = ET.SubElement(sanitizer, 'valid')
            valid.set('initial', 'default')
            
            # Add allowed special characters
            allowed_chars = config.get('allowed_chars', [])
            for char in allowed_chars:
                add_elem = ET.SubElement(valid, 'add')
                # Handle special XML characters
                if char == '<':
                    add_elem.set('value', '&lt;')
                elif char == '>':
                    add_elem.set('value', '&gt;')
                elif char == '&':
                    add_elem.set('value', '&amp;')
                else:
                    add_elem.set('value', char)
        
        # Also check if parameter description mentions semicolon separator
        elif param_key and ';' in param.get('help', ''):
            # Auto-detect need for semicolon sanitizer
            sanitizer = ET.SubElement(param, 'sanitizer')
            valid = ET.SubElement(sanitizer, 'valid')
            valid.set('initial', 'default')
            add_elem = ET.SubElement(valid, 'add')
            add_elem.set('value', ';')
    
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
        command.text = f"<![CDATA[\n\n{full_command}\n\n]]>"
    
    def _group_parameters(self) -> Dict[Optional[str], List[Dict]]:
        """Group parameters by their paramGroup field."""
        groups = OrderedDict()
        
        # Process parameters maintaining order
        for param_def in self.blueprint.get('parameters', []):
            group = param_def.get('paramGroup')
            if group not in groups:
                groups[group] = []
            groups[group].append(param_def)
        
        return groups
    
    def _make_section_id(self, group_name: str) -> str:
        """Convert group name to valid section ID."""
        # Convert to lowercase and replace spaces with underscores
        section_id = group_name.lower().replace(' ', '_')
        # Remove special characters
        section_id = re.sub(r'[^a-z0-9_]', '', section_id)
        return section_id
    
    def _get_galaxy_param_type(self, param_type: str) -> str:
        """Map blueprint paramType to Galaxy param type."""
        type_mapping = {
            'BOOLEAN': 'boolean',
            'INTEGER': 'integer',
            'INT': 'integer',
            'Positive integer': 'integer',  # Special case
            'NUMBER': 'float',
            'FLOAT': 'float',
            'SELECT': 'select',
            'STRING': 'text',
            'TEXT': 'text',
            'FILE': 'data',
            'LIST': 'repeat',  # Special case, handled separately
        }
        return type_mapping.get(param_type, 'text')
    
    def _add_inputs(self, tool: ET.Element):
        """
        Add inputs section with automatic section grouping.
        
        Follows supervisor's logic:
        1. Parameters without groups should be listed first
        2. Then parameters within groups
        3. Groups should be ordered based on their first appearance in orderedMustacheKeys
        """
        inputs = ET.SubElement(tool, 'inputs')
        
        # Build lookup dictionaries for all parameter types
        datasets_dict = {d['key']: d for d in self.blueprint.get('inputDatasets', [])}
        params_dict = {p['key']: p for p in self.blueprint.get('parameters', [])}
        columns_dict = {c['key']: c for c in self.blueprint.get('columns', [])}
        
        # Get ordered keys or fall back to natural order
        ordered_keys = self.blueprint.get('orderedMustacheKeys', [])
        
        if not ordered_keys:
            # Fall back to original order if no orderedMustacheKeys
            # First datasets, then params, then columns
            ordered_keys = []
            ordered_keys.extend([d['key'] for d in self.blueprint.get('inputDatasets', [])])
            ordered_keys.extend([p['key'] for p in self.blueprint.get('parameters', [])])
            ordered_keys.extend([c['key'] for c in self.blueprint.get('columns', [])])
        
        # Track what we've processed
        processed_keys = set()
        
        # Collect ungrouped and grouped parameters while maintaining order
        ungrouped_params = []
        sections = OrderedDict()  # Will maintain order of first appearance
        section_first_index = {}  # Track first appearance index for each section
        
        # Process in orderedMustacheKeys order
        for idx, key in enumerate(ordered_keys):
            if key in processed_keys:
                continue
                
            # Find the item
            item = None
            item_type = None
            
            if key in datasets_dict:
                item = datasets_dict[key]
                item_type = 'dataset'
            elif key in params_dict:
                item = params_dict[key]
                item_type = 'param'
            elif key in columns_dict:
                item = columns_dict[key]
                item_type = 'column'
            
            if not item:
                continue  # Skip unknown keys
            
            processed_keys.add(key)
            
            # Input datasets always go first (no sections)
            if item_type == 'dataset':
                self._add_dataset_param(inputs, item)
            else:
                # Check if it has a paramGroup
                group = item.get('paramGroup')
                
                if not group:
                    # No group - add to ungrouped list
                    ungrouped_params.append((item, item_type))
                else:
                    # Has a group - add to appropriate section
                    if group not in sections:
                        sections[group] = []
                        section_first_index[group] = idx  # Track first appearance
                    sections[group].append((item, item_type))
        
        # Add ungrouped parameters (after datasets, before sections)
        for item, item_type in ungrouped_params:
            if item_type == 'param':
                self._add_parameter(inputs, item)
            elif item_type == 'column':
                self._add_column(inputs, item)
        
        # Sort sections by their first appearance in orderedMustacheKeys
        sorted_sections = sorted(sections.items(), 
                               key=lambda x: section_first_index.get(x[0], float('inf')))
        
        # Add grouped parameters in sections
        for group_name, items in sorted_sections:
            # Create section for this group
            section = ET.SubElement(inputs, 'section')
            section_id = self._make_section_id(group_name)
            section.set('name', section_id)
            section.set('title', group_name)
            section.set('expanded', 'false')
            
            # Add parameters to section
            for item, item_type in items:
                if item_type == 'param':
                    self._add_parameter(section, item)
                elif item_type == 'column':
                    self._add_column(section, item)
    
    def _add_dataset_param(self, parent: ET.Element, dataset: Dict[str, Any]):
        """Add dataset parameter."""
        param = ET.SubElement(parent, 'param')
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
    
    def _add_parameter(self, parent: ET.Element, param_def: Dict[str, Any]):
        """Add parameter based on type with proper datatype handling and sanitizer support."""
        param_type = param_def.get('paramType', 'STRING')
        param_key = param_def.get('key')
        display_name = param_def.get('displayName', param_key)
        description = param_def.get('description', '')
        default_value = param_def.get('defaultValue')
        
        # Check for custom sanitizer config in parameter definition
        sanitizer_config = param_def.get('sanitizer')
        
        if param_type == 'LIST':
            # Use Galaxy repeat for lists
            repeat = ET.SubElement(parent, 'repeat')
            repeat.set('name', f"{param_key}_repeat")
            repeat.set('title', display_name)
            
            param = ET.SubElement(repeat, 'param')
            param.set('name', 'value')
            param.set('type', 'text')
            param.set('label', 'Value')
            param.set('help', description)
            
            # Add sanitizer for LIST parameters that need special characters
            self._add_sanitizer(param, param_key, sanitizer_config)
            
            # Add default values if specified
            if default_value:
                if isinstance(default_value, str):
                    # Parse string that looks like a list
                    if default_value.startswith('[') and default_value.endswith(']'):
                        try:
                            default_list = json.loads(default_value)
                            if default_list and len(default_list) > 0:
                                param.set('value', str(default_list[0]))
                        except:
                            param.set('value', default_value)
                    else:
                        param.set('value', default_value)
                elif isinstance(default_value, list) and default_value:
                    param.set('value', str(default_value[0]))
        
        elif param_type == 'BOOLEAN':
            param = ET.SubElement(parent, 'param')
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
            param = ET.SubElement(parent, 'param')
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
            param = ET.SubElement(parent, 'param')
            param.set('name', param_key)
            
            # Map blueprint types to Galaxy types
            galaxy_type = self._get_galaxy_param_type(param_type)
            param.set('type', galaxy_type)
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
                if galaxy_type == 'integer':
                    param.set('value', '0')
                elif galaxy_type == 'float':
                    param.set('value', '0.0')
            
            if description:
                param.set('help', description)
        
        else:  # STRING and other text types
            param = ET.SubElement(parent, 'param')
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
            
            # Check if this STRING parameter needs special sanitizer
            self._add_sanitizer(param, param_key, sanitizer_config)
    
    def _add_column(self, parent: ET.Element, column: Dict[str, Any]):
        """Add column parameter - using repeat for multi-value columns with sanitizer support."""
        param_key = column.get('key')
        display_name = column.get('displayName', param_key)
        description = column.get('description', '')
        is_multi = column.get('isMulti', False)
        sanitizer_config = column.get('sanitizer')
        
        if is_multi:
            # Multi-value column: use Galaxy repeat structure (like LIST params)
            repeat = ET.SubElement(parent, 'repeat')
            repeat.set('name', f"{param_key}_repeat")
            repeat.set('title', display_name)
            
            param = ET.SubElement(repeat, 'param')
            param.set('name', 'value')
            param.set('type', 'text')
            param.set('label', 'Value')
            param.set('help', description)
            
            # Add sanitizer if needed
            self._add_sanitizer(param, param_key, sanitizer_config)
            
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
            param = ET.SubElement(parent, 'param')
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
            
            # Add sanitizer if needed
            self._add_sanitizer(param, param_key, sanitizer_config)
    
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
        """Generate help text with sanitizer info for special parameters."""
        title = self.blueprint.get('title', 'Tool')
        desc = self._clean_text(self.blueprint.get('description', ''))
        
        help_lines = [
            f"**{title}**",
            desc
        ]
        
        # Check if we have parameters that need special characters
        has_special_params = False
        for param in self.blueprint.get('parameters', []):
            if param.get('key') in self.SANITIZER_CONFIG:
                has_special_params = True
                break
        
        if has_special_params:
            help_lines.extend([
                "",
                "**Special Characters Allowed:**",
                ""
            ])
            
            # Document Anchor_Neighbor_List specifically
            if 'Anchor_Neighbor_List' in [p.get('key') for p in self.blueprint.get('parameters', [])]:
                help_lines.extend([
                    "**Anchor Neighbor List Format:**",
                    "Enter anchor and neighbor cell types separated by semicolon (;)",
                    "Examples:",
                    "- T cells; B cells",
                    "- CD4+; CD8+",
                    "- FOXP3+/CD25+; Tregs",
                    "- PD-1high; PD-L1+",
                    "",
                    "The following special characters are allowed:",
                    "- Semicolon (;) - as separator between anchor and neighbor",
                    "- Plus (+) - for positive markers (CD4+, FOXP3+)",
                    "- Slash (/) - for combinations (CD4/CD8, FOXP3/CD25)",
                    "- Brackets [] {} - for concentrations or sets",
                    "- Comparison operators (<, >, =)",
                    "- Apostrophe (') - for 5' or 3' notation",
                    ""
                ])
        
        help_lines.extend([
            "**Outputs:**"
        ])
        
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
    
    # Report on features used
    features = []
    
    # Check if sanitizer was added
    if 'Anchor_Neighbor_List' in xml_content and '<sanitizer>' in xml_content:
        features.append("sanitizer")
    
    # Check if sections were added
    if '<section' in xml_content:
        features.append("sections")
    
    # Check for numeric types
    if 'type="integer"' in xml_content:
        features.append("integer params")
    if 'type="float"' in xml_content:
        features.append("float params")
    
    if features:
        print(f"  -> Features: {', '.join(features)}")
    
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
    feature_summary = {
        'sanitizer': [],
        'sections': [],
        'integer': [],
        'float': []
    }
    
    for blueprint_file in sorted(blueprint_files):
        try:
            xml_file = process_blueprint(blueprint_file, output_path, docker_image)
            generated_files.append(xml_file)
            
            # Check features in generated XML
            with open(xml_file, 'r') as f:
                content = f.read()
                if '<sanitizer>' in content:
                    feature_summary['sanitizer'].append(blueprint_file.stem)
                if '<section' in content:
                    feature_summary['sections'].append(blueprint_file.stem)
                if 'type="integer"' in content:
                    feature_summary['integer'].append(blueprint_file.stem)
                if 'type="float"' in content:
                    feature_summary['float'].append(blueprint_file.stem)
                    
        except Exception as e:
            print(f"  ERROR processing {blueprint_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Successfully generated {len(generated_files)} Galaxy XML files")
    
    # Print feature summary
    if any(feature_summary.values()):
        print("\nFeature Summary:")
        if feature_summary['sections']:
            print(f"  Tools with sections: {len(feature_summary['sections'])}")
        if feature_summary['sanitizer']:
            print(f"  Tools with sanitizers: {len(feature_summary['sanitizer'])}")
        if feature_summary['integer']:
            print(f"  Tools with integer params: {len(feature_summary['integer'])}")
        if feature_summary['float']:
            print(f"  Tools with float params: {len(feature_summary['float'])}")
    
    return 0


def main():
    """Main entry point for the synthesizer."""
    parser = argparse.ArgumentParser(
        description="Generate Galaxy tool XML from SPAC blueprint JSON files with sanitizer and section support"
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
