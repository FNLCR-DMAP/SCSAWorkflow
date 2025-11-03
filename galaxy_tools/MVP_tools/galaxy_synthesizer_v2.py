#!/usr/bin/env python3
"""
galaxy_synthesizer_v2.py - Generate Galaxy XML tool wrappers from blueprint JSON.

Version 2.0 - No Cheetah needed! Direct parameter passing to format_values.py

This synthesizer reads a template blueprint JSON and generates the corresponding
Galaxy XML file with:
1. Proper repeat blocks for list parameters
2. Boolean parameters with truevalue/falsevalue
3. Direct command that processes raw Galaxy parameters
4. No configfile/Cheetah needed

Usage:
    python galaxy_synthesizer_v2.py template_json_boxplot.json \
        --output-xml boxplot_galaxy.xml \
        --template-path boxplot_template.py
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List
import xml.etree.ElementTree as ET
from xml.dom import minidom


def get_param_info(blueprint: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameter information from blueprint JSON.
    
    Returns dict mapping parameter names to their types and defaults.
    """
    param_info = {}
    
    for param in blueprint.get("parameters", []):
        key = param["key"]
        param_info[key] = {
            "type": param["paramType"],
            "display_name": param["displayName"],
            "description": param.get("description", ""),
            "default": param.get("defaultValue", ""),
            "group": param.get("paramGroup", None)
        }
    
    return param_info


def identify_param_types(param_info: Dict[str, Dict[str, Any]]) -> tuple:
    """
    Identify which parameters are booleans and which are lists.
    
    Returns:
        tuple: (list_of_boolean_params, list_of_list_params)
    """
    bool_params = []
    list_params = []
    
    for param_name, info in param_info.items():
        if info["type"] == "BOOLEAN":
            bool_params.append(param_name)
        elif info["type"] == "LIST":
            list_params.append(param_name)
    
    return bool_params, list_params


def create_galaxy_param(param_name: str, param_info: Dict[str, Any]) -> ET.Element:
    """
    Create a Galaxy parameter XML element based on parameter type.
    """
    param_type = param_info["type"]
    display_name = param_info["display_name"]
    description = param_info["description"]
    default_value = str(param_info["default"])
    
    if param_type == "BOOLEAN":
        # Boolean parameter
        param = ET.Element("param", {
            "name": param_name,
            "type": "boolean",
            "truevalue": "True",
            "falsevalue": "False",
            "checked": "true" if default_value.lower() == "true" else "false",
            "label": display_name
        })
        if description:
            param.set("help", description)
            
    elif param_type == "LIST":
        # Create a repeat block for list parameters
        repeat = ET.Element("repeat", {
            "name": f"{param_name}_repeat",
            "title": display_name,
            "min": "0"
        })
        
        # Add the value parameter inside repeat
        value_param = ET.Element("param", {
            "name": "value",
            "type": "text",
            "label": f"{display_name} item"
        })
        if description:
            value_param.set("help", description)
        
        repeat.append(value_param)
        return repeat
        
    elif param_type == "NUMBER":
        # Numeric parameter
        param = ET.Element("param", {
            "name": param_name,
            "type": "float",
            "value": default_value,
            "label": display_name
        })
        if description:
            param.set("help", description)
            
    else:
        # Default to text parameter
        param = ET.Element("param", {
            "name": param_name,
            "type": "text",
            "value": default_value,
            "label": display_name
        })
        if description:
            param.set("help", description)
    
    return param


def generate_galaxy_xml(
    blueprint: Dict[str, Any],
    template_path: str,
    tool_id: str = None,
    tool_version: str = "1.0.0"
) -> str:
    """
    Generate complete Galaxy XML from blueprint JSON.
    No Cheetah configfile needed - direct parameter processing.
    """
    # Extract tool metadata
    title = blueprint.get("title", "SPAC Tool")
    description = blueprint.get("description", "")
    
    # Clean up description (remove markdown links, etc.)
    if description:
        import re
        description = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', description)
        description = description.replace('\\n', ' ').strip()
    
    # Get parameter information
    param_info = get_param_info(blueprint)
    bool_params, list_params = identify_param_types(param_info)
    
    # Generate tool ID if not provided
    if not tool_id:
        tool_id = title.lower().replace(" ", "_").replace("[", "").replace("]", "")
    
    # Create root tool element
    tool = ET.Element("tool", {
        "id": tool_id,
        "name": title,
        "version": tool_version,
        "profile": "24.2"
    })
    
    # Add description
    desc_elem = ET.SubElement(tool, "description")
    desc_elem.text = description if description else "SPAC Galaxy Tool"
    
    # Add requirements
    requirements = ET.SubElement(tool, "requirements")
    container = ET.SubElement(requirements, "container", {"type": "docker"})
    container.text = "nciccbr/spac:v1"
    
    # Build SIMPLIFIED command without Cheetah
    command_lines = [
        "## Create output directories",
        "mkdir -p dataframe_folder &&",
        "mkdir -p figure_folder &&",
        "",
        "## Save raw Galaxy parameters to JSON",
        "echo '$__json__' > galaxy_params.json &&",
        "",
        'echo "=== Raw Galaxy Parameters ===" &&',
        "cat galaxy_params.json &&",
        'echo "===========================" &&',
        "",
        "## Normalize Galaxy parameters for template consumption",
        "python3 '$__tool_directory__/format_values_v2.py' galaxy_params.json cleaned_params.json \\",
    ]
    
    # Add boolean parameters
    if bool_params:
        bool_args = " ".join(bool_params)
        command_lines.append(f"    --bool-values {bool_args} \\")
    
    # Add list parameters  
    if list_params:
        list_args = " ".join(list_params)
        command_lines.append(f"    --list-values {list_args} \\")
    
    # Remove trailing backslash from last line
    if command_lines[-1].endswith(" \\"):
        command_lines[-1] = command_lines[-1][:-2]
    
    command_lines.extend([
        "    --debug &&",
        "",
        'echo "=== Cleaned Parameters ===" &&',
        "cat cleaned_params.json &&",
        'echo "===========================" &&',
        "",
        "## Add output directories to cleaned params",
        "python3 -c \"",
        "import json",
        "with open('cleaned_params.json', 'r') as f:",
        "    params = json.load(f)",
        "params['outputs'] = {'DataFrames': 'dataframe_folder', 'figures': 'figure_folder'}",
        "with open('cleaned_params.json', 'w') as f:",
        "    json.dump(params, f, indent=2)",
        "\" &&",
        "",
        "## Execute the template with cleaned parameters",
        f"python3 '$__tool_directory__/{template_path}' cleaned_params.json"
    ])
    
    command = ET.SubElement(tool, "command", {"detect_errors": "exit_code"})
    command.text = "\n        ".join(command_lines)
    
    # Add inputs section (NO configfiles section needed!)
    inputs = ET.SubElement(tool, "inputs")
    
    # Add Upstream_Analysis as data input
    if "Upstream_Analysis" in param_info:
        upstream_param = ET.SubElement(inputs, "param", {
            "name": "Upstream_Analysis",
            "type": "data",
            "format": "h5ad,binary",
            "label": param_info["Upstream_Analysis"]["display_name"]
        })
    
    # Add other parameters
    for param_name, info in param_info.items():
        if param_name == "Upstream_Analysis":
            continue
        
        param_elem = create_galaxy_param(param_name, info)
        inputs.append(param_elem)
    
    # Add outputs section
    outputs = ET.SubElement(tool, "outputs")
    
    # Raw Galaxy parameters for debugging
    ET.SubElement(outputs, "data", {
        "name": "galaxy_params",
        "format": "json",
        "from_work_dir": "galaxy_params.json",
        "label": "Raw Galaxy Parameters"
    })
    
    # Cleaned parameters
    ET.SubElement(outputs, "data", {
        "name": "cleaned_params",
        "format": "json",
        "from_work_dir": "cleaned_params.json",
        "label": "Cleaned Parameters"
    })
    
    # DataFrames collection
    df_collection = ET.SubElement(outputs, "collection", {
        "name": "dataframes",
        "type": "list",
        "label": "DataFrames"
    })
    ET.SubElement(df_collection, "discover_datasets", {
        "pattern": "__name_and_ext__",
        "directory": "dataframe_folder"
    })
    
    # Figures collection
    fig_collection = ET.SubElement(outputs, "collection", {
        "name": "figures",
        "type": "list",
        "label": "Figures"
    })
    ET.SubElement(fig_collection, "discover_datasets", {
        "pattern": "__name_and_ext__",
        "directory": "figure_folder"
    })
    
    # Convert to pretty-printed XML string
    xml_str = ET.tostring(tool, encoding='unicode')
    
    # Pretty print with proper indentation
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="    ")
    
    # Clean up extra blank lines
    lines = pretty_xml.split('\n')
    clean_lines = [line for line in lines if line.strip()]
    
    return '\n'.join(clean_lines[1:])  # Skip XML declaration


def main():
    """Main entry point for the synthesizer."""
    parser = argparse.ArgumentParser(
        description="Generate Galaxy XML from template blueprint JSON (No Cheetah version)"
    )
    parser.add_argument(
        "blueprint_json",
        help="Path to template blueprint JSON file"
    )
    parser.add_argument(
        "--output-xml",
        default=None,
        help="Output XML file path (default: derived from blueprint title)"
    )
    parser.add_argument(
        "--template-path",
        default="template.py",
        help="Path to the Python template file (default: template.py)"
    )
    parser.add_argument(
        "--tool-id",
        default=None,
        help="Galaxy tool ID (default: derived from title)"
    )
    parser.add_argument(
        "--tool-version",
        default="1.0.0",
        help="Tool version (default: 1.0.0)"
    )
    
    args = parser.parse_args()
    
    # Read blueprint JSON
    blueprint_path = Path(args.blueprint_json)
    if not blueprint_path.exists():
        print(f"Error: Blueprint file '{blueprint_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    with open(blueprint_path, 'r') as f:
        blueprint = json.load(f)
    
    # Generate XML
    xml_content = generate_galaxy_xml(
        blueprint,
        template_path=args.template_path,
        tool_id=args.tool_id,
        tool_version=args.tool_version
    )
    
    # Determine output path
    if args.output_xml:
        output_path = Path(args.output_xml)
    else:
        # Derive from blueprint title
        title = blueprint.get("title", "tool")
        filename = title.lower().replace(" ", "_").replace("[", "").replace("]", "")
        output_path = Path(f"{filename}.xml")
    
    # Write XML file
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"Successfully generated Galaxy XML: {output_path}")
    print("NO CHEETAH NEEDED - Direct parameter processing!")
    
    # Report parameter types identified
    param_info = get_param_info(blueprint)
    bool_params, list_params = identify_param_types(param_info)
    
    print(f"\nIdentified parameter types:")
    print(f"  Boolean parameters: {', '.join(bool_params) if bool_params else 'None'}")
    print(f"  List parameters: {', '.join(list_params) if list_params else 'None'}")
    print(f"\nThe Galaxy tool will use format_values_v2.py to process raw Galaxy JSON directly.")


if __name__ == "__main__":
    main()
