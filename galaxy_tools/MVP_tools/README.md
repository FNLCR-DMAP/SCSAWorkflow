# MVP_tools: Blueprint to Galaxy Tool Pipeline

This guide walks you through converting NIDAP blueprint JSON files into fully functional Galaxy tools. It covers:

1. **What changes are needed** in the blueprint to make it Galaxy-ready
2. **How to run the synthesizer** to generate Galaxy XML
3. **What `format_values.py` does** to normalize parameters at runtime

This documentation is intended for developers who want to generate Galaxy tools from blueprints, including adapting this pipeline for other projects (e.g., CCBR).

---

## Table of Contents

- [Overview: The Pipeline](#overview-the-pipeline)
- [Step 1: Preparing Your Blueprint](#step-1-preparing-your-blueprint)
  - [The Critical `outputs` Section](#the-critical-outputs-section)
  - [Output Types Explained](#output-types-explained)
  - [Complete Blueprint Example](#complete-blueprint-example)
- [Step 2: Running the Synthesizer](#step-2-running-the-synthesizer)
  - [Basic Usage](#basic-usage)
  - [Batch Processing](#batch-processing)
  - [Command Line Options](#command-line-options)
- [Step 3: Understanding format_values.py](#step-3-understanding-format_valuespy)
  - [Why format_values.py Is Needed](#why-format_valuespy-is-needed)
  - [Raw vs Cleaned JSON](#raw-vs-cleaned-json)
  - [What format_values.py Expects](#what-format_valuespy-expects)
  - [What format_values.py Outputs](#what-format_valuespy-outputs)
  - [Adapting for Your Project](#adapting-for-your-project)
- [How It All Fits Together](#how-it-all-fits-together)
- [Directory Structure](#directory-structure)
- [Parameter Type Mapping](#parameter-type-mapping)
- [Available Tools](#available-tools)
- [Troubleshooting](#troubleshooting)

---

## Overview: The Pipeline

The pipeline converts a NIDAP blueprint into a Galaxy tool through these steps:

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Blueprint JSON │ ──▶ │ galaxy_xml_synthesizer│ ──▶│  Galaxy XML     │
│  (with outputs) │     │        .py            │    │  (tool def)     │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
                        ┌──────────────────────────────────────────────────┐
                        │              At Runtime in Galaxy:               │
                        │                                                  │
                        │  galaxy_params.json ──▶ format_values.py ──▶     │
                        │  cleaned_params.json ──▶ template.py ──▶ outputs │
                        └──────────────────────────────────────────────────┘
```

**Key components:**

| Component | Purpose |
|-----------|---------|
| `blueprint JSON` | Defines tool parameters, inputs, and **outputs** |
| `galaxy_xml_synthesizer.py` | Converts blueprint → Galaxy XML |
| `format_values.py` | Normalizes Galaxy's raw JSON → clean JSON for templates |
| `*_template.py` | Python script that performs the actual analysis |

---

## Step 1: Preparing Your Blueprint

The main change required to make a NIDAP blueprint Galaxy-ready is **adding an `outputs` section**. This tells the synthesizer what files/directories the tool will produce.

### The Critical `outputs` Section

**Original NIDAP blueprints do NOT have an `outputs` section.** You must add one to specify what the tool generates:

```json
{
  "title": "My Tool [SPAC]",
  "description": "...",
  "inputDatasets": [...],
  "parameters": [...],
  
  "outputs": {
    "analysis": {
      "type": "file",
      "name": "analysis_output.pickle"
    },
    "figures": {
      "type": "directory",
      "name": "figures_dir"
    },
    "dataframe": {
      "type": "file",
      "name": "summary.csv"
    }
  }
}
```

### Output Types Explained

The `outputs` section supports different output types based on what your tool produces:

| Output Type | When to Use | Example |
|-------------|-------------|---------|
| **AnnData/Analysis Object** | Tool produces a processed dataset | `"analysis": {"type": "file", "name": "output.pickle"}` |
| **Figures (PNG/PDF)** | Tool generates one or more plots | `"figures": {"type": "directory", "name": "figures_dir"}` |
| **HTML Figures** | Tool generates interactive HTML plots | `"html": {"type": "directory", "name": "html_dir"}` |
| **DataFrame/CSV** | Tool produces tabular results | `"dataframe": {"type": "file", "name": "results.csv"}` |

**Important:** Use `"type": "directory"` when the tool may produce multiple files (e.g., multiple figures). Use `"type": "file"` for single output files.

### Complete Blueprint Example

Here's a complete blueprint with the `outputs` section added (changes highlighted):

```json
{
  "rid": "ri.vector.main.template.266414a4-406e-4c90-938b-893932beca90",
  "title": "Boxplot [SPAC] [DMAP]",
  "description": "Create a boxplot visualization of the features...",
  
  "inputDatasets": [
    {
      "key": "Upstream_Analysis",
      "displayName": "Upstream Analysis",
      "description": "Link to prior processed dataset",
      "dataType": "PYTHON_TRANSFORM_INPUT"
    }
  ],
  
  "parameters": [
    {
      "key": "Feature_s_to_Plot",
      "displayName": "Feature(s) to Plot",
      "paramType": "LIST",
      "defaultValue": "All"
    },
    {
      "key": "Horizontal_Plot",
      "displayName": "Horizontal Plot",
      "paramType": "BOOLEAN",
      "paramGroup": "Figure Configuration",
      "defaultValue": "False"
    },
    {
      "key": "Figure_Width",
      "displayName": "Figure Width",
      "paramType": "NUMBER",
      "paramGroup": "Figure Configuration",
      "defaultValue": "12"
    }
  ],
  
  "outputs": {
    "dataframe": {
      "type": "file",
      "name": "dataframe.csv"
    },
    "figures": {
      "type": "directory",
      "name": "figures_dir"
    }
  }
}
```

---

## Step 2: Running the Synthesizer

Once your blueprint has the `outputs` section, generate the Galaxy XML.

### Basic Usage

```bash
# Navigate to MVP_tools directory
cd galaxy_tools/MVP_tools

# Generate XML from a single blueprint
python galaxy_xml_synthesizer.py blueprint_json/template_json_boxplot.json -o .
```

This creates `boxplot.xml` in the current directory.

### Batch Processing

```bash
# Process all blueprints in a directory
python galaxy_xml_synthesizer.py blueprint_json/ -o .

# Or use a glob pattern
python galaxy_xml_synthesizer.py "blueprint_json/template_json_*.json" -o .
```

### Command Line Options

```
usage: galaxy_xml_synthesizer.py [-h] [-o OUTPUT] [--docker DOCKER] [--debug] blueprint

positional arguments:
  blueprint             Path to blueprint JSON file or directory

optional arguments:
  -h, --help            Show help message
  -o, --output OUTPUT   Output directory for XML files (default: galaxy_tools)
  --docker DOCKER       Docker image name (default: spac:mvp)
  --debug               Enable debug output with feature summary
```

### What the Synthesizer Generates

The generated XML includes:

1. **Input parameters** mapped from blueprint `parameters` array
2. **Sections** automatically created from `paramGroup` fields
3. **Output definitions** from your `outputs` section
4. **Command section** that chains `format_values.py` → `template.py`

Example command section in generated XML:

```xml
<command detect_errors="exit_code"><![CDATA[

echo '{"dataframe": {"type": "file", "name": "dataframe.csv"}, ...}' > outputs_config.json && \
python '$__tool_directory__/format_values.py' 'galaxy_params.json' 'cleaned_params.json' \
    --bool-values Horizontal_Plot Keep_Outliers Value_Axis_Log_Scale \
    --list-values Feature_s_to_Plot \
    --inject-outputs --outputs-config outputs_config.json && \
python '$__tool_directory__/boxplot_template.py' 'cleaned_params.json'

]]></command>
```

---

## Step 3: Understanding format_values.py

`format_values.py` is a **critical runtime component** that transforms Galaxy's raw parameter JSON into a clean format that Python templates can consume directly.

### Why format_values.py Is Needed

Galaxy serializes user inputs in a specific JSON structure that doesn't match what Python templates expect:

| Problem | Galaxy Raw Format | What Templates Need |
|---------|-------------------|---------------------|
| **Booleans** | `"True"` (string) | `true` (boolean) |
| **Lists (repeat)** | `[{"value": "CD4"}, {"value": "CD8"}]` | `["CD4", "CD8"]` |

Without `format_values.py`, every template would need custom parsing logic.

### Raw vs Cleaned JSON

**Raw Galaxy JSON (`galaxy_params.json`):**

```json
{
  "Upstream_Analysis": "/path/to/data.pickle",
  "Feature_s_to_Plot_repeat": [
    {"value": "CD4"},
    {"value": "CD8"},
    {"value": "CD20"}
  ],
  "Horizontal_Plot": "True",
  "Keep_Outliers": "False",
  "Figure_Width": "12"
}
```

**Cleaned JSON (`cleaned_params.json`):**

```json
{
  "Upstream_Analysis": "/path/to/data.pickle",
  "Feature_s_to_Plot": ["CD4", "CD8", "CD20"],
  "Horizontal_Plot": true,
  "Keep_Outliers": false,
  "Figure_Width": "12",
  "outputs": {
    "dataframe": {"type": "file", "name": "dataframe.csv"},
    "figures": {"type": "directory", "name": "figures_dir"}
  },
  "save_results": true
}
```

### What format_values.py Expects

**Input:** A JSON file with Galaxy's raw parameter structure

**Command-line flags:**

| Flag | Purpose | Example |
|------|---------|---------|
| `--bool-values` | Parameter names to convert to Python booleans | `--bool-values Horizontal_Plot Keep_Outliers` |
| `--list-values` | Parameter names to extract from repeat structures | `--list-values Feature_s_to_Plot` |
| `--list-sep` | Separator for delimited text fields | `--list-sep '\n'` |
| `--list-fields` | Fields to parse with the separator | `--list-fields Anchor_Neighbor_List` |
| `--inject-outputs` | Add outputs config to cleaned JSON | `--inject-outputs` |
| `--outputs-config` | Path to outputs configuration JSON | `--outputs-config outputs_config.json` |

### What format_values.py Outputs

A cleaned JSON file where:

1. **Boolean strings** → Python booleans (`"True"` → `true`)
2. **Repeat structures** → Simple lists (`[{"value": "x"}]` → `["x"]`)
3. **Output configuration** → Injected into params (if `--inject-outputs`)
4. **All other parameters** → Passed through unchanged

### Adapting for Your Project

`format_values.py` is **template-agnostic** with no hardcoded parameter names. To adapt it for another project (e.g., CCBR):

1. **Use as-is** if your templates expect the same clean format
2. **Modify the boolean/list flags** in the synthesizer if your parameters have different names
3. **Add custom transformations** by extending the `process_galaxy_params()` function

**Example: Adding a custom transformation**

```python
# In format_values.py, add to process_galaxy_params():

# Custom: Convert comma-separated string to list
if 'my_custom_param' in params:
    value = params['my_custom_param']
    if isinstance(value, str):
        cleaned['my_custom_param'] = [x.strip() for x in value.split(',')]
```

---

## How It All Fits Together

Here's the complete flow from blueprint to Galaxy execution:

```
1. PREPARATION (one-time)
   ├── Add `outputs` section to blueprint JSON
   └── Run: python galaxy_xml_synthesizer.py blueprint.json -o .
            ↓
       Creates: tool.xml + references format_values.py & template.py

2. GALAXY RUNTIME (each execution)
   ├── User fills in Galaxy form
   ├── Galaxy creates: galaxy_params.json (raw format)
   │
   ├── format_values.py runs:
   │   ├── Input:  galaxy_params.json
   │   ├── Fixes:  booleans, lists, injects outputs
   │   └── Output: cleaned_params.json
   │
   └── template.py runs:
       ├── Input:  cleaned_params.json
       ├── Executes analysis logic
       └── Output: files defined in outputs section
```

---

## Directory Structure

```
MVP_tools/
├── galaxy_xml_synthesizer.py    # Converts blueprint → Galaxy XML
├── format_values.py             # Runtime: normalizes Galaxy params
├── template_utils.py            # Shared utilities for templates
│
├── blueprint_json/              # Input: NIDAP blueprints (with outputs added)
│   ├── template_json_boxplot.json
│   ├── template_json_arcsinh_normalization.json
│   └── ...
│
├── boxplot_template.py          # Python analysis template
├── boxplot.xml                  # Generated Galaxy XML
└── ...
```

---

## Parameter Type Mapping

The synthesizer maps blueprint parameter types to Galaxy types:

| Blueprint Type | Galaxy Type | Notes |
|---------------|-------------|-------|
| `STRING` | `text` | Basic text input |
| `INTEGER` | `integer` | Integer with optional min/max |
| `NUMBER` / `FLOAT` | `float` | Floating point with optional bounds |
| `BOOLEAN` | `boolean` | Checkbox (requires format_values.py normalization) |
| `SELECT` | `select` | Dropdown from `paramValues` |
| `LIST` | `repeat` | Repeatable input (requires format_values.py normalization) |

---

## Available Tools

| Tool | Description |
|------|-------------|
| `arcsinh_normalization` | Arcsinh transformation for data normalization |
| `boxplot` | Create boxplot visualizations |
| `calculate_centroid` | Calculate cell centroids |
| `downsample_cells` | Downsample cell populations |
| `hierarchical_heatmap` | Generate hierarchical heatmaps |
| `interactive_spatial_plot` | Interactive spatial visualizations |
| `nearest_neighbor_calculation` | Calculate nearest neighbors |
| `neighborhood_profile` | Generate neighborhood profiles |
| `phenograph_clustering` | PhenoGraph clustering analysis |
| `quantile_scaling` | Quantile-based data scaling |
| `relational_heatmap` | Create relational heatmaps |
| `rename_labels` | Rename annotation labels |
| `ripley_l_calculation` | Ripley's L function calculation |
| `sankey_plot` | Generate Sankey diagrams |
| `select_values` | Select/filter data values |
| `setup_analysis` | Initialize analysis workspace |
| `spatial_interaction` | Spatial interaction analysis |
| `subset_analysis` | Subset data for analysis |
| `visualize_nearest_neighbor` | Visualize nearest neighbor results |
| `visualize_ripley_l` | Visualize Ripley's L results |
| `z-score_normalization` | Z-score normalization |

---

## Troubleshooting

### Common Issues

**1. "No outputs configuration found"**
- Make sure your blueprint has an `outputs` section
- Check that output names match what your template produces

**2. Boolean parameters not working correctly**
- Ensure the parameter is listed in `--bool-values` flag
- Check that your template expects Python `True`/`False`, not strings

**3. List parameters empty or malformed**
- Ensure the parameter is listed in `--list-values` flag
- Check that Galaxy is generating the `_repeat` suffix structure

**4. Files not found in Galaxy output**
- Verify `from_work_dir` in XML matches actual output filename
- For directories, ensure `discover_datasets` pattern is correct

### Debug Mode

```bash
python galaxy_xml_synthesizer.py blueprint_json/ -o . --debug
```

Shows feature summary for all generated tools.

### Testing format_values.py Locally

```bash
# Create a test input
echo '{"Horizontal_Plot": "True", "Feature_s_to_Plot_repeat": [{"value": "CD4"}]}' > test_input.json

# Run format_values.py
python format_values.py test_input.json test_output.json \
    --bool-values Horizontal_Plot \
    --list-values Feature_s_to_Plot \
    --debug

# Check output
cat test_output.json
```

---

## License

See the main repository [LICENSE](../../LICENSE) file.
