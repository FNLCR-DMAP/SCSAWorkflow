# Template Utils Refactoring - Generalized save_results

## Overview
The `save_results` function in `template_utils.py` has been refactored to handle dynamic output configurations based on blueprint JSON specifications. This allows each template to save results according to its specific output requirements without hardcoding paths.

## Key Changes

### 1. New `save_results` Function Signature
```python
def save_results(
    results: Dict[str, Any],
    outputs_config: Dict[str, str],
    output_base_dir: Union[str, Path] = ".",
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]
```

### 2. Parameters
- **`results`**: Dictionary of results organized by type (e.g., "analysis", "dataframes", "figures", "html")
- **`outputs_config`**: Configuration from blueprint JSON specifying output destinations
- **`output_base_dir`**: Base directory for all outputs
- **`params`**: Optional parameters for additional context

### 3. Output Configuration Examples

#### Boxplot Template
```json
{
  "outputs": {
    "DataFrames": "dataframe_folder",
    "figures": "figure_folder"
  }
}
```

#### Full Analysis Template
```json
{
  "outputs": {
    "analysis": "transform_output.pickle",
    "DataFrames": "dataframe_folder",
    "figures": "figure_folder",
    "html": "html_folder"
  }
}
```

## Template Implementation Pattern

### Step 1: Parse Parameters and Get Output Config
```python
params = parse_params(json_path)

# Get outputs configuration from blueprint or use default
if outputs_config is None:
    outputs_config = params.get("outputs", {
        "analysis": "transform_output.pickle"
    })
```

### Step 2: Organize Results by Type
```python
results = {}

# Add analysis output if configured
if "analysis" in outputs_config:
    results["analysis"] = adata

# Add dataframes if configured  
if "DataFrames" in outputs_config:
    results["dataframes"] = {
        "summary": summary_df,
        "statistics": stats_df
    }

# Add figures if configured
if "figures" in outputs_config:
    results["figures"] = [fig1, fig2]
    # Or as dictionary: {"plot1": fig1, "plot2": fig2}

# Add HTML if configured
if "html" in outputs_config:
    results["html"] = {"report": html_content}
```

### Step 3: Save Results
```python
output_base_dir = params.get("output_dir", ".")

saved_files = save_results(
    results=results,
    outputs_config=outputs_config,
    output_base_dir=output_base_dir,
    params=params
)
```

## Directory Structure Created

For a blueprint with all output types configured:
```
output_base_dir/
├── transform_output.pickle        # Main analysis result
├── dataframe_folder/              # DataFrames directory
│   ├── summary.csv
│   └── statistics.csv
├── figure_folder/                 # Figures directory
│   ├── figures_0.png
│   └── figures_1.png
└── html_folder/                   # HTML directory
    └── report.html
```

## Key Features

### 1. Automatic Directory Creation
- Creates directories for outputs ending with "_folder", "_dir", or "_directory"
- Creates parent directories as needed

### 2. Flexible Input Formats
Results can be provided as:
- **Single object**: `results["analysis"] = adata`
- **List**: `results["figures"] = [fig1, fig2, fig3]`
- **Dictionary**: `results["dataframes"] = {"summary": df1, "stats": df2}`

### 3. Smart File Format Detection
- DataFrames → CSV by default
- Matplotlib figures → PNG by default  
- AnnData → pickle or h5ad based on extension
- Unknown types → pickle

### 4. Backward Compatibility
- Original `save_outputs` function preserved for legacy code
- Templates can be migrated incrementally

## Migration Guide for Existing Templates

### Before (Old Pattern)
```python
# Hardcoded output handling
output_file = params.get("Output_File", "results.csv")
saved_files = save_outputs({output_file: summary_df})

figure_file = params.get("Figure_File", None)
if figure_file:
    saved_files.update(save_outputs({figure_file: fig}))
```

### After (New Pattern)
```python
# Dynamic output handling based on blueprint
results = {
    "dataframes": {"summary": summary_df},
    "figures": {"boxplot": fig}
}

saved_files = save_results(
    results=results,
    outputs_config=outputs_config,
    output_base_dir=output_base_dir
)
```

## Benefits

1. **Consistency**: All templates use the same output pattern
2. **Flexibility**: Output structure defined by blueprint JSON
3. **Maintainability**: Single implementation to maintain
4. **Galaxy Integration**: Proper directory structure for Galaxy file discovery
5. **Cross-platform**: Works in both Galaxy and Code Ocean environments

## Testing

Run the test script to see the functionality:
```bash
python test_save_results.py
```

This demonstrates:
- Different output configurations
- Multiple file types
- Directory structure creation
- File naming conventions

## Next Steps for Full Implementation

1. Update all templates in `/src/spac/templates/` to use the new pattern
2. Update blueprint JSON files to include appropriate `outputs` configuration
3. Test with Galaxy XML wrappers to ensure proper file discovery
4. Update synthesizer to inject output configurations if needed
