# SPAC Tools - Code Ocean README

## Overview

This directory contains the Code Ocean synthesizer and generated capsules for deploying SPAC tools to the Code Ocean platform.

## Quick Start

### Generate Capsule Files

```bash
# Generate all tools from blueprint JSONs
python code_ocean_synthesizer.py blueprint_jsons/ -o code_ocean_tools/

# Generate a single tool
python code_ocean_synthesizer.py blueprint_jsons/template_json_boxplot.json -o code_ocean_tools/
```

### Generated Structure

```
code_ocean_tools/
├── format_values.py           # Shared utility (one copy for all tools)
├── boxplot/
│   ├── .codeocean/
│   │   └── app-panel.json     # UI configuration (auto-generates App Panel)
│   └── run.sh                 # Entry point with all logic
├── histogram/
│   └── ...
└── ... (38+ tools)
```

---

## Capsule Components

### `run.sh`

The single entry point script that:
1. Copies shared `format_values.py` from parent directory
2. Initializes parameters with defaults
3. Parses named arguments (`--param_name=value`)
4. Finds input data (`.pickle`, `.pkl`, `.h5ad`)
5. Creates output directories (`/results/figures`, `/results/jsons`)
6. Generates and normalizes `params.json`
7. Executes the SPAC template

### `.codeocean/app-panel.json`

Defines the App Panel UI. Code Ocean automatically generates the UI when this file is present.

**Parameter Type Mapping:**

| Blueprint Type | app-panel `type` | app-panel `value_type` |
|----------------|------------------|------------------------|
| STRING         | text             | string                 |
| INTEGER/INT    | text             | number                 |
| NUMBER/FLOAT   | text             | number                 |
| BOOLEAN        | list             | boolean                |
| SELECT         | list             | string (+ extra_data)  |
| LIST           | text             | string (comma-sep)     |

### `format_values.py`

Shared utility that normalizes parameters:
- Converts `"True"` → `true` (Python boolean)
- Converts `"a, b, c"` → `["a", "b", "c"]` (list)
- Injects output configuration

---

## Deployment to Code Ocean

### 1. Push to Git Repository

```bash
cd code_ocean_tools/boxplot
git init
git add .
git commit -m "Initial boxplot capsule"
gh repo create fnlcr-dmap/spac-boxplot --public --source=. --push
```

### 2. Create Capsule in Code Ocean

1. Log in to [Code Ocean NIH](https://codeocean.nih.gov)
2. Click **"+ New Capsule"** → **"Import from Git"**
3. Enter repository URL
4. App Panel UI auto-generates from `.codeocean/app-panel.json`

### 3. Configure Environment

- Go to **Environment** tab
- Select **Docker** and enter: `nciccbr/spac:v0.9.1`

### 4. Attach Data & Run

1. Go to **Data** tab → attach input files
2. Go to **App Panel** tab → fill parameters
3. Click **"Reproducible Run"**

---

## Runtime Flow

```
User fills App Panel
        ↓
Code Ocean executes: bash run.sh --Param1=value1 --Param2=value2 ...
        ↓
run.sh parses arguments → creates params.json
        ↓
format_values.py normalizes → cleaned_params.json
        ↓
SPAC template runs with cleaned params
        ↓
Results saved to /results/
├── figures/
├── jsons/params.json
└── dataframe.csv
```

---

## Local Testing

```bash
cd code_ocean_tools/boxplot
bash run.sh --Table_to_Visualize=Test --Figure_Width=20 --Keep_Outliers=False
```

---

## Batch Deployment Script

```bash
#!/bin/bash
for tool_dir in code_ocean_tools/*/; do
    tool_name=$(basename "$tool_dir")
    echo "Deploying: $tool_name"
    
    cd "$tool_dir"
    git init
    git add .
    git commit -m "Initial $tool_name capsule"
    gh repo create "fnlcr-dmap/spac-$tool_name" --public --source=. --push
    cd ../..
done
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App Panel not appearing | Verify `.codeocean/app-panel.json` exists and has valid JSON |
| `format_values.py` not found | Ensure it's at `code_ocean_tools/` root level |
| No input file found | Attach data in Code Ocean; check file extensions |
| Template not found | Verify Docker image contains `spac.templates` |
| Results not appearing | Ensure outputs go to `/results/` directory |

---

## Key Files

| File | Purpose |
|------|---------|
| `code_ocean_synthesizer.py` | Generates capsule files from blueprint JSONs |
| `format_values.py` | Shared parameter normalization utility |
| `blueprint_jsons/` | Input JSON definitions for each tool |
| `code_ocean_tools/` | Generated capsule output directory |

---

## References

- [CODE_OCEAN_DEPLOYMENT_GUIDE.md](CODE_OCEAN_DEPLOYMENT_GUIDE.md) - Detailed deployment guide
- [REFACTOR_SUMMARY_v8.md](../docs/REFACTOR_SUMMARY_v8.md) - v8.0 single run.sh refactor notes
