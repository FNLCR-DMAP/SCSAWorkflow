# SPAC Tools - Code Ocean Deployment Guide

## Overview

This guide explains how to deploy SPAC tools to Code Ocean platform using the synthesizer approach.

---

## Generated Structure (Monorepo)

```
code_ocean_tools/                    # Git repository root
├── format_values.py                 # SHARED - one copy for all 38 tools
├── boxplot/
│   ├── .codeocean/
│   │   └── app-panel.json          # UI config (auto-generates App Panel)
│   ├── run.sh                      # Entry point (copies shared file)
│   └── main.sh                     # Parameter parsing + execution
├── histogram/
│   └── ...
└── ... (38 tools)
```

**Key Point:** `format_values.py` exists ONCE at the root. Each tool's `run.sh` copies it at runtime.

---

## Step 1: Generate Capsule Files

Run the synthesizer on your blueprint JSONs:

```bash
# Single tool
python code_ocean_synthesizer.py template_json_boxplot.json -o code_ocean_tools/

# All tools (batch)
python code_ocean_synthesizer.py blueprint_jsons/ -o code_ocean_tools/
```

**Output:**
```
code_ocean_tools/
├── format_values.py              # SHARED (generated once)
├── boxplot/
│   ├── .codeocean/app-panel.json
│   ├── run.sh
│   └── main.sh
├── histogram/
│   └── ...
└── ... (38 tools)
```

---

## Step 2: Push to Git Repository

Each tool folder needs to be a git repository for Code Ocean import.

### Option A: Separate repos per tool (Recommended)

```bash
cd code_ocean_tools/boxplot

# Initialize git
git init
git add .
git commit -m "Initial boxplot capsule"

# Create GitHub repo and push
gh repo create fnlcr-dmap/spac-boxplot --public --source=. --push
```

### Option B: Monorepo with tool subfolders

Keep all tools in your main SCSAWorkflow repo:
```
SCSAWorkflow/
└── tools/
    └── code_ocean_tools/
        ├── boxplot/
        ├── histogram/
        └── ...
```

---

## Step 3: Create Capsule in Code Ocean

### 3.1 Log in to Code Ocean
- Go to: https://codeocean.nih.gov (or your instance)

### 3.2 Create New Capsule from Git

1. Click **"+ New Capsule"**
2. Select **"Import from Git"** or **"Copy from Git"**
3. Enter your repository URL:
   - Separate repo: `https://github.com/fnlcr-dmap/spac-boxplot.git`
   - Monorepo: `https://github.com/FNLCR-DMAP/SCSAWorkflow.git`

### 3.3 Automatic App Panel Generation

**Key Point:** When Code Ocean detects `.codeocean/app-panel.json`, it automatically generates the App Panel UI!

You should see:
- Parameters organized by categories
- Default values populated
- Input types (text, dropdown, boolean) configured

### 3.4 Configure Environment

1. Go to **Environment** tab
2. Click **"Add Environment"**
3. Select **Docker** and enter: `nciccbr/spac:v2-dev`
4. Click **"Save"**

### 3.5 Attach Data

1. Go to **Data** tab
2. Click **"Add Data Asset"**
3. Select or upload your input data (`.pickle`, `.pkl`, or `.h5ad` files)

### 3.6 Test Run

1. Go to **App Panel** tab
2. Fill in parameter values
3. Click **"Reproducible Run"**

---

## File Details

### `.codeocean/app-panel.json`

This file defines the App Panel UI:

```json
{
    "version": 1,
    "named_parameters": false,
    "general": {
        "title": "Boxplot [SPAC] [DMAP]",
        "instructions": "Create a boxplot visualization..."
    },
    "categories": [
        {"id": "figure_config", "name": "Figure Configuration", "help_text": "..."}
    ],
    "parameters": [
        {
            "id": "abc123",
            "name": "Figure Width",
            "param_name": "Figure_Width",
            "type": "text",
            "value_type": "number",
            "default_value": "12"
        }
    ]
}
```

**Parameter Type Mapping:**

| Blueprint Type | app-panel `type` | app-panel `value_type` |
|---------------|------------------|------------------------|
| STRING        | text             | string                 |
| INTEGER/INT   | text             | number                 |
| NUMBER/FLOAT  | text             | number                 |
| BOOLEAN       | list             | boolean                |
| SELECT        | list             | string (+ extra_data)  |
| LIST          | text             | string (comma-sep)     |

### `run.sh`

Entry point that copies shared `format_values.py` before running:

```bash
#!/usr/bin/env bash
set -ex

# Copy shared format_values.py from parent directory
cp ../format_values.py . 2>/dev/null || true

# Run main script
bash main.sh "$@"
```

### `main.sh`

Handles parameter parsing and template execution:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Capture App Panel arguments (positional: $1, $2, ...)
FIGURE_WIDTH="${8:-12}"
FIGURE_HEIGHT="${9:-8}"

# Find input data
INPUT=$(find -L ../data -type f \( -name "*.pickle" -o -name "*.h5ad" \) | head -n 1)

# Create parameters JSON
cat > /results/jsons/params.json << EOF
{
    "Upstream_Analysis": "$INPUT",
    "Figure_Width": "$FIGURE_WIDTH",
    ...
}
EOF

# Normalize parameters
python format_values.py /results/jsons/params.json /results/jsons/cleaned_params.json \
    --bool-values Horizontal_Plot --list-values Feature_s_to_Plot

# Run template
python -c "from spac.templates.boxplot_template import run_from_json; run_from_json('/results/jsons/cleaned_params.json')"
```

### `format_values.py` (SHARED)

**Location:** `code_ocean_tools/format_values.py` (root level, one copy)

Normalizes parameters for templates:
- Converts `"True"` → `true` (Python boolean)
- Converts `"a, b, c"` → `["a", "b", "c"]` (list)
- Injects output configuration

Each tool's `run.sh` copies this file to its working directory at runtime.

---

## Runtime Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CODE OCEAN RUNTIME                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User fills App Panel                                        │
│     ┌────────────────────────┐                                  │
│     │ Figure Width: [12]     │                                  │
│     │ Horizontal:   [False▼] │                                  │
│     └────────────────────────┘                                  │
│                   ↓                                             │
│  2. Code Ocean executes: bash run.sh "Original" "All" ...       │
│                   ↓                                             │
│  3. main.sh captures args → creates params.json                 │
│                   ↓                                             │
│  4. format_values.py normalizes → cleaned_params.json           │
│                   ↓                                             │
│  5. SPAC template runs with cleaned params                      │
│                   ↓                                             │
│  6. Results saved to /results/                                  │
│     ├── figures/                                                │
│     ├── jsons/params.json (debug)                               │
│     └── dataframe.csv                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Code Ocean Requirements

| Requirement | How It's Handled |
|-------------|------------------|
| Entry point | `run.sh` at capsule root |
| Output location | All outputs to `/results/` |
| Environment | Docker image: `nciccbr/spac:v2-dev` |
| Data location | Input data in `../data/` |

---

## Batch Deployment (40 Tools)

### Generate all capsules:
```bash
python code_ocean_synthesizer.py blueprint_jsons/ -o code_ocean_tools/
```

### Create repos and push (script):
```bash
#!/bin/bash
for tool_dir in code_ocean_tools/*/; do
    tool_name=$(basename "$tool_dir")
    echo "Deploying: $tool_name"
    
    cd "$tool_dir"
    git init
    git add .
    git commit -m "Initial $tool_name capsule"
    
    # Create GitHub repo (requires gh CLI)
    gh repo create "fnlcr-dmap/spac-$tool_name" --public --source=. --push
    
    cd ../..
done
```

---

## Troubleshooting

### App Panel not appearing
- Verify `.codeocean/app-panel.json` exists
- Check JSON syntax: `python -m json.tool .codeocean/app-panel.json`
- Ensure `"version": 1` is present

### "format_values.py not found"
- Ensure `format_values.py` is at the root of `code_ocean_tools/`
- The shared file must be in the parent directory of each tool
- Check that `run.sh` has the copy command: `cp ../format_values.py .`

### "No input file found"
- Attach data asset in Code Ocean Data tab
- Verify file extension is `.pickle`, `.pkl`, or `.h5ad`

### Template not found
- Ensure Docker image `nciccbr/spac:v2-dev` contains `spac.templates`
- Check template name matches: `{tool_name}_template`

### Results not appearing
- All outputs must go to `/results/` directory
- Check `main.sh` creates `/results/figures` and `/results/jsons`

---

## Quick Reference

### Synthesizer Command
```bash
python code_ocean_synthesizer.py <blueprint_json_or_dir> -o <output_dir>
```

### Generated Structure
```
code_ocean_tools/
├── format_values.py           ← SHARED (one copy for all tools)
├── tool_name/
│   ├── .codeocean/app-panel.json  ← UI config
│   ├── run.sh                     ← Entry point (copies shared file)
│   └── main.sh                    ← Logic
└── ...
```

### Docker Image
```
nciccbr/spac:v2-dev
```

### Key Conor Insight
> "If you create a capsule by importing a `.codeocean/app-panel.json` from a git repo, 
> the corresponding capsule app panel will be generated."

This eliminates manual App Panel building! 🎉
