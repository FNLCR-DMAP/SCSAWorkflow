# SPAC Tool Generation Pipeline

## Side-by-Side Comparison: Galaxy vs Code Ocean

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SPAC TOOL GENERATION PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│                                    ┌───────────────────────┐                                        │
│                                    │   Blueprint JSONs     │                                        │
│                                    │ (template_json_*.json)│                                        │
│                                    │                       │                                        │
│                                    │  • title              │                                        │
│                                    │  • description        │                                        │
│                                    │  • parameters         │                                        │
│                                    │  • inputDatasets      │                                        │
│                                    │  • outputs            │                                        │
│                                    │  • orderedMustacheKeys│                                        │
│                                    └───────────┬───────────┘                                        │
│                                                │                                                    │
│                              ┌─────────────────┴─────────────────┐                                  │
│                              │                                   │                                  │
│                              ▼                                   ▼                                  │
│  ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐          │
│  │      galaxy_xml_synthesizer.py          │   │     code_ocean_synthesizer.py           │          │
│  │                                         │   │                                         │          │
│  │  python galaxy_xml_synthesizer.py \     │   │  python code_ocean_synthesizer.py \     │          │
│  │    blueprint_jsons/ \                   │   │    blueprint_jsons/ \                   │          │
│  │    -o galaxy_tools/                     │   │    -o code_ocean_tools/                 │          │
│  └───────────────┬─────────────────────────┘   └───────────────┬─────────────────────────┘          │
│                  │                                             │                                    │
│                  ▼                                             ▼                                    │
│  ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐          │
│  │         GALAXY TOOLS OUTPUT             │   │       CODE OCEAN TOOLS OUTPUT           │          │
│  │                                         │   │                                         │          │
│  │  galaxy_tools/                          │   │  code_ocean_tools/                      │          │
│  │  ├── spac_boxplot.xml                   │   │  ├── boxplot/                           │          │
│  │  ├── spac_histogram.xml                 │   │  │   ├── .codeocean/                    │          │
│  │  ├── spac_spatial_plot.xml              │   │  │   │   └── app-panel.json             │          │
│  │  ├── format_values.py (shared)          │   │  │   ├── format_values.py               │          │
│  │  └── ... (38+ XML files)                │   │  │   └── run.sh                         │          │
│  │                                         │   │  ├── histogram/                         │          │
│  │                                         │   │  │   └── ...                            │          │
│  │                                         │   │  └── ... (38+ tool folders)             │          │
│  └─────────────────────────────────────────┘   └─────────────────────────────────────────┘          │
│                  │                                             │                                    │
│                  ▼                                             ▼                                    │
│  ┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐          │
│  │          GALAXY DEPLOYMENT              │   │        CODE OCEAN DEPLOYMENT            │          │
│  │                                         │   │                                         │          │
│  │  1. Copy XML + format_values.py to      │   │  1. Push each tool folder to Git repo   │          │
│  │     Galaxy tool directory               │   │  2. Import from Git in Code Ocean       │          │
│  │  2. Restart Galaxy or reload tools      │   │  3. App Panel auto-generates from       │          │
│  │  3. Tools appear in Galaxy UI           │   │     app-panel.json                      │          │
│  │                                         │   │  4. Configure Docker environment        │          │
│  │  Docker: nciccbr/spac:v0.9.1            │   │  Docker: nciccbr/spac:v0.9.1            │          │
│  └─────────────────────────────────────────┘   └─────────────────────────────────────────┘          │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Differences

| Aspect | Galaxy | Code Ocean |
|--------|--------|------------|
| **Synthesizer** | `galaxy_xml_synthesizer.py` | `code_ocean_synthesizer.py` |
| **Output Format** | Single `.xml` file per tool | Folder with 3 files per tool |
| **UI Definition** | XML `<inputs>` section | `.codeocean/app-panel.json` |
| **Entry Point** | Galaxy command wrapper | `run.sh` bash script |
| **Parameter Passing** | Galaxy JSON (`$params_json`) | Named args (`--param=value`) |
| **Parameter Types** | `<param type="text\|integer\|...">` | `"value_type": "string\|number\|..."` |
| **Sections/Categories** | `<section>` elements | `"categories"` array |
| **Output Discovery** | Galaxy `<outputs>` | Results in `/results/` |

---

## File Structure Comparison

```
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│          GALAXY TOOL                 │    │        CODE OCEAN CAPSULE            │
├──────────────────────────────────────┤    ├──────────────────────────────────────┤
│                                      │    │                                      │
│  spac_boxplot.xml                    │    │  boxplot/                            │
│  ├── <tool id="spac_boxplot">        │    │  ├── .codeocean/                     │
│  ├── <description>                   │    │  │   └── app-panel.json              │
│  ├── <requirements>                  │    │  │       ├── "named_parameters": true│
│  │   └── <container type="docker">   │    │  │       ├── "categories": [...]     │
│  ├── <configfiles>                   │    │  │       └── "parameters": [...]     │
│  ├── <command>                       │    │  │                                   │
│  │   └── python format_values.py ... │    │  ├── format_values.py                │
│  │       python -c "from spac..."    │    │  │   └── (parameter normalization)   │
│  ├── <inputs>                        │    │  │                                   │
│  │   ├── <section>                   │    │  └── run.sh                          │
│  │   └── <param type="...">          │    │      ├── Parse --param=value args    │
│  ├── <outputs>                       │    │      ├── Create params.json          │
│  │   └── <data name="..." format=""> │    │      ├── Call format_values.py       │
│  └── <help>                          │    │      └── Run SPAC template           │
│                                      │    │                                      │
│  format_values.py (shared)           │    │  (format_values.py per tool)         │
│                                      │    │                                      │
└──────────────────────────────────────┘    └──────────────────────────────────────┘
```

---

## Runtime Flow Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     GALAXY RUNTIME                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   User fills Galaxy form                                                                │
│            │                                                                            │
│            ▼                                                                            │
│   Galaxy serializes inputs → galaxy_params.json                                         │
│            │                                                                            │
│            ▼                                                                            │
│   format_values.py normalizes → cleaned_params.json                                     │
│            │                                                                            │
│            ▼                                                                            │
│   python -c "from spac.templates.boxplot_template import run_from_json; ..."            │
│            │                                                                            │
│            ▼                                                                            │
│   Outputs collected by Galaxy (based on <outputs> XML)                                  │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CODE OCEAN RUNTIME                                    │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   User fills App Panel                                                                  │
│            │                                                                            │
│            ▼                                                                            │
│   Code Ocean executes: bash run.sh --Figure_Width=12 --Keep_Outliers=False ...          │
│            │                                                                            │
│            ▼                                                                            │
│   run.sh parses args → /results/jsons/params.json                                       │
│            │                                                                            │
│            ▼                                                                            │
│   format_values.py normalizes → /results/jsons/cleaned_params.json                      │
│            │                                                                            │
│            ▼                                                                            │
│   python -c "from spac.templates.boxplot_template import run_from_json; ..."            │
│            │                                                                            │
│            ▼                                                                            │
│   Results saved to /results/ (figures/, jsons/, CSVs)                                   │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Shared Components

Both synthesizers share:

1. **Blueprint JSON** - Same input format for both platforms
2. **format_values.py** - Same parameter normalization logic
3. **SPAC Templates** - Same `run_from_json()` entry point
4. **Docker Image** - Same `nciccbr/spac:v0.9.1` container
5. **Parameter Type Mapping** - Similar type conversions

```
Blueprint Type    →    Galaxy Type       →    Code Ocean Type
─────────────────────────────────────────────────────────────
STRING            →    text              →    text/string
INTEGER/INT       →    integer           →    text/number
NUMBER/FLOAT      →    float             →    text/number
BOOLEAN           →    boolean           →    list/boolean
SELECT            →    select            →    list/string
LIST              →    repeat            →    text/string (comma-sep)
```

---

## Quick Commands

```bash
# Generate all Galaxy tools
python galaxy_xml_synthesizer.py blueprint_jsons/ -o galaxy_tools/

# Generate all Code Ocean capsules
python code_ocean_synthesizer.py blueprint_jsons/ -o code_ocean_tools/

# Generate a single tool (either platform)
python galaxy_xml_synthesizer.py blueprint_jsons/template_json_boxplot.json -o galaxy_tools/
python code_ocean_synthesizer.py blueprint_jsons/template_json_boxplot.json -o code_ocean_tools/
```
