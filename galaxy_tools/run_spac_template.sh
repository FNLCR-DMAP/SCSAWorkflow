#!/usr/bin/env bash
# run_spac_template.sh - Minimal universal wrapper for SPAC templates
# Version: 2.0 - Refactored with separated Python code
set -euo pipefail

PARAMS_JSON="${1:?Missing params.json path}"
TEMPLATE_NAME="${2:?Missing template name}"

# Log start
echo "=== SPAC Template Execution ==="
echo "Template: $TEMPLATE_NAME"
echo "Parameters: $PARAMS_JSON"

# Execute Python bridge and capture output
"${SPAC_PYTHON:-python3}" tool_utils.py "$PARAMS_JSON" "$TEMPLATE_NAME" 2>&1 | tee tool_stdout.txt

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Execution failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "=== Complete ==="
exit 0