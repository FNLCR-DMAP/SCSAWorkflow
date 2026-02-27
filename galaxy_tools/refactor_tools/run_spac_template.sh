#!/usr/bin/env bash
# run_spac_template.sh - Universal wrapper for SPAC Galaxy tools
set -eu

PARAMS_JSON="${1:?Missing params.json path}"
TEMPLATE_NAME="${2:?Missing template name}"

# Get the directory where this script is located (the tool directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Look for spac_galaxy_runner.py in multiple locations
if [ -f "$SCRIPT_DIR/spac_galaxy_runner.py" ]; then
    # If it's in the same directory as this script
    RUNNER_PATH="$SCRIPT_DIR/spac_galaxy_runner.py"
elif [ -f "$__tool_directory__/spac_galaxy_runner.py" ]; then
    # If Galaxy provides tool directory
    RUNNER_PATH="$__tool_directory__/spac_galaxy_runner.py"
else
    # Fallback to trying the module approach
    echo "Warning: spac_galaxy_runner.py not found locally, trying as module" >&2
    python3 -m spac_galaxy_runner "$PARAMS_JSON" "$TEMPLATE_NAME"
    exit $?
fi

# Run the runner script directly
echo "Running: python3 $RUNNER_PATH $PARAMS_JSON $TEMPLATE_NAME" >&2
python3 "$RUNNER_PATH" "$PARAMS_JSON" "$TEMPLATE_NAME"