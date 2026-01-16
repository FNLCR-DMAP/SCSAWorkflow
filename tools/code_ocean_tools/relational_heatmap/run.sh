#!/usr/bin/env bash
set -ex

# Code Ocean Entry Point
# Copy shared format_values.py from parent directory
cp ../format_values.py . 2>/dev/null || true

# Run main script
bash main.sh "$@"
