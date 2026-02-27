#!/bin/bash

# SPAC Performance Validation Script
# This script runs performance tests to validate speedup improvements in SPAC

set -e

echo "=============================================="
echo "SPAC Performance Validation Test"
echo "=============================================="

# Check if we're in a Docker container or local environment
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container..."
    PYTHON_CMD="/opt/conda/envs/spac/bin/python"
    # In Docker, expect files to be volume-mounted in current directory
    PROJECT_DIR="/home/reviewer/SCSAWorkflow"
else
    echo "Running in local environment..."
    # Assume conda environment is activated
    PYTHON_CMD="python"
    # Find project directory
    if [ -d "tests/test_performance" ]; then
        PROJECT_DIR="."
    elif [ -d "../tests/test_performance" ]; then
        PROJECT_DIR=".."
    else
        PROJECT_DIR="$(dirname "$0")"
    fi
fi

echo "Project directory: $PROJECT_DIR"

# Verify SPAC installation
echo ""
echo "1. Verifying SPAC installation..."
$PYTHON_CMD -c "
import spac
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
print(f'✅ SPAC version: {spac.__version__}')
print('✅ All required modules imported successfully!')
print('✅ Performance test dependencies available')
"

echo ""
echo "2. Checking for performance test files..."
if [ ! -f "$PROJECT_DIR/tests/test_performance/test_boxplot_performance.py" ]; then
    echo "❌ Error: test_boxplot_performance.py not found"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/tests/test_performance/test_histogram_performance.py" ]; then
    echo "❌ Error: test_histogram_performance.py not found"
    exit 1
fi

echo "✅ Performance test files found"

# Create results directory
if [ -f /.dockerenv ]; then
    # In Docker, save to workspace which is volume mounted
    RESULTS_DIR="/workspace/performance_results"
else
    # Local environment, save to project directory
    RESULTS_DIR="$PROJECT_DIR/performance_results"
fi
mkdir -p "$RESULTS_DIR"

echo ""
echo "3. Running performance tests..."
echo ""
echo "⚠️  SYSTEM REQUIREMENTS:"
echo "    • Boxplot tests: 1M, 5M, and 10M cell datasets"
echo "    • Histogram tests: 1M, 5M, and 10M cell datasets"
echo "    • Memory requirement: 12-16 GB Docker memory allocation"
echo "    • Total execution time: 5-10 minutes depending on hardware"
echo "    • Expected speedups: 2-25x improvement demonstrations"
echo ""

# Generate timestamped output filename to avoid conflicts
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BOXPLOT_OUTPUT="$RESULTS_DIR/boxplot_performance_${TIMESTAMP}.txt"
HISTOGRAM_OUTPUT="$RESULTS_DIR/histogram_performance_${TIMESTAMP}.txt"

echo "📊 BOXPLOT PERFORMANCE TESTS"
echo "=================================="
echo "Testing boxplot vs boxplot_interactive speedups..."
echo "Output will be saved to: $BOXPLOT_OUTPUT"
echo ""

cd "$PROJECT_DIR"

# Run boxplot performance tests (full suite: 1M, 5M, and 10M cells)
echo "Running boxplot performance tests..."
echo "Note: Testing complete dataset range - 1M, 5M, and 10M cells (requires 12+ GB Docker memory)"
$PYTHON_CMD -u -m pytest tests/test_performance/test_boxplot_performance.py -v -s --tb=short 2>&1 | tee "$BOXPLOT_OUTPUT"

BOXPLOT_STATUS=${PIPESTATUS[0]}

echo ""
echo "📊 HISTOGRAM PERFORMANCE TESTS" 
echo "=================================="
echo "Testing histogram_old vs histogram speedups..."
echo "Output will be saved to: $HISTOGRAM_OUTPUT"
echo ""

# Run histogram performance tests
echo "Running histogram performance tests..."
$PYTHON_CMD -u -m pytest tests/test_performance/test_histogram_performance.py -v -s --tb=short 2>&1 | tee "$HISTOGRAM_OUTPUT"

HISTOGRAM_STATUS=${PIPESTATUS[0]}

echo ""
echo "4. Performance test summary..."
echo ""

# Check if tests completed successfully
if [ $BOXPLOT_STATUS -eq 0 ]; then
    echo "✅ Boxplot performance tests completed successfully"
else
    echo "❌ Boxplot performance tests failed (exit code: $BOXPLOT_STATUS)"
fi

if [ $HISTOGRAM_STATUS -eq 0 ]; then
    echo "✅ Histogram performance tests completed successfully"
else
    echo "❌ Histogram performance tests failed (exit code: $HISTOGRAM_STATUS)"
fi

echo ""
echo "📁 RESULTS LOCATION"
echo "==================="
echo "Full test outputs saved to:"
echo "  Boxplot results:   $BOXPLOT_OUTPUT"
echo "  Histogram results: $HISTOGRAM_OUTPUT"



echo ""
echo "=============================================="
if [ $BOXPLOT_STATUS -eq 0 ] && [ $HISTOGRAM_STATUS -eq 0 ]; then
    echo "🎉 All performance tests completed successfully!"
    echo "   Speedup improvements have been validated."
else
    echo "⚠️  Some performance tests encountered issues."
    echo "   Please check the detailed outputs above."
fi
echo "=============================================="

# Exit with appropriate code
if [ $BOXPLOT_STATUS -eq 0 ] && [ $HISTOGRAM_STATUS -eq 0 ]; then
    exit 0
else
    exit 1
fi