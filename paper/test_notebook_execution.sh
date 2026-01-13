#!/bin/bash

# SPAC Notebook Execution Test Script
# This script tests that the SPAC installation works correctly by executing the example notebook

set -e

echo "=============================================="
echo "SPAC Notebook Execution Test"
echo "=============================================="

# Check if we're in a Docker container or local environment
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container..."
    PYTHON_CMD="/opt/conda/envs/spac/bin/python"
    JUPYTER_CMD="/opt/conda/envs/spac/bin/jupyter"
    # In Docker, expect files to be volume-mounted in current directory
    DATA_DIR="."
    OUTPUT_DIR="."
else
    echo "Running in local environment..."
    # Assume conda environment is activated
    PYTHON_CMD="python"
    JUPYTER_CMD="jupyter"
    # Check if running from paper/examples directory
    if [ -f "lymphnode_analysis.ipynb" ]; then
        DATA_DIR="."
        OUTPUT_DIR="."
    else
        # Running from project root or paper directory
        DATA_DIR="$(dirname "$0")/examples"
        OUTPUT_DIR="$(dirname "$0")/results"
        mkdir -p "$OUTPUT_DIR"
    fi
fi

# Verify SPAC installation
echo "1. Verifying SPAC installation..."
$PYTHON_CMD -c "
import spac
import scimap
print(f'✅ SPAC version: {spac.__version__}')
print(f'✅ scimap version: {scimap.__version__}')
print('✅ All modules imported successfully!')
"

echo ""
echo "2. Checking for required files..."
if [ ! -f "$DATA_DIR/example_lymphnode_data.csv" ]; then
    echo "❌ Error: example_lymphnode_data.csv not found in $DATA_DIR"
    exit 1
fi

if [ ! -f "$DATA_DIR/lymphnode_analysis.ipynb" ]; then
    echo "❌ Error: lymphnode_analysis.ipynb not found in $DATA_DIR"
    exit 1
fi

echo "✅ Required files found"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "3. Executing notebook..."

# Generate timestamped output filename to avoid conflicts
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_NAME="lymphnode_analysis_executed_${TIMESTAMP}.ipynb"

echo "   Input: $DATA_DIR/lymphnode_analysis.ipynb"
echo "   Output: $OUTPUT_DIR/$OUTPUT_NAME"

cd "$DATA_DIR"
$JUPYTER_CMD nbconvert --execute --to notebook lymphnode_analysis.ipynb --output-dir "$OUTPUT_DIR" --output "$OUTPUT_NAME"

echo ""
echo "4. Verification complete!"
echo "✅ Notebook executed successfully"
echo "✅ Output saved to: $OUTPUT_DIR/$OUTPUT_NAME"

# Show file size as verification
if [ -f "$OUTPUT_DIR/$OUTPUT_NAME" ]; then
    FILE_SIZE=$(ls -lh "$OUTPUT_DIR/$OUTPUT_NAME" | awk '{print $5}')
    echo "✅ Output file size: $FILE_SIZE"
else
    echo "❌ Error: Expected output file not found"
    exit 1
fi

echo ""
echo "=============================================="
echo "🎉 SPAC installation and notebook execution verified successfully!"
echo "=============================================="

# Check if we're in Docker and provide Jupyter instructions
if [ -f /.dockerenv ]; then
    echo ""
    echo "📓 To view the executed notebook in Jupyter:"
    echo "   1. Exit this container (Ctrl+C or let it finish)"
    echo "   2. Run the following command to start Jupyter:"
    echo "      docker run --rm -p 8888:8888 -v \$(pwd):/workspace spac"
    echo "   3. Open your browser to: http://localhost:8888"
    echo "   4. Navigate to the executed notebook: $OUTPUT_NAME"
    echo ""
    echo "💡 Quick command with auto-cleanup:"
    echo "   docker stop \$(docker ps -q --filter \"publish=8888\") 2>/dev/null || true && \\"
    echo "   docker run --rm -p 8888:8888 -v \$(pwd):/workspace spac"
    echo ""
    echo "🚀 Start Jupyter now? (y/N)"
    read -t 10 -r response || response="n"
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Starting Jupyter server..."
        echo "Open your browser to: http://localhost:8888"
        exec $JUPYTER_CMD notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/workspace
    fi
else
    echo ""
    echo "📓 To view the executed notebook in Jupyter:"
    echo "   Run: jupyter notebook $OUTPUT_DIR/$OUTPUT_NAME"
    echo "   Or browse all notebooks: jupyter notebook $OUTPUT_DIR"
    echo ""
    echo "🚀 Start Jupyter now? (y/N)"
    read -t 10 -r response || response="n"
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Starting Jupyter server..."
        $JUPYTER_CMD notebook "$OUTPUT_DIR"
    fi
fi