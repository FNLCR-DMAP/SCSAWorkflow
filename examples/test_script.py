#!/usr/bin/env python3
"""
Test script for load_csv_files_with_config.py
This demonstrates how to use the SPAC load_csv_files_with_config template
"""

import os
import sys
import json
import shutil
import pandas as pd
from pathlib import Path

def setup_test_environment():
    """Create test directories and files"""

    # Create directory structure
    base_dir = Path("test_load_csv")
    input_dir = base_dir / "input_data"
    output_dir = base_dir / "output"

    # Create directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample_data_1.csv
    data1 = {
        'CellID': [1, 2, 3, 4, 5],
        'X_centroid': [100.5, 110.2, 120.8, 130.5, 140.3],
        'Y_centroid': [200.3, 205.7, 210.4, 215.2, 220.8],
        'Area': [150, 145, 160, 155, 148],
        'CD45': [0.85, 0.78, 0.82, 0.90, 0.75],
        'CD3D': [0.92, 0.88, 0.10, 0.05, 0.89],
        'CD20': [0.12, 0.15, 0.95, 0.93, 0.08],
        'cell_type': ['T_cell', 'T_cell', 'B_cell', 'B_cell', 'T_cell']
    }
    df1 = pd.DataFrame(data1)
    df1.to_csv(input_dir / "sample_data_1.csv", index=False)

    # Create sample_data_2.csv
    data2 = {
        'CellID': [6, 7, 8, 9, 10],
        'X_centroid': [200.5, 210.2, 220.8, 230.5, 240.3],
        'Y_centroid': [300.3, 305.7, 310.4, 315.2, 320.8],
        'Area': [152, 147, 162, 157, 149],
        'CD45': [0.88, 0.79, 0.83, 0.91, 0.76],
        'CD3D': [0.91, 0.87, 0.09, 0.04, 0.90],
        'CD20': [0.11, 0.14, 0.96, 0.94, 0.07],
        'cell_type': ['T_cell', 'T_cell', 'B_cell', 'B_cell', 'T_cell']
    }
    df2 = pd.DataFrame(data2)
    df2.to_csv(input_dir / "sample_data_2.csv", index=False)

    # Create configuration table
    config_data = {
        'file_name': ['sample_data_1.csv', 'sample_data_2.csv'],
        'slide_number': ['S001', 'S002'],
        'tissue_type': ['tumor', 'normal'],
        'patient_id': ['P001', 'P002']
    }
    config_df = pd.DataFrame(config_data)
    config_df.to_csv(input_dir / "config_table.csv", index=False)

    # Create params.json
    params = {
        "CSV_Files": str(input_dir),
        "CSV_Files_Configuration": str(input_dir / "config_table.csv"),
        "String_Columns": ["cell_type", "tissue_type", "patient_id"],
        "Output_File": str(output_dir / "combined_data.csv")
    }

    with open(base_dir / "params.json", 'w') as f:
        json.dump(params, f, indent=2)

    print(f"✅ Test environment created in '{base_dir}'")
    return base_dir

def simulate_load_csv_with_config(base_dir):
    """
    Simulates the behavior of load_csv_files_with_config.py
    This shows what the actual SPAC script would do
    """

    # Load parameters
    with open(base_dir / "params.json", 'r') as f:
        params = json.load(f)

    # Read configuration table
    config_df = pd.read_csv(params["CSV_Files_Configuration"])
    print(f"\n📋 Configuration table loaded with {len(config_df)} files")
    print(config_df)

    # Process each CSV file
    combined_data = []

    for idx, row in config_df.iterrows():
        file_name = row['file_name']
        file_path = Path(params["CSV_Files"]) / file_name

        if file_path.exists():
            print(f"\n📂 Processing: {file_name}")

            # Read CSV with string columns specified
            dtype_dict = {col: str for col in params.get("String_Columns", [])}
            df = pd.read_csv(file_path, dtype=dtype_dict)

            # Add metadata from configuration table
            for col in config_df.columns:
                if col != 'file_name':
                    df[col] = row[col]

            combined_data.append(df)
            print(f"   - Loaded {len(df)} rows")
            print(f"   - Added metadata: {', '.join([c for c in config_df.columns if c != 'file_name'])}")
        else:
            print(f"\n❌ File not found: {file_path}")

    # Combine all dataframes
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)

        # Save combined data
        output_path = Path(params["Output_File"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)

        print(f"\n✅ Combined data saved to: {output_path}")
        print(f"   - Total rows: {len(final_df)}")
        print(f"   - Columns: {list(final_df.columns)}")

        # Display sample of final data
        print(f"\n📊 Sample of combined data:")
        print(final_df.head(10))
    else:
        print("\n❌ No data to combine")

def run_actual_script(base_dir):
    """
    This function would run the actual load_csv_files_with_config.py script
    """
    script_path = Path("load_csv_files_with_config.py")

    if script_path.exists():
        print("\n🚀 Running actual SPAC script...")
        import subprocess

        # Change to test directory
        os.chdir(base_dir)

        # Run the script with parameters
        cmd = [sys.executable, str(script_path.absolute()), "params.json"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Script executed successfully")
            print(result.stdout)
        else:
            print("❌ Script failed")
            print(result.stderr)
    else:
        print(f"\n⚠️  Script not found at {script_path}")
        print("   Using simulation instead...")

def main():
    """Main test function"""
    print("=" * 60)
    print("Testing load_csv_files_with_config.py")
    print("=" * 60)

    # Setup test environment
    base_dir = setup_test_environment()

    # Simulate the script behavior
    simulate_load_csv_with_config(base_dir)

    # Optional: Try to run actual script
    # run_actual_script(base_dir)

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
