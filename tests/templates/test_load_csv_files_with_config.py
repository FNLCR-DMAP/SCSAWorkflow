# tests/templates/test_load_csv_template.py
"""Unit tests for the Load CSV Files template."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.load_csv_files_with_config import run_from_json


def create_mock_csv_files(tmp_dir: Path) -> tuple:
    """Create minimal CSV files for testing."""
    # Create first CSV file
    csv1_data = pd.DataFrame({
        'CellID': [1, 2, 3],
        'X_centroid': [10.0, 20.0, 30.0],
        'Y_centroid': [15.0, 25.0, 35.0],
        'cell_type': ['TypeA', 'TypeB', 'TypeA']
    })
    csv1_path = tmp_dir / 'sample1.csv'
    csv1_data.to_csv(csv1_path, index=False)

    # Create second CSV file
    csv2_data = pd.DataFrame({
        'CellID': [4, 5, 6],
        'X_centroid': [40.0, 50.0, 60.0],
        'Y_centroid': [45.0, 55.0, 65.0],
        'cell_type': ['TypeB', 'TypeB', 'TypeA']
    })
    csv2_path = tmp_dir / 'sample2.csv'
    csv2_data.to_csv(csv2_path, index=False)

    # Create configuration file
    config_data = pd.DataFrame({
        'file_name': ['sample1.csv', 'sample2.csv'],
        'slide_number': ['S1', 'S2']
    })
    config_path = tmp_dir / 'config.csv'
    config_data.to_csv(config_path, index=False)

    return csv1_path, csv2_path, config_path


class TestLoadCSVFilesWithConfig(unittest.TestCase):
    """Unit tests for the Load CSV Files template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Create mock CSV files
        self.csv1, self.csv2, self.config = create_mock_csv_files(
            self.tmp_path
        )

        # Minimal parameters
        self.params = {
            "CSV_Files": str(self.tmp_path),
            "CSV_Files_Configuration": str(self.config),
            "String_Columns": [""],
            "Output_File": "combined.csv"
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering all input/output scenarios."""
        # Test 1: Run with save_results=True
        saved_files = run_from_json(self.params)
        self.assertIn("combined.csv", saved_files)
        output_path = Path(saved_files["combined.csv"])
        self.assertTrue(output_path.exists())

        # Verify content
        result_df = pd.read_csv(output_path)
        self.assertEqual(len(result_df), 6)  # 3 + 3 rows
        self.assertIn('file_name', result_df.columns)
        self.assertIn('slide_number', result_df.columns)
        self.assertIn('CellID', result_df.columns)

        # Test 2: Run with save_results=False
        df_result = run_from_json(self.params, save_results=False)
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(len(df_result), 6)

        # Test 3: JSON file input
        json_path = self.tmp_path / "params.json"
        with open(json_path, "w") as f:
            json.dump(self.params, f)
        saved_from_json = run_from_json(json_path)
        self.assertIn("combined.csv", saved_from_json)

        # Test 4: String columns specified
        params_with_strings = self.params.copy()
        params_with_strings["String_Columns"] = ["CellID"]
        df_with_strings = run_from_json(params_with_strings, save_results=False)
        self.assertEqual(df_with_strings['CellID'].dtype, 'object')

        # Test 5: Special characters in column names
        special_csv = pd.DataFrame({
            'Cell-ID': [1, 2],
            'Area µm²': [100.0, 200.0],
            'CD4+': ['pos', 'neg']
        })
        special_path = self.tmp_path / 'special.csv'
        special_csv.to_csv(special_path, index=False)

        special_config = pd.DataFrame({
            'file_name': ['special.csv'],
            'experiment': ['Exp1']
        })
        special_config_path = self.tmp_path / 'special_config.csv'
        special_config.to_csv(special_config_path, index=False)

        params_special = {
            "CSV_Files": str(self.tmp_path),
            "CSV_Files_Configuration": str(special_config_path),
            "String_Columns": [""]
        }
        df_special = run_from_json(params_special, save_results=False)
        # Check column names were cleaned
        self.assertIn('Cell_ID', df_special.columns)
        self.assertIn('Area_um2', df_special.columns)
        self.assertIn('CD4_pos', df_special.columns)

    def test_error_messages(self) -> None:
        """Test exact error messages for various failure scenarios."""
        # Test 1: Missing CSV file
        bad_config = pd.DataFrame({
            'file_name': ['missing.csv'],
            'slide_number': ['S1']
        })
        bad_config_path = self.tmp_path / 'bad_config.csv'
        bad_config.to_csv(bad_config_path, index=False)

        params_missing = self.params.copy()
        params_missing["CSV_Files_Configuration"] = str(bad_config_path)

        with self.assertRaises(TypeError) as context:
            run_from_json(params_missing)
        expected_msg = "The following files are not found: missing.csv"
        self.assertEqual(expected_msg, str(context.exception))

        # Test 2: Empty CSV file
        empty_path = self.tmp_path / 'empty.csv'
        empty_path.write_text('')

        empty_config = pd.DataFrame({
            'file_name': ['empty.csv'],
            'slide_number': ['S1']
        })
        empty_config_path = self.tmp_path / 'empty_config.csv'
        empty_config.to_csv(empty_config_path, index=False)

        params_empty = self.params.copy()
        params_empty["CSV_Files_Configuration"] = str(empty_config_path)

        with self.assertRaises(TypeError) as context:
            run_from_json(params_empty)
        expected_msg = 'The file: "empty.csv" is empty.'
        self.assertEqual(expected_msg, str(context.exception))

        # Test 3: Invalid CSV file
        invalid_path = self.tmp_path / 'invalid.csv'
        # Create a truly invalid CSV that will cause parser error
        invalid_path.write_text('col1,col2,col3\n"unclosed quote,value2,value3\nvalue4,value5')

        invalid_config = pd.DataFrame({
            'file_name': ['invalid.csv'],
            'slide_number': ['S1']
        })
        invalid_config_path = self.tmp_path / 'invalid_config.csv'
        invalid_config.to_csv(invalid_config_path, index=False)

        params_invalid = self.params.copy()
        params_invalid["CSV_Files_Configuration"] = str(invalid_config_path)

        with self.assertRaises(TypeError) as context:
            run_from_json(params_invalid)
        expected_msg = (
            'The file "invalid.csv" could not be parsed. '
            'Please check that the file is a valid CSV.'
        )
        self.assertEqual(expected_msg, str(context.exception))

        # Test 4: Invalid string_columns parameter
        params_bad_strings = self.params.copy()
        params_bad_strings["String_Columns"] = "not_a_list"

        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad_strings)
        expected_msg = (
            "String Columns must be a *list* of column names (strings)."
        )
        self.assertEqual(expected_msg, str(context.exception))

    def test_metadata_mapping(self) -> None:
        """Test that metadata columns are correctly mapped."""
        df_result = run_from_json(self.params, save_results=False)

        # Check slide_number mapping
        sample1_rows = df_result[df_result['file_name'] == 'sample1.csv']
        sample2_rows = df_result[df_result['file_name'] == 'sample2.csv']

        self.assertTrue(all(sample1_rows['slide_number'] == 'S1'))
        self.assertTrue(all(sample2_rows['slide_number'] == 'S2'))

    @patch('builtins.print')
    def test_console_output(self, mock_print) -> None:
        """Test that progress is printed to console."""
        run_from_json(self.params, save_results=False)

        # Check for expected print statements
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list
                      if call[0]]
        
        # Should print processing messages
        processing_msgs = [msg for msg in print_calls 
                          if 'Processing file:' in msg]
        self.assertEqual(len(processing_msgs), 2)  # Two files

        # Should print completion message
        completion_msgs = [msg for msg in print_calls
                          if 'Load CSV Files completed' in msg]
        self.assertTrue(len(completion_msgs) > 0)

    def test_duplicate_file_handling(self) -> None:
        """Test handling of duplicate file names in config."""
        # Create config with duplicate entries
        dup_config = pd.DataFrame({
            'file_name': ['sample1.csv', 'sample1.csv'],
            'slide_number': ['S1', 'S2']
        })
        dup_config_path = self.tmp_path / 'dup_config.csv'
        dup_config.to_csv(dup_config_path, index=False)

        params_dup = self.params.copy()
        params_dup["CSV_Files_Configuration"] = str(dup_config_path)

        with self.assertRaises(RuntimeError) as context:
            run_from_json(params_dup)
        self.assertIn(
            "Failed to process CSV files", 
            str(context.exception)
        )

    def test_string_columns_validation(self) -> None:
        """Test validation of string_columns parameter."""
        # Test non-existent column
        params_bad_col = self.params.copy()
        params_bad_col["String_Columns"] = ["NonExistentColumn"]

        with self.assertRaises(ValueError):
            run_from_json(params_bad_col)

        # Test None handling
        params_none = self.params.copy()
        params_none["String_Columns"] = ["None"]
        df_none = run_from_json(params_none, save_results=False)
        self.assertIsInstance(df_none, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()