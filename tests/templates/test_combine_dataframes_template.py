# tests/templates/test_combine_dataframes_template.py
"""Unit tests for the Combine DataFrames template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock

import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.combine_dataframes_template import run_from_json


def mock_dataframe(n_rows: int = 10, prefix: str = "") -> pd.DataFrame:
    """Return a minimal synthetic DataFrame for fast tests."""
    return pd.DataFrame({
        f'{prefix}col1': range(n_rows),
        f'{prefix}col2': [f'val_{i}' for i in range(n_rows)],
        f'{prefix}col3': [i * 2.5 for i in range(n_rows)]
    })


class TestCombineDataFramesTemplate(unittest.TestCase):
    """Unit tests for the Combine DataFrames template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        
        # Create test CSV files
        self.csv_file1 = os.path.join(
            self.tmp_dir.name, "dataframe1.csv"
        )
        self.csv_file2 = os.path.join(
            self.tmp_dir.name, "dataframe2.csv"
        )
        
        # Create test pickle files
        self.pkl_file1 = os.path.join(
            self.tmp_dir.name, "dataframe1.pkl"
        )
        self.pkl_file2 = os.path.join(
            self.tmp_dir.name, "dataframe2.pkl"
        )
        
        # Save test dataframes
        df1 = mock_dataframe(10, 'A_')
        df2 = mock_dataframe(15, 'B_')
        
        df1.to_csv(self.csv_file1, index=False)
        df2.to_csv(self.csv_file2, index=False)
        
        with open(self.pkl_file1, 'wb') as f:
            pickle.dump(df1, f)
        with open(self.pkl_file2, 'wb') as f:
            pickle.dump(df2, f)
        
        self.out_file = "combined_output.csv"

        # Minimal parameters
        self.params = {
            "First_Dataframe": self.csv_file1,
            "Second_Dataframe": self.csv_file2,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with CSV files
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertIn(self.out_file, result)
            
            # Verify combined file exists and has correct shape
            combined_df = pd.read_csv(result[self.out_file])
            self.assertEqual(len(combined_df), 25)  # 10 + 15 rows
            self.assertEqual(len(combined_df.columns), 6)  # 3 + 3 columns
            
            # Test 2: Run with pickle files
            params_pkl = self.params.copy()
            params_pkl["First_Dataframe"] = self.pkl_file1
            params_pkl["Second_Dataframe"] = self.pkl_file2
            
            result_pkl = run_from_json(params_pkl)
            self.assertIsInstance(result_pkl, dict)
            
            # Test 3: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            self.assertIsInstance(result_no_save, pd.DataFrame)
            self.assertEqual(len(result_no_save), 25)
            
            # Test 4: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test unsupported file format
        params_bad = self.params.copy()
        params_bad["First_Dataframe"] = "file.txt"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check error message contains expected text
        self.assertIn("Unsupported file format", str(context.exception))
        self.assertIn(".txt", str(context.exception))

    @patch('spac.templates.combine_dataframes_template.combine_dfs')
    def test_function_calls(self, mock_combine) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the combine_dfs function
        mock_combine.return_value = mock_dataframe(25, 'combined_')
        
        run_from_json(self.params)
        
        # Verify function was called correctly
        mock_combine.assert_called_once()
        # Check that two dataframes were passed
        call_args = mock_combine.call_args[0][0]
        self.assertEqual(len(call_args), 2)
        self.assertIsInstance(call_args[0], pd.DataFrame)
        self.assertIsInstance(call_args[1], pd.DataFrame)

    def test_direct_dataframe_input(self) -> None:
        """Test passing DataFrames directly instead of file paths."""
        df1 = mock_dataframe(5, 'X_')
        df2 = mock_dataframe(8, 'Y_')
        
        params_df = {
            "First_Dataframe": df1,
            "Second_Dataframe": df2,
            "Output_File": "direct_df_output.csv"
        }
        
        result = run_from_json(params_df, save_results=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 13)  # 5 + 8 rows


if __name__ == "__main__":
    unittest.main()