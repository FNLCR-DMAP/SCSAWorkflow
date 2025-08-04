# tests/templates/test_select_values_template.py
"""Unit tests for the Select Values template."""

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

from spac.templates.select_values_template import run_from_json


def mock_dataframe(n_rows: int = 10) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame for fast tests."""
    return pd.DataFrame({
        "file_name": [f"Halo_Synthetic_Example_{i % 3 + 1}" 
                      for i in range(n_rows)],
        "cell_type": (["TypeA", "TypeB"] * ((n_rows + 1) // 2))[:n_rows],
        "marker_value": range(n_rows)
    })


class TestSelectValuesTemplate(unittest.TestCase):
    """Unit tests for the Select Values template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.csv"
        )
        self.out_file = "select_values.csv"

        # Save minimal mock data
        mock_df = mock_dataframe()
        mock_df.to_csv(self.in_file, index=False)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Dataset": self.in_file,
            "Annotation_of_Interest": "file_name",
            "Label_s_of_Interest": ["Halo_Synthetic_Example_1"],
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertIn(self.out_file, result)
            
            # Verify the output file exists and has correct content
            output_path = result[self.out_file]
            self.assertTrue(os.path.exists(output_path))
            
            # Read and verify filtered data
            filtered_df = pd.read_csv(output_path)
            self.assertEqual(
                filtered_df["file_name"].unique().tolist(),
                ["Halo_Synthetic_Example_1"]
            )
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            self.assertIsInstance(result_no_save, pd.DataFrame)
            self.assertEqual(
                result_no_save["file_name"].unique().tolist(),
                ["Halo_Synthetic_Example_1"]
            )
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test with non-existent annotation
        params_bad = self.params.copy()
        params_bad["Annotation_of_Interest"] = "non_existent_column"
        
        with self.assertRaises(Exception):
            # The select_values function should raise an error
            run_from_json(params_bad)

    @patch('spac.templates.select_values_template.select_values')
    def test_function_calls(self, mock_select) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the select_values function
        mock_df = mock_dataframe(3)
        mock_select.return_value = mock_df
        
        run_from_json(self.params)
        
        # Verify function was called correctly
        mock_select.assert_called_once()
        call_args = mock_select.call_args
        
        # Check the arguments
        self.assertEqual(call_args[1]['annotation'], "file_name")
        self.assertEqual(
            call_args[1]['values'], 
            ["Halo_Synthetic_Example_1"]
        )

    def test_multiple_file_formats(self) -> None:
        """Test loading from different file formats."""
        # Test CSV (already covered above)
        
        # Test pickle format
        pickle_file = os.path.join(self.tmp_dir.name, "input.pkl")
        mock_df = mock_dataframe()
        with open(pickle_file, 'wb') as f:
            pickle.dump(mock_df, f)
        
        params_pickle = self.params.copy()
        params_pickle["Upstream_Dataset"] = pickle_file
        
        result = run_from_json(params_pickle, save_results=False)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Test direct DataFrame input
        params_df = self.params.copy()
        params_df["Upstream_Dataset"] = mock_df
        
        result = run_from_json(params_df, save_results=False)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()