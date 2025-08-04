# tests/templates/test_downsample_cells_template.py
"""Unit tests for the Downsample Cells template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.downsample_cells_template import run_from_json


def mock_dataframe(n_rows: int = 1000) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame for fast tests."""
    rng = np.random.default_rng(0)
    
    # Create a dataframe with multiple annotations for downsampling
    data = {
        "cell_id": range(n_rows),
        "region": rng.choice(["region1", "region2", "region3"], n_rows),
        "day": rng.choice(["day1", "day2", "day3", "day4"], n_rows),
        "cell_type": rng.choice(["TypeA", "TypeB", "TypeC"], n_rows),
        "marker1": rng.normal(100, 15, n_rows),
        "marker2": rng.normal(50, 10, n_rows),
    }
    
    return pd.DataFrame(data)


class TestDownsampleCellsTemplate(unittest.TestCase):
    """Unit tests for the Downsample Cells template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.csv"
        )
        self.out_file = "downsampled_data"

        # Save minimal mock data as CSV
        mock_df = mock_dataframe()
        mock_df.to_csv(self.in_file, index=False)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Dataset": self.in_file,
            "Annotations_List": ["region", "day"],
            "Number_of_Samples": 100,
            "Stratify_Option": False,
            "Random_Selection": True,
            "New_Combined_Annotation_Name": "_combined_",
            "Minimum_Threshold": 5,
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
            
            # Verify file was saved with correct extension
            self.assertTrue(len(result) > 0)
            saved_file = list(result.values())[0]
            self.assertTrue(saved_file.endswith('.csv'))
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type
            self.assertIsInstance(result_no_save, pd.DataFrame)
            # Verify downsampling occurred
            self.assertLessEqual(len(result_no_save), 1000)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_different_input_formats(self) -> None:
        """Test loading from different file formats."""
        # Test pickle input
        pickle_file = os.path.join(self.tmp_dir.name, "input.pickle")
        mock_df = mock_dataframe(500)
        with open(pickle_file, 'wb') as f:
            pickle.dump(mock_df, f)
        
        params_pickle = self.params.copy()
        params_pickle["Upstream_Dataset"] = pickle_file
        
        result = run_from_json(params_pickle, save_results=False)
        self.assertIsInstance(result, pd.DataFrame)
        
        # Test direct DataFrame input
        params_df = self.params.copy()
        params_df["Upstream_Dataset"] = mock_dataframe(300)
        
        result = run_from_json(params_df, save_results=False)
        self.assertIsInstance(result, pd.DataFrame)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test unsupported file format
        params_bad = self.params.copy()
        params_bad["Upstream_Dataset"] = "data.xlsx"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check error message contains expected text
        self.assertIn("Unsupported file format", str(context.exception))
        self.assertIn(".xlsx", str(context.exception))

    @patch('spac.templates.downsample_cells_template.downsample_cells')
    def test_function_calls(self, mock_downsample) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the downsample function
        mock_downsample.return_value = pd.DataFrame({"col": [1, 2, 3]})
        
        # Test with stratify option
        params_stratify = self.params.copy()
        params_stratify["Stratify_Option"] = True
        params_stratify["Random_Selection"] = False
        params_stratify["Number_of_Samples"] = 50
        
        run_from_json(params_stratify, save_results=False)
        
        # Verify function was called correctly
        mock_downsample.assert_called_once()
        call_args = mock_downsample.call_args
        
        # Check specific parameter conversions
        self.assertEqual(call_args[1]['annotations'], ["region", "day"])
        self.assertEqual(call_args[1]['n_samples'], 50)
        self.assertEqual(call_args[1]['stratify'], True)
        self.assertEqual(call_args[1]['rand'], False)
        self.assertEqual(call_args[1]['combined_col_name'], "_combined_")
        self.assertEqual(call_args[1]['min_threshold'], 5)


if __name__ == "__main__":
    unittest.main()