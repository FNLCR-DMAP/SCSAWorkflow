# tests/templates/test_calculate_centroid_template.py
"""Unit tests for the Calculate Centroid template."""

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

from spac.templates.calculate_centroid_template import run_from_json


def mock_dataframe(n_cells: int = 10) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame for fast tests."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "XMin": rng.uniform(0, 50, n_cells),
        "XMax": rng.uniform(50, 100, n_cells),
        "YMin": rng.uniform(0, 50, n_cells),
        "YMax": rng.uniform(50, 100, n_cells),
        "CellID": range(n_cells),
        "CellType": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells]
    })


class TestCalculateCentroidTemplate(unittest.TestCase):
    """Unit tests for the Calculate Centroid template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.csv"
        )
        self.out_file = "centroid_calculated"

        # Save minimal mock data as CSV
        mock_df = mock_dataframe()
        mock_df.to_csv(self.in_file, index=False)

        # Minimal parameters from JSON template
        self.params = {
            "Upstream_Dataset": self.in_file,
            "Min_X_Coordinate_Column_Name": "XMin",
            "Max_X_Coordinate_Column_Name": "XMax",
            "Min_Y_Coordinate_Column_Name": "YMin",
            "Max_Y_Coordinate_Column_Name": "YMax",
            "X_Centroid_Name": "XCentroid",
            "Y_Centroid_Name": "YCentroid",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters (CSV input)
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            # Verify file was saved
            self.assertTrue(len(result) > 0)
            csv_files = [f for f in result.values() if '.csv' in str(f)]
            self.assertTrue(len(csv_files) > 0)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type
            self.assertIsInstance(result_no_save, pd.DataFrame)
            # Verify centroids were calculated
            self.assertIn("XCentroid", result_no_save.columns)
            self.assertIn("YCentroid", result_no_save.columns)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)
            
            # Test 4: Pickle file input
            pickle_file = os.path.join(self.tmp_dir.name, "input.pickle")
            mock_df = mock_dataframe()
            with open(pickle_file, 'wb') as f:
                pickle.dump(mock_df, f)
            
            params_pickle = self.params.copy()
            params_pickle["Upstream_Dataset"] = pickle_file
            result_pickle = run_from_json(params_pickle, save_results=False)
            self.assertIsInstance(result_pickle, pd.DataFrame)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test unsupported file format
        params_bad = self.params.copy()
        params_bad["Upstream_Dataset"] = "invalid.txt"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check error message
        self.assertIn("Unsupported file format", str(context.exception))

    @patch('spac.templates.calculate_centroid_template.calculate_centroid')
    def test_function_calls(self, mock_calc) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the calculate_centroid function
        mock_df = mock_dataframe()
        mock_df["XCentroid"] = (mock_df["XMin"] + mock_df["XMax"]) / 2
        mock_df["YCentroid"] = (mock_df["YMin"] + mock_df["YMax"]) / 2
        mock_calc.return_value = mock_df
        
        run_from_json(self.params, save_results=False)
        
        # Verify function was called correctly
        mock_calc.assert_called_once()
        call_kwargs = mock_calc.call_args[1]
        
        # Check specific parameter conversions
        self.assertEqual(call_kwargs['x_min'], "XMin")
        self.assertEqual(call_kwargs['x_max'], "XMax")
        self.assertEqual(call_kwargs['y_min'], "YMin")
        self.assertEqual(call_kwargs['y_max'], "YMax")
        self.assertEqual(call_kwargs['new_x'], "XCentroid")
        self.assertEqual(call_kwargs['new_y'], "YCentroid")

    def test_direct_dataframe_input(self) -> None:
        """Test that DataFrame can be passed directly."""
        mock_df = mock_dataframe()
        params_df = self.params.copy()
        params_df["Upstream_Dataset"] = mock_df
        
        calc_centroid_patch = (
            'spac.templates.calculate_centroid_template.calculate_centroid'
        )
        with patch(calc_centroid_patch) as mock_calc:
            # Mock return value
            result_df = mock_df.copy()
            result_df["XCentroid"] = 75.0
            result_df["YCentroid"] = 75.0
            mock_calc.return_value = result_df
            
            result = run_from_json(params_df, save_results=False)
            self.assertIsInstance(result, pd.DataFrame)
            
            # Verify the input DataFrame was passed correctly
            call_args = mock_calc.call_args[0]
            pd.testing.assert_frame_equal(call_args[0], mock_df)


if __name__ == "__main__":
    unittest.main()