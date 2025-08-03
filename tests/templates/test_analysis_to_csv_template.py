# tests/templates/test_analysis_to_csv_template.py
"""Unit tests for the Analysis to CSV template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.analysis_to_csv_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "batch": (["Batch1", "Batch2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Marker1", "Marker2", "Marker3"]
    # Add spatial coordinates
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    # Add a test layer
    adata.layers["normalized"] = rng.normal(size=(n_cells, 3))
    return adata


class TestAnalysisToCSVTemplate(unittest.TestCase):
    """Unit tests for the Analysis to CSV template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "analysis.csv"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Export": "Original",
            "Save_as_CSV_File": False,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with Save_as_CSV_File=False (returns DataFrame)
            result = run_from_json(self.params)
            self.assertIsInstance(result, pd.DataFrame)
            # Verify dataframe has expected columns
            self.assertIn("Marker1", result.columns)
            self.assertIn("Marker2", result.columns)
            self.assertIn("Marker3", result.columns)
            self.assertIn("cell_type", result.columns)
            self.assertIn("spatial_x", result.columns)
            self.assertIn("spatial_y", result.columns)
            
            # Test 2: Run with Save_as_CSV_File=True
            params_save = self.params.copy()
            params_save["Save_as_CSV_File"] = True
            result_save = run_from_json(params_save)
            self.assertIsInstance(result_save, dict)
            self.assertIn(self.out_file, result_save)
            
            # Test 3: Export specific layer
            params_layer = self.params.copy()
            params_layer["Table_to_Export"] = "normalized"
            result_layer = run_from_json(params_layer)
            self.assertIsInstance(result_layer, pd.DataFrame)
            # Verify it has the normalized data
            self.assertEqual(len(result_layer), 10)  # n_cells
            
            # Test 4: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, pd.DataFrame)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test missing layer
        params_bad = self.params.copy()
        params_bad["Table_to_Export"] = "nonexistent_layer"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check that error mentions the missing layer
        self.assertIn("nonexistent_layer", str(context.exception))

    @patch('spac.templates.analysis_to_csv_template.check_table')
    def test_function_calls(self, mock_check) -> None:
        """Test that main function is called with correct parameters."""
        # Run with a specific layer
        params_with_layer = self.params.copy()
        params_with_layer["Table_to_Export"] = "normalized"
        
        result = run_from_json(params_with_layer)
        
        # Verify check_table was called for the layer
        mock_check.assert_called_once()
        call_args = mock_check.call_args
        self.assertEqual(call_args[1]['tables'], "normalized")
        
        # Verify result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()