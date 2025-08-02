# tests/templates/test_arcsinh_normalization_template.py
"""Unit tests for the Arcsinh Normalization template."""

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

from spac.templates.arcsinh_normalization_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    # Create expression data with some high values to normalize
    x_mat = rng.exponential(scale=10, size=(n_cells, 5))
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "batch": (["Batch1", "Batch2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Marker{i}" for i in range(5)]
    # Add an empty layers dict
    adata.layers = {}
    return adata


class TestArcsinhNormalizationTemplate(unittest.TestCase):
    """Unit tests for the Arcsinh Normalization template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "Co_Factor": "5.0",
            "Percentile": "None",
            "Output_Table_Name": "arcsinh",
            "Per_Batch": "False",
            "Annotation": "None",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.arcsinh_normalization_template.arcsinh_transformation')
    def test_complete_io_workflow(self, mock_arcsinh) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the arcsinh_transformation function
        def mock_transform(adata, **kwargs):
            # Simulate transformation by adding a layer
            adata.layers[kwargs['output_layer']] = np.log1p(adata.X) / 5.0
            return adata

        mock_arcsinh.side_effect = mock_transform

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            # Verify file was saved
            self.assertTrue(len(result) > 0)
            pickle_files = [f for f in result.values() if '.pickle' in str(f)]
            self.assertTrue(len(pickle_files) > 0)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type based on template
            self.assertIsInstance(result_no_save, ad.AnnData)
            self.assertIn("arcsinh", result_no_save.layers)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

        # Verify arcsinh_transformation was called with correct parameters
        call_args = mock_arcsinh.call_args
        self.assertEqual(call_args[1]['input_layer'], None)  # "Original" → None
        self.assertEqual(call_args[1]['co_factor'], 5.0)
        self.assertEqual(call_args[1]['percentile'], None)
        self.assertEqual(call_args[1]['output_layer'], "arcsinh")
        self.assertEqual(call_args[1]['per_batch'], False)
        self.assertEqual(call_args[1]['annotation'], None)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid float conversion for co_factor
        params_bad = self.params.copy()
        params_bad["Co_Factor"] = "invalid_number"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check exact error message
        expected_msg = (
            "Error: can't convert co_factor to float. "
            "Received:\"invalid_number\""
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.arcsinh_normalization_template.arcsinh_transformation')
    def test_function_calls(self, mock_arcsinh) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function to return transformed data
        mock_arcsinh.return_value = mock_adata()
        mock_arcsinh.return_value.layers["arcsinh"] = np.zeros((10, 5))
        
        # Test with percentile instead of co_factor
        params_alt = self.params.copy()
        params_alt["Co_Factor"] = "None"
        params_alt["Percentile"] = "10"
        params_alt["Per_Batch"] = "True"
        params_alt["Annotation"] = "batch"
        
        run_from_json(params_alt, save_results=False)
        
        # Verify function was called correctly
        mock_arcsinh.assert_called_once()
        call_args = mock_arcsinh.call_args
        
        # Check specific parameter conversions
        self.assertEqual(call_args[1]['co_factor'], None)
        self.assertEqual(call_args[1]['percentile'], 10.0)
        self.assertEqual(call_args[1]['per_batch'], True)
        self.assertEqual(call_args[1]['annotation'], "batch")


if __name__ == "__main__":
    unittest.main()