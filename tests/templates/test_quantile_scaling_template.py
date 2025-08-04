# tests/templates/test_quantile_scaling_template.py
"""Unit tests for the Quantile Scaling template."""

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

from spac.templates.quantile_scaling_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    # Create expression data with variance for normalization
    x_mat = rng.exponential(scale=5, size=(n_cells, 5))
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "batch": (["Batch1", "Batch2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Marker{i}" for i in range(5)]
    # Add an existing layer for testing
    adata.layers["arcsinh"] = np.arcsinh(x_mat / 5)
    return adata


class TestQuantileScalingTemplate(unittest.TestCase):
    """Unit tests for the Quantile Scaling template."""

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
            "Table_to_Process": "arcsinh",
            "Low_Quantile": "0.02",
            "High_Quantile": "0.98",
            "Interpolation": "nearest",
            "Output_Table_Name": "scaled_arcsinh",
            "Per_Batch": "False",
            "Annotation": "",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.quantile_scaling_template.normalize_features')
    def test_complete_io_workflow(self, mock_normalize) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the normalize_features function
        def mock_transform(adata, **kwargs):
            # Simulate normalization by adding a layer
            output_layer = kwargs['output_layer']
            input_layer = kwargs.get('input_layer')
            data = adata.layers[input_layer] if input_layer else adata.X
            # Simple quantile scaling simulation
            adata.layers[output_layer] = (
                (data - np.min(data)) / (np.max(data) - np.min(data))
            )
            return adata

        mock_normalize.side_effect = mock_transform

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
            result_no_save = run_from_json(
                self.params, save_results=False, show_plot=False
            )
            # Check appropriate return type - should be tuple (adata, figure)
            self.assertIsInstance(result_no_save, tuple)
            self.assertEqual(len(result_no_save), 2)
            adata_result, fig_result = result_no_save
            self.assertIsInstance(adata_result, ad.AnnData)
            self.assertIn("scaled_arcsinh", adata_result.layers)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path, show_plot=False)
            self.assertIsInstance(result_json, dict)

        # Verify normalize_features was called with correct parameters
        call_args = mock_normalize.call_args
        self.assertEqual(call_args[1]['input_layer'], "arcsinh")
        self.assertEqual(call_args[1]['low_quantile'], 0.02)
        self.assertEqual(call_args[1]['high_quantile'], 0.98)
        self.assertEqual(call_args[1]['interpolation'], "nearest")
        self.assertEqual(call_args[1]['output_layer'], "scaled_arcsinh")
        self.assertEqual(call_args[1]['per_batch'], False)
        # Empty string becomes None
        self.assertIsNone(call_args[1]['annotation'])

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test 1: output layer already exists
        # First create adata with existing layer
        adata = mock_adata()
        adata.layers["scaled_arcsinh"] = adata.X.copy()
        
        with open(self.in_file, 'wb') as f:
            pickle.dump(adata, f)
        
        with self.assertRaises(ValueError) as context:
            run_from_json(self.params)
        
        # Check exact error message
        expected_msg = (
            "Output Table Name 'scaled_arcsinh' already exists, "
            "please rename it."
        )
        self.assertEqual(str(context.exception), expected_msg)
        
        # Test 2: Missing annotation when per_batch is True
        # Reset the input file
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)
            
        params_bad = self.params.copy()
        params_bad["Per_Batch"] = "True"
        params_bad["Annotation"] = ""  # Empty annotation
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        expected_msg = (
            'Parameter "Annotation" is required when "Per Batch" is set '
            'to True.'
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.quantile_scaling_template.normalize_features')
    @patch('builtins.print')
    def test_console_output(self, mock_print, mock_normalize) -> None:
        """Test that correct messages are printed."""
        # Mock the normalize function to return adata with the new layer
        def mock_transform(adata, **kwargs):
            output_layer = kwargs['output_layer']
            # Add the output layer
            adata.layers[output_layer] = np.ones(
                (adata.n_obs, adata.n_vars)
            )
            return adata
        
        mock_normalize.side_effect = mock_transform
        
        # Test with different quantile values
        params_alt = self.params.copy()
        params_alt["Low_Quantile"] = "0.05"
        params_alt["High_Quantile"] = "0.95"
        params_alt["Per_Batch"] = "True"
        params_alt["Annotation"] = "batch"
        
        run_from_json(params_alt, save_results=False, show_plot=False)
        
        # Check print statements
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list
                       if call[0]]
        
        # Should print quantile values
        self.assertTrue(
            any("High quantile used: 0.95" in msg for msg in print_calls)
        )
        self.assertTrue(
            any("Low quantile used: 0.05" in msg for msg in print_calls)
        )
        
        # Verify function was called with per_batch=True
        call_args = mock_normalize.call_args
        self.assertEqual(call_args[1]['per_batch'], True)
        self.assertEqual(call_args[1]['annotation'], "batch")


if __name__ == "__main__":
    unittest.main()