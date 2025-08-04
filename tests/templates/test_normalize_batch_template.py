# tests/templates/test_normalize_batch_template.py
"""Unit tests for the Normalize Batch template."""

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

from spac.templates.normalize_batch_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "batch": (["Batch1", "Batch2"] * ((n_cells + 1) // 2))[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    # Add an empty layers dict
    adata.layers = {}
    return adata


class TestNormalizeBatchTemplate(unittest.TestCase):
    """Unit tests for the Normalize Batch template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from JSON template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Need_Normalization": False,
            "Input_Table_Name": "Original",
            "Output_Table_Name": "batch_normalized_table",
            "Annotation": "batch",
            "Normalization_Method": "median",
            "Take_Log": False,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.normalize_batch_template.batch_normalize')
    def test_complete_io_workflow(self, mock_batch_normalize) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the batch_normalize function
        def mock_normalize(adata, **kwargs):
            # Simulate normalization by adding a layer
            adata.layers[kwargs['output_layer']] = adata.X.copy()
            return adata

        mock_batch_normalize.side_effect = mock_normalize

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with Need_Normalization=False (no normalization)
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertTrue(len(result) > 0)
            # Verify batch_normalize was NOT called
            mock_batch_normalize.assert_not_called()
            
            # Test 2: Run with Need_Normalization=True
            params_norm = self.params.copy()
            params_norm["Need_Normalization"] = True
            
            result_norm = run_from_json(params_norm)
            self.assertIsInstance(result_norm, dict)
            # Verify batch_normalize WAS called
            mock_batch_normalize.assert_called_once()
            
            # Test 3: Run without saving
            result_no_save = run_from_json(params_norm, save_results=False)
            self.assertIsInstance(result_no_save, ad.AnnData)
            
            # Test 4: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    @patch('spac.templates.normalize_batch_template.batch_normalize')
    def test_function_calls(self, mock_batch_normalize) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the batch_normalize function to modify adata in-place
        def mock_normalize(adata, **kwargs):
            # Simulate normalization by adding a layer
            output_layer = kwargs.get('output_layer', 'batch_normalized_table')
            adata.layers[output_layer] = adata.X.copy()
            # batch_normalize modifies in-place, returns None
            return None
        
        mock_batch_normalize.side_effect = mock_normalize
        
        # Test with normalization enabled
        params_enabled = self.params.copy()
        params_enabled["Need_Normalization"] = True
        params_enabled["Normalization_Method"] = "z-score"
        params_enabled["Take_Log"] = True
        
        run_from_json(params_enabled, save_results=False)
        
        # Verify function was called correctly
        mock_batch_normalize.assert_called_once()
        call_args = mock_batch_normalize.call_args
        
        # Check specific parameter conversions
        self.assertEqual(call_args[1]['annotation'], "batch")
        # "Original" → None
        self.assertEqual(call_args[1]['input_layer'], None)
        self.assertEqual(
            call_args[1]['output_layer'], "batch_normalized_table"
        )
        self.assertEqual(call_args[1]['method'], "z-score")
        self.assertEqual(call_args[1]['log'], True)

    def test_parameter_defaults(self) -> None:
        """Test that default parameters work correctly."""
        # Minimal required parameters only
        params_minimal = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "batch"
        }
        
        with patch('spac.templates.normalize_batch_template.batch_normalize'):
            result = run_from_json(params_minimal)
            self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()