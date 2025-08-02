# tests/templates/test_zscore_normalization_template.py
"""Unit tests for the Z-Score Normalization template."""

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

from spac.templates.zscore_normalization_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Marker1", "Marker2", "Marker3"]
    
    # Add a test layer with different values
    adata.layers["test_layer"] = rng.normal(loc=5, scale=2, size=(n_cells, 3))
    
    return adata


class TestZScoreNormalizationTemplate(unittest.TestCase):
    """Unit tests for the Z-Score Normalization template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "normalized_output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from JSON template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "Output_Table_Name": "z_scores",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.zscore_normalization_template.'
           'z_score_normalization')
    def test_complete_io_workflow(self, mock_z_score_norm) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Mock the z_score_normalization function
            def mock_z_score_side_effect(adata, **kwargs):
                # Simulate what z_score_normalization does
                output_layer = kwargs.get('output_layer', 'z_scores')
                input_layer = kwargs.get('input_layer', None)
                
                # Create normalized data
                if input_layer is None:
                    data = adata.X
                else:
                    data = adata.layers[input_layer]
                
                # Simple z-score normalization simulation
                normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
                adata.layers[output_layer] = normalized
                return None
            
            mock_z_score_norm.side_effect = mock_z_score_side_effect
            
            # Test 1: Run with default parameters (Original layer)
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            
            # Verify z_score_normalization was called correctly
            mock_z_score_norm.assert_called_once()
            call_args = mock_z_score_norm.call_args
            self.assertEqual(call_args[1]['output_layer'], "z_scores")
            # Original -> None
            self.assertEqual(call_args[1]['input_layer'], None)
            
            # Test 2: Run without saving
            mock_z_score_norm.reset_mock()
            result_no_save = run_from_json(self.params, save_results=False)
            self.assertIsInstance(result_no_save, ad.AnnData)
            self.assertIn("z_scores", result_no_save.layers)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            mock_z_score_norm.reset_mock()
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)
            
            # Test 4: Using a specific layer
            params_with_layer = self.params.copy()
            params_with_layer["Table_to_Process"] = "test_layer"
            params_with_layer["Output_Table_Name"] = "test_z_scores"
            
            mock_z_score_norm.reset_mock()
            result_layer = run_from_json(params_with_layer, save_results=False)
            
            # Verify correct layer was used
            call_args = mock_z_score_norm.call_args
            self.assertEqual(call_args[1]['input_layer'], "test_layer")
            self.assertEqual(call_args[1]['output_layer'], "test_z_scores")

    @patch('builtins.print')
    @patch('spac.templates.zscore_normalization_template.'
           'z_score_normalization')
    def test_console_output(
        self, mock_z_score_norm, mock_print
    ) -> None:
        """Test that summary statistics are printed to console."""
        # Mock z_score_normalization to create the expected layer
        def mock_z_score_side_effect(adata, **kwargs):
            output_layer = kwargs.get('output_layer', 'z_scores')
            # Create dummy normalized data
            adata.layers[output_layer] = np.zeros_like(adata.X)
            return None
        
        mock_z_score_norm.side_effect = mock_z_score_side_effect
        
        run_from_json(self.params, save_results=False)
        
        # Check that describe() output was printed
        print_calls = [
            str(call[0][0]) for call in mock_print.call_args_list
            if call[0]
        ]
        
        # Should contain statistical summary
        summary_printed = any(
            'mean' in str(call).lower() or 
            'std' in str(call).lower() or
            'count' in str(call).lower()
            for call in print_calls
        )
        self.assertTrue(
            summary_printed, 
            "DataFrame summary statistics were not printed"
        )
        
        # Should print adata info
        adata_printed = any(
            'AnnData' in str(call) for call in print_calls
        )
        self.assertTrue(adata_printed, "AnnData info was not printed")

    def test_missing_upstream_analysis_error(self) -> None:
        """Test exact error for missing Upstream_Analysis parameter."""
        params_bad = self.params.copy()
        del params_bad["Upstream_Analysis"]
        
        with self.assertRaises(KeyError) as context:
            run_from_json(params_bad)
        
        self.assertIn("Upstream_Analysis", str(context.exception))

    @patch('spac.templates.zscore_normalization_template.'
           'z_score_normalization')
    def test_output_file_extension_handling(
        self, mock_z_score_norm
    ) -> None:
        """Test that output defaults to pickle format."""
        # Mock z_score_normalization to create the expected layer
        def mock_z_score_side_effect(adata, **kwargs):
            output_layer = kwargs.get('output_layer', 'z_scores')
            # Create dummy normalized data
            adata.layers[output_layer] = np.zeros_like(adata.X)
            return None
        
        mock_z_score_norm.side_effect = mock_z_score_side_effect
        
        params = self.params.copy()
        params["Output_File"] = "results.dat"  # No standard extension
        
        saved_files = run_from_json(params)
        
        # Should save as pickle by default
        pickle_files = [f for f in saved_files.values()
                       if '.pickle' in str(f)]
        self.assertTrue(len(pickle_files) > 0)


if __name__ == "__main__":
    unittest.main()