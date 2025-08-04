# tests/templates/test_tsne_analysis_template.py
"""Unit tests for the tSNE Analysis template."""

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

from spac.templates.tsne_analysis_template import run_from_json


def mock_adata(n_cells: int = 100) -> ad.AnnData:
    """
    Return a minimal synthetic AnnData for fast tests.
    
    Note: tSNE requires n_cells > perplexity (default 30),
    so we use 100 cells by default.
    """
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 10))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(10)]
    # Add spatial coordinates if needed
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    return adata


class TestTsneAnalysisTemplate(unittest.TestCase):
    """Unit tests for the tSNE Analysis template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(100), f)

        # Minimal parameters - adjust based on template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
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
            # Verify a pickle file was created
            self.assertTrue(
                any('.pickle' in str(f) for f in result.values())
            )
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type based on template
            self.assertIsInstance(result_no_save, ad.AnnData)
            # Verify that tSNE was added to obsm
            self.assertIn("X_tsne", result_no_save.obsm)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_layer_parameter(self) -> None:
        """Test that layer parameter is handled correctly."""
        # Create adata with a layer - use 100 cells for tSNE
        adata = mock_adata(100)
        adata.layers["normalized"] = adata.X * 2
        
        with open(self.in_file, 'wb') as f:
            pickle.dump(adata, f)
        
        # Test with specific layer
        params_layer = self.params.copy()
        params_layer["Table_to_Process"] = "normalized"
        
        result = run_from_json(params_layer, save_results=False)
        self.assertIn("X_tsne", result.obsm)

    @patch('spac.templates.tsne_analysis_template.tsne')
    def test_function_calls(self, mock_tsne) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the tsne function
        def mock_tsne_func(adata, layer=None):
            # Simulate adding tSNE results
            adata.obsm["X_tsne"] = np.random.rand(adata.n_obs, 2)
            return adata
        
        mock_tsne.side_effect = mock_tsne_func
        
        run_from_json(self.params)
        
        # Verify function was called correctly
        mock_tsne.assert_called_once()
        call_args = mock_tsne.call_args
        
        # Check that layer=None when "Original" is specified
        self.assertIsNone(call_args[1]['layer'])


if __name__ == "__main__":
    unittest.main()