# tests/templates/test_phenograph_clustering_template.py
"""Unit tests for the Phenograph Clustering template."""

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

from spac.templates.phenograph_clustering_template import run_from_json


def mock_adata(n_cells: int = 100) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    # Create expression matrix with some structure for clustering
    x_mat = rng.normal(size=(n_cells, 10))
    # Add some signal to make clustering meaningful
    # First half of cells higher in first 5 genes
    x_mat[:n_cells//2, :5] += 2.0
    # Second half higher in last 5 genes
    x_mat[n_cells//2:, 5:] += 2.0
    
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(10)]
    adata.var.index.name = None  # Match NIDAP style
    return adata


class TestPhenographClusteringTemplate(unittest.TestCase):
    """Unit tests for the Phenograph Clustering template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - using proper Python types for defaults
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "K_Nearest_Neighbors": 30,
            "Seed": 42,
            "Resolution_Parameter": 1.0,
            "Output_Annotation_Name": "phenograph",
            "Resolution_List": [],
            "Number_of_Iterations": 100,
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
            # Verify file was saved
            self.assertTrue(len(result) > 0)
            output_path = list(result.values())[0]
            self.assertTrue(os.path.exists(output_path))
            
            # Load and verify the output
            with open(output_path, 'rb') as f:
                adata_out = pickle.load(f)
            self.assertIn("phenograph", adata_out.obs.columns)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type based on template
            self.assertIsInstance(result_no_save, ad.AnnData)
            self.assertIn("phenograph", result_no_save.obs.columns)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_function_calls(self) -> None:
        """Test that main function is called with correct parameters."""
        with patch('spac.templates.phenograph_clustering_template.'
                   'phenograph_clustering') as mock_pheno:
            # Mock the main function
            def mock_clustering(adata, **kwargs):
                # Add the phenograph column
                n_half = len(adata) // 2
                adata.obs['phenograph'] = pd.Categorical(
                    ['0'] * n_half + ['1'] * (len(adata) - n_half)
                )
                return None
            
            mock_pheno.side_effect = mock_clustering
            
            # Test with custom annotation name
            params_custom = self.params.copy()
            params_custom["Output_Annotation_Name"] = "my_clusters"
            params_custom["K_Nearest_Neighbors"] = 50
            params_custom["Resolution_Parameter"] = 0.5
            
            result = run_from_json(params_custom, save_results=False)
            
            # Verify function was called correctly
            mock_pheno.assert_called_once()
            call_args = mock_pheno.call_args
            
            # Check specific parameter conversions
            self.assertEqual(call_args[1]['k'], 50)
            self.assertEqual(call_args[1]['resolution_parameter'], 0.5)
            self.assertEqual(call_args[1]['seed'], 42)
            self.assertEqual(call_args[1]['n_iterations'], 100)
            # "Original" -> None
            self.assertEqual(call_args[1]['layer'], None)
            
            # Check that phenograph was renamed
            self.assertIn("my_clusters", result.obs.columns)
            self.assertNotIn("phenograph", result.obs.columns)


if __name__ == "__main__":
    unittest.main()