# tests/templates/test_utag_clustering_template.py
"""Unit tests for the UTAG Clustering template."""

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

from spac.templates.utag_clustering_template import run_from_json


def mock_adata(n_cells: int = 50) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "slide_id": (["Slide1", "Slide2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 5))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Marker1", "Marker2", "Marker3", "Marker4", "Marker5"]
    # Add spatial coordinates required for UTAG
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    return adata


class TestUTAGClusteringTemplate(unittest.TestCase):
    """Unit tests for the UTAG Clustering template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - adjust based on template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "Features": ["All"],
            "Slide_Annotation": "None",
            "Distance_Threshold": 20.0,
            "K_Nearest_Neighbors": 15,
            "Resolution_Parameter": 1,
            "PCA_Components": "None",
            "Random_Seed": 42,
            "N_Jobs": 1,
            "Leiden_Iterations": 5,
            "Parallel_Processes": False,
            "Output_Annotation_Name": "UTAG",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Mock the run_utag_clustering function
            with patch('spac.templates.utag_clustering_template.'
                      'run_utag_clustering') as mock_utag:
                # Mock function adds UTAG column to adata.obs
                def add_utag_column(adata, **kwargs):
                    # Add mock UTAG clusters
                    n_cells = adata.n_obs
                    clusters = ["UTAG_" + str(i % 3) 
                               for i in range(n_cells)]
                    adata.obs[kwargs['output_annotation']] = \
                        pd.Categorical(clusters)
                
                mock_utag.side_effect = add_utag_column
                
                # Test 1: Run with default parameters
                result = run_from_json(self.params)
                self.assertIsInstance(result, dict)
                self.assertTrue(len(result) > 0)
                
                # Test 2: Run without saving
                result_no_save = run_from_json(
                    self.params, save_results=False
                )
                # Check appropriate return type based on template
                self.assertIsInstance(result_no_save, ad.AnnData)
                self.assertIn("UTAG", result_no_save.obs.columns)
                
                # Test 3: JSON file input
                json_path = os.path.join(
                    self.tmp_dir.name, "params.json"
                )
                with open(json_path, "w") as f:
                    json.dump(self.params, f)
                
                result_json = run_from_json(json_path)
                self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid integer conversion for principal_components
        params_bad = self.params.copy()
        params_bad["PCA_Components"] = "invalid_number"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check exact error message
        expected_msg = (
            "Error: can't convert principal_components to integer. "
            "Received:\"invalid_number\""
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.utag_clustering_template.run_utag_clustering')
    def test_function_calls(self, mock_utag) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function
        def add_utag_column(adata, **kwargs):
            adata.obs[kwargs['output_annotation']] = pd.Categorical(
                ["UTAG_0"] * adata.n_obs
            )
        
        mock_utag.side_effect = add_utag_column
        
        # Test with specific features instead of "All"
        params_features = self.params.copy()
        params_features["Features"] = ["Marker1", "Marker3"]
        params_features["Slide_Annotation"] = "slide_id"
        params_features["PCA_Components"] = "10"
        
        run_from_json(params_features, save_results=False)
        
        # Verify function was called correctly
        mock_utag.assert_called_once()
        call_args = mock_utag.call_args
        
        # Check parameter conversions
        self.assertEqual(call_args[1]['features'], ["Marker1", "Marker3"])
        self.assertEqual(call_args[1]['k'], 15)
        self.assertEqual(call_args[1]['resolution'], 1)
        self.assertEqual(call_args[1]['max_dist'], 20.0)
        self.assertEqual(call_args[1]['n_pcs'], 10)
        self.assertEqual(call_args[1]['slide_key'], "slide_id")
        self.assertEqual(call_args[1]['layer'], None)  # "Original" → None
        self.assertEqual(call_args[1]['output_annotation'], "UTAG")
        self.assertEqual(call_args[1]['parallel'], False)


if __name__ == "__main__":
    unittest.main()