# tests/templates/test_umap_transformation_template.py
"""Unit tests for the UMAP transformation template."""

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

from spac.templates.umap_transformation_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 5))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(5)]
    # Add a layer to test layer processing
    adata.layers["arcsinh_z_scores"] = rng.normal(size=(n_cells, 5))
    return adata


class TestUmapTransformationTemplate(unittest.TestCase):
    """Unit tests for the UMAP transformation template."""

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
            "Number_of_Neighbors": 5,  # Small for fast test
            "Minimum_Distance_between_Points": 0.1,
            "Target_Dimension_Number": 2,
            "Computational_Metric": "euclidean",
            "Random_State": 0,
            "Transform_Seed": 42,
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
            with patch('spac.templates.umap_transformation_template.'
                      'run_umap') as mock_umap:
                # Mock the run_umap function
                mock_adata_obj = mock_adata()
                mock_adata_obj.obsm["X_umap"] = np.random.rand(10, 2)
                mock_umap.return_value = mock_adata_obj
                
                result = run_from_json(self.params)
                self.assertIsInstance(result, dict)
                # Verify file was saved
                self.assertTrue(len(result) > 0)
                
                # Verify run_umap was called with correct parameters
                mock_umap.assert_called_once()
                call_kwargs = mock_umap.call_args[1]
                self.assertEqual(call_kwargs['n_neighbors'], 5)
                self.assertEqual(call_kwargs['min_dist'], 0.1)
                self.assertEqual(call_kwargs['n_components'], 2)
                self.assertEqual(call_kwargs['metric'], 'euclidean')
                self.assertEqual(call_kwargs['random_state'], 0)
                self.assertEqual(call_kwargs['transform_seed'], 42)
                self.assertIsNone(call_kwargs['layer'])  # "Original" → None
                self.assertTrue(call_kwargs['verbose'])
            
            # Test 2: Run without saving
            with patch('spac.templates.umap_transformation_template.'
                      'run_umap') as mock_umap:
                mock_adata_obj = mock_adata()
                mock_adata_obj.obsm["X_umap"] = np.random.rand(10, 2)
                mock_umap.return_value = mock_adata_obj
                
                result_no_save = run_from_json(
                    self.params, save_results=False
                )
                # Check appropriate return type based on template
                self.assertIsInstance(result_no_save, ad.AnnData)
                self.assertIn("X_umap", result_no_save.obsm)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            with patch('spac.templates.umap_transformation_template.'
                      'run_umap') as mock_umap:
                mock_adata_obj = mock_adata()
                mock_adata_obj.obsm["X_umap"] = np.random.rand(10, 2)
                mock_umap.return_value = mock_adata_obj
                
                result_json = run_from_json(json_path)
                self.assertIsInstance(result_json, dict)

    def test_layer_parameter_handling(self) -> None:
        """Test that layer parameter is handled correctly."""
        # Test with specific layer
        params_with_layer = self.params.copy()
        params_with_layer["Table_to_Process"] = "arcsinh_z_scores"
        
        with patch('spac.templates.umap_transformation_template.'
                  'run_umap') as mock_umap:
            mock_adata_obj = mock_adata()
            mock_adata_obj.obsm["X_umap"] = np.random.rand(10, 2)
            mock_umap.return_value = mock_adata_obj
            
            run_from_json(params_with_layer, save_results=False)
            
            # Verify layer parameter was passed correctly
            call_kwargs = mock_umap.call_args[1]
            self.assertEqual(call_kwargs['layer'], "arcsinh_z_scores")

    def test_default_parameters(self) -> None:
        """Test that default parameters from JSON template are used."""
        # Minimal params - only required fields
        minimal_params = {
            "Upstream_Analysis": self.in_file,
        }
        
        with patch('spac.templates.umap_transformation_template.'
                  'run_umap') as mock_umap:
            mock_adata_obj = mock_adata()
            mock_adata_obj.obsm["X_umap"] = np.random.rand(10, 2)
            mock_umap.return_value = mock_adata_obj
            
            run_from_json(minimal_params, save_results=False)
            
            # Verify defaults from JSON template were used
            call_kwargs = mock_umap.call_args[1]
            self.assertEqual(call_kwargs['n_neighbors'], 75)
            self.assertEqual(call_kwargs['min_dist'], 0.1)
            self.assertEqual(call_kwargs['n_components'], 2)
            self.assertEqual(call_kwargs['metric'], 'euclidean')
            self.assertEqual(call_kwargs['random_state'], 0)
            self.assertEqual(call_kwargs['transform_seed'], 42)


if __name__ == "__main__":
    unittest.main()