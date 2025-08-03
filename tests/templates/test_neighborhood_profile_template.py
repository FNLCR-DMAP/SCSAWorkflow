# tests/templates/test_neighborhood_profile_template.py
"""Unit tests for the Neighborhood Profile template."""

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

from spac.templates.neighborhood_profile_template import run_from_json


def mock_adata(n_cells: int = 20) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    
    # Create simple cell types that repeat properly for any n_cells
    cell_types = ["T cells", "B cells", "Tumor cells"]
    obs = pd.DataFrame({
        "cell_type": [cell_types[i % len(cell_types)] 
                      for i in range(n_cells)],
        "slide": (["Slide1", "Slide2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Marker1", "Marker2", "Marker3"]
    
    # Add spatial coordinates
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    
    return adata


class TestNeighborhoodProfileTemplate(unittest.TestCase):
    """Unit tests for the Neighborhood Profile template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation_of_interest": "cell_type",
            "Bins": [0, 10, 30, 50],
            "Stratify_By": "slide",
            "Anchor_Neighbor_List": [
                "T cells;B cells", 
                "Tumor cells;T cells"
            ]
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Mock the neighborhood_profile function to add expected data
            with patch('spac.templates.neighborhood_profile_template.'
                      'neighborhood_profile') as mock_np:
                # Setup mock to add the expected data structures
                def mock_neighborhood_profile(adata, **kwargs):
                    # Add mock neighborhood profile data
                    n_cells = adata.n_obs
                    n_phenotypes = 3  # T cells, B cells, Tumor cells
                    n_bins = len(kwargs['distances']) - 1  # 3 bins
                    
                    # Create mock profile data
                    adata.obsm["neighborhood_profile"] = np.random.rand(
                        n_cells, n_phenotypes, n_bins
                    )
                    
                    # Add labels to uns
                    adata.uns["neighborhood_profile"] = {
                        "labels": np.array([
                            "T cells", "B cells", "Tumor cells"
                        ])
                    }
                    
                mock_np.side_effect = mock_neighborhood_profile
                
                # Test 1: Run with default parameters (save files)
                result = run_from_json(self.params)
                self.assertIsInstance(result, dict)
                
                # Should have 2 CSV files based on anchor_neighbor_list
                self.assertEqual(len(result), 2)
                
                # Check filenames
                expected_files = [
                    "anchor_T cells_neighbor_B cells.csv",
                    "anchor_Tumor cells_neighbor_T cells.csv"
                ]
                for expected in expected_files:
                    self.assertIn(expected, result)
                
                # Test 2: Run without saving
                result_no_save = run_from_json(
                    self.params, save_results=False
                )
                # Should return dict of dataframes
                self.assertIsInstance(result_no_save, dict)
                self.assertEqual(len(result_no_save), 2)
                
                # Check that keys are tuples
                for key in result_no_save:
                    self.assertIsInstance(key, tuple)
                    self.assertEqual(len(key), 2)
                
                # Test 3: JSON file input
                json_path = os.path.join(self.tmp_dir.name, "params.json")
                with open(json_path, "w") as f:
                    json.dump(self.params, f)
                
                result_json = run_from_json(json_path)
                self.assertIsInstance(result_json, dict)
                self.assertEqual(len(result_json), 2)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Mock neighborhood_profile to add required data structures
        with patch('spac.templates.neighborhood_profile_template.'
                  'neighborhood_profile') as mock_np:
            def mock_neighborhood_profile(adata, **kwargs):
                adata.obsm["neighborhood_profile"] = np.random.rand(
                    adata.n_obs, 2, 3  # Only 2 phenotypes
                )
                adata.uns["neighborhood_profile"] = {
                    "labels": np.array([
                        "T cells", "B cells"
                    ])  # Missing "Unknown"
                }
            
            mock_np.side_effect = mock_neighborhood_profile
            
            # Test with invalid neighbor label
            params_bad = self.params.copy()
            params_bad["Anchor_Neighbor_List"] = ["T cells;Unknown"]
            
            with self.assertRaises(ValueError) as context:
                run_from_json(params_bad)
            
            # Check exact error message
            expected_msg = ("Neighbor label 'Unknown' not found in "
                           "neighborhood_profile labels.")
            self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.neighborhood_profile_template.'
           'neighborhood_profile')
    def test_function_calls(self, mock_np) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the neighborhood_profile function
        def mock_neighborhood_profile(adata, **kwargs):
            # Add required data structures
            adata.obsm["neighborhood_profile"] = np.zeros(
                (adata.n_obs, 3, 3)
            )
            adata.uns["neighborhood_profile"] = {
                "labels": np.array(["T cells", "B cells", "Tumor cells"])
            }
        
        mock_np.side_effect = mock_neighborhood_profile
        
        # Test with None slide_names
        params_alt = self.params.copy()
        params_alt["Stratify_By"] = "None"
        
        run_from_json(params_alt, save_results=False)
        
        # Verify function was called correctly
        mock_np.assert_called_once()
        call_args = mock_np.call_args
        
        # Check specific parameters
        self.assertEqual(call_args[1]['phenotypes'], "cell_type")
        self.assertEqual(call_args[1]['distances'], [0.0, 10.0, 30.0, 50.0])
        self.assertIsNone(call_args[1]['regions'])  # "None" -> None
        self.assertEqual(call_args[1]['spatial_key'], "spatial")
        self.assertIsNone(call_args[1]['normalize'])
        self.assertEqual(
            call_args[1]['associated_table_name'], "neighborhood_profile"
        )


if __name__ == "__main__":
    unittest.main()