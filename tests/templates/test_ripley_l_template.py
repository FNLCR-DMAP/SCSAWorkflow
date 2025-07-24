# tests/templates/test_ripley_l_template.py
"""Unit‑tests for the Ripley‑L template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.ripley_l_template import run_from_json


def mock_adata(n_cells: int = 40) -> ad.AnnData:
    """Return a tiny synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "renamed_phenotypes": np.where(
                rng.random(n_cells) > 0.5, "B cells", "CD8 T cells"
            )
        }
    )
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 300.0
    return adata


class TestRipleyLTemplate(unittest.TestCase):
    """Light‑weight sanity checks for the Ripley‑L template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")
        self.out_file = os.path.join(self.tmp_dir.name, "output.pickle")

        # Save as pickle
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        self.params = {
            "Upstream_Analysis": self.in_file,
            "Radii": [0, 50, 100],
            "Annotation": "renamed_phenotypes",
            "Center_Phenotype": "B cells",
            "Neighbor_Phenotype": "CD8 T cells",
            "Output_path": self.out_file,
            "Stratify_By": "None",
            "Number_of_Simulations": 100,
            "Area": "None",
            "Seed": 42,
            "Spatial_Key": "spatial",
            "Edge_Correction": True
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.ripley_l_template.ripley_l')
    def test_run_from_dict_with_save(self, mock_ripley_l) -> None:
        """Test run_from_json with dict parameters and file saving."""
        # Mock the ripley_l function 
        def mock_ripley_side_effect(adata, **kwargs):
            # Simulate what ripley_l does - adds results to adata.uns
            phenotypes = kwargs.get('phenotypes', ['B cells', 'CD8 T cells'])
            key = f"ripley_l_{phenotypes[0]}_{phenotypes[1]}"
            adata.uns[key] = {
                "radius": kwargs.get('distances', [0, 50, 100]),
                "ripley_l": [0.0, 1.2, 2.5],
                "simulations": [[0.0, 0.8, 1.9], [0.0, 1.1, 2.3]],
            }
            return None
        
        mock_ripley_l.side_effect = mock_ripley_side_effect

        # Run with save_results=True (default)
        saved_files = run_from_json(self.params)
        
        # Check that ripley_l was called with correct parameters
        mock_ripley_l.assert_called_once()
        call_args = mock_ripley_l.call_args
        
        # Verify the call arguments
        self.assertEqual(call_args[1]['annotation'], "renamed_phenotypes")
        self.assertEqual(call_args[1]['phenotypes'], ["B cells", "CD8 T cells"])
        self.assertEqual(call_args[1]['distances'], [0.0, 50.0, 100.0])
        self.assertEqual(call_args[1]['n_simulations'], 100)
        self.assertEqual(call_args[1]['seed'], 42)
        self.assertEqual(call_args[1]['spatial_key'], "spatial")
        self.assertEqual(call_args[1]['edge_correction'], True)
        
        # Check that output file was created - check if any file was saved
        self.assertTrue(
            len(saved_files) > 0,
            f"Expected files to be saved, but got {saved_files}"
        )
        
        # Check that at least one pickle file was created
        pickle_files = [f for f in saved_files.values() 
                       if f.endswith('.pickle')]
        self.assertTrue(
            len(pickle_files) > 0,
            f"Expected at least one pickle file, got {saved_files}"
        )

    @patch('spac.templates.ripley_l_template.ripley_l')
    def test_run_from_dict_without_save(self, mock_ripley_l) -> None:
        """Test run_from_json with dict parameters and no file saving."""
        # Mock the ripley_l function
        def mock_ripley_side_effect(adata, **kwargs):
            phenotypes = kwargs.get('phenotypes', ['B cells', 'CD8 T cells'])
            key = f"ripley_l_{phenotypes[0]}_{phenotypes[1]}"
            adata.uns[key] = {
                "radius": kwargs.get('distances', [0, 50, 100]),
                "ripley_l": [0.0, 1.2, 2.5],
            }
            return None
        
        mock_ripley_l.side_effect = mock_ripley_side_effect
        
        # Run with save_results=False
        adata_result = run_from_json(self.params, save_results=False)
        
        # Check that ripley_l was called
        mock_ripley_l.assert_called_once()
        
        # Check that we got an AnnData object back
        self.assertIsInstance(adata_result, ad.AnnData)
        
        # Check that results are in the object
        ripley_key = "ripley_l_B cells_CD8 T cells"
        self.assertIn(ripley_key, adata_result.uns)

    @patch('spac.templates.ripley_l_template.ripley_l')
    def test_run_from_json_file(self, mock_ripley_l) -> None:
        """Test run_from_json accepts a JSON file path."""
        # Mock the ripley_l function
        def mock_ripley_side_effect(adata, **kwargs):
            phenotypes = kwargs.get('phenotypes', ['B cells', 'CD8 T cells'])
            key = f"ripley_l_{phenotypes[0]}_{phenotypes[1]}"
            adata.uns[key] = {
                "radius": kwargs.get('distances', [0, 50, 100]),
                "ripley_l": [0.0, 1.2, 2.5],
                "simulations": [[0.0, 0.8, 1.9], [0.0, 1.1, 2.3]],
            }
            return None
        
        mock_ripley_l.side_effect = mock_ripley_side_effect
        
        json_path = os.path.join(self.tmp_dir.name, "params.json")
        with open(json_path, "w") as handle:
            json.dump(self.params, handle)
        
        saved_files = run_from_json(json_path)
        
        # Verify ripley_l was called
        mock_ripley_l.assert_called_once()
        
        # Check that save_outputs was called
        mock_ripley_l.assert_called_once()

        # Check that files were saved
        self.assertTrue(len(saved_files) > 0)

    @patch('spac.templates.ripley_l_template.ripley_l')
    def test_radii_conversion(self, mock_ripley_l) -> None:
        """Test that radii strings are converted to floats."""
        # Use string radii
        params_str = self.params.copy()
        params_str["Radii"] = ["0", "50", "100"]
        
        def mock_ripley_side_effect(adata, **kwargs):
            phenotypes = kwargs.get('phenotypes', ['B cells', 'CD8 T cells'])
            key = f"ripley_l_{phenotypes[0]}_{phenotypes[1]}"
            adata.uns[key] = {"radius": [0, 50, 100], "ripley_l": [0, 1, 2]}
            return None
            
        mock_ripley_l.side_effect = mock_ripley_side_effect

        run_from_json(params_str, save_results=False)
        
        # Check that radii were converted to floats
        call_args = mock_ripley_l.call_args
        self.assertEqual(call_args[1]['distances'], [0.0, 50.0, 100.0])

    def test_invalid_radius_conversion(self) -> None:
        """Test that invalid radius values raise appropriate errors."""
        params_bad = self.params.copy()
        params_bad["Radii"] = ["0", "50", "invalid"]
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        expected_msg = "Failed to convert the radius: 'invalid' to float"
        self.assertIn(expected_msg, str(context.exception))

    def test_parameter_validation(self) -> None:
        """Test that missing required parameters raise errors."""
        params_missing = self.params.copy()
        del params_missing["Center_Phenotype"]
        
        with self.assertRaises(KeyError) as context:
            run_from_json(params_missing)
        
        self.assertIn("Center_Phenotype", str(context.exception))

    @patch('spac.templates.ripley_l_template.ripley_l')
    def test_regions_parameter(self, mock_ripley_l) -> None:
        """Test that regions parameter is processed correctly."""
        params_regions = self.params.copy()
        params_regions["Stratify_By"] = "tumor_region"
        
        mock_ripley_l.side_effect = lambda adata, **kwargs: None
        
        run_from_json(params_regions, save_results=False)
        
        # Check that regions was passed correctly (not as "None")
        call_args = mock_ripley_l.call_args
        self.assertEqual(call_args[1]['regions'], "tumor_region")

    def test_pickle_output_format(self) -> None:
        """Test that output defaults to pickle format."""
        params = self.params.copy()
        params["Output_File"] = "results.dat"  # No extension
        
        with patch('spac.templates.ripley_l_template.ripley_l'):
            saved_files = run_from_json(params)
            
            # Should save as pickle by default
            pickle_files = [f for f in saved_files.values() 
                           if '.pickle' in str(f)]
            self.assertTrue(len(pickle_files) > 0)


if __name__ == "__main__":
    unittest.main()
