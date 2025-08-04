# tests/templates/test_nearest_neighbor_calculation_template.py
"""Unit tests for the Nearest Neighbor Calculation template."""

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

from spac.templates.nearest_neighbor_calculation_template import (
    run_from_json
)


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    # Add spatial coordinates
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    return adata


class TestNearestNeighborCalculationTemplate(unittest.TestCase):
    """Unit tests for the Nearest Neighbor Calculation template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "cell_type",
            "ImageID": "None",
            "Nearest_Neighbor_Associated_Table": "spatial_distance",
            "Verbose": True,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Mock the calculate_nearest_neighbor function
            with patch(
                'spac.templates.nearest_neighbor_calculation_template.'
                'calculate_nearest_neighbor'
            ) as mock_calc_nn:
                # Mock function adds a distance matrix to adata.obsm
                def side_effect(adata, annotation, spatial_associated_table,
                               imageid, label, verbose):
                    # Simulate adding distance matrix to obsm
                    n_cells = adata.n_obs
                    unique_types = adata.obs[annotation].unique()
                    n_types = len(unique_types)
                    # Create mock distance dataframe
                    dist_df = pd.DataFrame(
                        np.random.rand(n_cells, n_types),
                        columns=unique_types,
                        index=adata.obs.index
                    )
                    adata.obsm[label] = dist_df
                
                mock_calc_nn.side_effect = side_effect
                
                # Test 1: Run with default parameters
                result = run_from_json(self.params)
                self.assertIsInstance(result, dict)
                # Verify file was saved
                self.assertTrue(len(result) > 0)
                
                # Test 2: Run without saving
                result_no_save = run_from_json(
                    self.params, save_results=False
                )
                # Check appropriate return type
                self.assertIsInstance(result_no_save, ad.AnnData)
                # Verify nearest neighbor distances were added
                self.assertIn("spatial_distance", result_no_save.obsm)
                
                # Test 3: JSON file input
                json_path = os.path.join(self.tmp_dir.name, "params.json")
                with open(json_path, "w") as f:
                    json.dump(self.params, f)
                
                result_json = run_from_json(json_path)
                self.assertIsInstance(result_json, dict)

    def test_function_calls(self) -> None:
        """Test that main function is called with correct parameters."""
        with patch(
            'spac.templates.nearest_neighbor_calculation_template.'
            'calculate_nearest_neighbor'
        ) as mock_calc_nn:
            # Test with different parameters
            params_alt = self.params.copy()
            params_alt["ImageID"] = "sample"
            params_alt["Nearest_Neighbor_Associated_Table"] = "nn_distances"
            params_alt["Verbose"] = False
            
            run_from_json(params_alt, save_results=False)
            
            # Verify function was called correctly
            mock_calc_nn.assert_called_once()
            call_args = mock_calc_nn.call_args
            
            # Check parameter conversions
            self.assertEqual(call_args[1]['annotation'], "cell_type")
            self.assertEqual(
                call_args[1]['spatial_associated_table'], "spatial"
            )
            self.assertEqual(call_args[1]['imageid'], "sample")
            self.assertEqual(call_args[1]['label'], "nn_distances")
            self.assertEqual(call_args[1]['verbose'], False)

    def test_imageid_none_conversion(self) -> None:
        """Test that string 'None' is converted to actual None."""
        with patch(
            'spac.templates.nearest_neighbor_calculation_template.'
            'calculate_nearest_neighbor'
        ) as mock_calc_nn:
            # Test with "None" string
            self.params["ImageID"] = "None"
            run_from_json(self.params, save_results=False)
            
            # Verify imageid was converted to None
            call_args = mock_calc_nn.call_args
            self.assertIsNone(call_args[1]['imageid'])


if __name__ == "__main__":
    unittest.main()