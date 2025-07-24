# tests/templates/test_visualize_ripley_template.py
"""Unit tests for the Visualize Ripley L template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings

import matplotlib
matplotlib.use("Agg")  # Headless backend for CI

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.visualize_ripley_template import run_from_json


def mock_adata_with_ripley(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData with Ripley L results for tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "phenotype": ["B cells", "CD8 T cells"] * (n_cells // 2)
    })
    x_mat = rng.normal(size=(n_cells, 2))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 50.0
    
    # Add mock Ripley L results in the expected format
    # When using pickle, the structure is preserved as-is
    adata.uns["ripley_l_B cells_CD8 T cells"] = {
        "radius": [0, 50, 100],
        "ripley_l": [0, 1.2, 2.5],
        "simulations": np.array([
            [0, 0.8, 1.9], [0, 1.1, 2.3], [0, 1.3, 2.7]
        ])
    }
    return adata


class TestVisualizeRipleyTemplate(unittest.TestCase):
    """Unit tests for the Visualize Ripley L template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "ripley_output.pickle"
        )
        self.out_file = "plots.csv"

        # Save minimal mock data with Ripley results as pickle
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata_with_ripley(), f)

        # Minimal parameters
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Center_Phenotype": "B cells",
            "Neighbor_Phenotype": "CD8 T cells",
            "Plot_Specific_Regions": False,
            "Regions_Labels": [],
            "Plot_Simulations": True,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.visualize_ripley_template.plot_ripley_l')
    def test_run_with_save(self, mock_plot_ripley) -> None:
        """Test visualization with file saving."""
        # Mock the plot_ripley_l function to return a figure and dataframe
        mock_fig = MagicMock()
        mock_df = pd.DataFrame({
            'radius': [0, 50, 100],
            'ripley_l': [0, 1.2, 2.5],
            'lower_ci': [0, 0.8, 1.9],
            'upper_ci': [0, 1.6, 3.1]
        })
        mock_plot_ripley.return_value = (mock_fig, mock_df)
        
        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test with save_results=True (default)
            saved_files = run_from_json(self.params)
            self.assertIn(self.out_file, saved_files)
            
            # Verify plot_ripley_l was called with correct parameters
            mock_plot_ripley.assert_called_once()
            call_args = mock_plot_ripley.call_args
            # Check that adata was passed as first argument
            self.assertEqual(call_args[0][0].n_obs, 10)
            # Check keyword arguments
            self.assertEqual(
                call_args[1]['phenotypes'], ("B cells", "CD8 T cells")
            )
            self.assertEqual(call_args[1]['regions'], None)
            self.assertEqual(call_args[1]['sims'], True)
            self.assertEqual(call_args[1]['return_df'], True)

    @patch('spac.templates.visualize_ripley_template.plot_ripley_l')
    def test_run_without_save(self, mock_plot_ripley) -> None:
        """Test visualization without file saving."""
        # Mock the plot_ripley_l function
        mock_fig = MagicMock()
        mock_df = pd.DataFrame({
            'radius': [0, 50, 100],
            'ripley_l': [0, 1.2, 2.5]
        })
        mock_plot_ripley.return_value = (mock_fig, mock_df)
        
        # Test with save_results=False
        fig, df = run_from_json(self.params, save_results=False)
        
        # Verify we got the figure and dataframe back
        self.assertEqual(fig, mock_fig)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('radius', df.columns)
        self.assertIn('ripley_l', df.columns)

    @patch('spac.templates.visualize_ripley_template.plot_ripley_l')
    def test_with_specific_regions(self, mock_plot_ripley) -> None:
        """Test with specific regions enabled."""
        params_regions = self.params.copy()
        params_regions["Plot_Specific_Regions"] = True
        params_regions["Regions_Labels"] = ["Region1", "Region2"]
        
        mock_fig = MagicMock()
        mock_df = pd.DataFrame({'radius': [0], 'ripley_l': [0]})
        mock_plot_ripley.return_value = (mock_fig, mock_df)
        
        run_from_json(params_regions, save_results=False)
        
        # Verify regions parameter was passed correctly
        call_args = mock_plot_ripley.call_args
        self.assertEqual(
            call_args[1]['regions'], ["Region1", "Region2"]
        )

    def test_regions_validation_error_message(self) -> None:
        """
        Test exact error message for empty regions 
        when Plot_Specific_Regions is True.
        """
        params_bad = self.params.copy()
        params_bad["Plot_Specific_Regions"] = True
        params_bad["Regions_Labels"] = []

        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)

        expected_msg = (
            'Please identify at least one region in the '
            '"Regions Label(s) parameter'
        )
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    @patch('spac.templates.visualize_ripley_template.plot_ripley_l')
    @patch('builtins.print')
    def test_console_output(self, mock_print, mock_plot_ripley) -> None:
        """Test that dataframe is printed to console."""
        # Mock the plot_ripley_l function
        mock_fig = MagicMock()
        mock_df = pd.DataFrame({
            'radius': [0, 50, 100],
            'ripley_l': [0, 1.2, 2.5]
        })
        mock_plot_ripley.return_value = (mock_fig, mock_df)
        
        run_from_json(self.params, save_results=False)
        
        # Verify dataframe was printed
        print_calls = mock_print.call_args_list
        # Check that to_string() output was printed
        df_printed = False
        for call in print_calls:
            if (len(call[0]) > 0 and 
                str(call[0][0]) == mock_df.to_string()):
                df_printed = True
                break
        self.assertTrue(
            df_printed, "DataFrame was not printed to console"
        )

    @patch('spac.templates.visualize_ripley_template.plot_ripley_l')
    def test_json_file_input(self, mock_plot_ripley) -> None:
        """Test with JSON file input."""
        mock_fig = MagicMock()
        mock_df = pd.DataFrame({'radius': [0], 'ripley_l': [0]})
        mock_plot_ripley.return_value = (mock_fig, mock_df)
        
        json_path = os.path.join(self.tmp_dir.name, "viz_params.json")
        with open(json_path, "w") as f:
            json.dump(self.params, f)
        
        result = run_from_json(json_path, save_results=False)
        
        # Should return tuple when save_results=False
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()