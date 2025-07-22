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
    # The key format is important: "ripley_l_phenotype1_phenotype2"
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
            self.tmp_dir.name, "ripley_output.h5ad"
        )
        self.out_file = "plots.csv"

        # Save minimal mock data with Ripley results
        mock_adata_with_ripley().write_h5ad(self.in_file)

        # Minimal parameters
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Center_Phenotype": "B cells",
            "Neighbor_Phenotype": "CD8 T cells",
            "Plot_Specific_Regions": False,
            "Regions_Label_s_": [],
            "Plot_Simulations": True,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.visualize_ripley_template.plot_ripley_l')
    def test_complete_io_workflow(self, mock_plot_ripley) -> None:
        """Single I/O test covering all input/output scenarios."""
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

            # Test 1: Basic workflow with dict input
            saved_files = run_from_json(self.params)
            self.assertIn(self.out_file, saved_files)
            self.assertTrue(os.path.exists(saved_files[self.out_file]))
            
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
            
            # Verify CSV output structure
            df_output = pd.read_csv(saved_files[self.out_file])
            self.assertEqual(len(df_output), 3)  # 3 radius points
            self.assertIn('radius', df_output.columns)
            self.assertIn('ripley_l', df_output.columns)

            # Test 2: With specific regions enabled
            params_regions = self.params.copy()
            params_regions["Plot_Specific_Regions"] = True
            params_regions["Regions_Label_s_"] = ["Region1", "Region2"]
            mock_plot_ripley.reset_mock()
            saved_files_regions = run_from_json(params_regions)
            # Verify regions parameter was passed correctly
            call_args = mock_plot_ripley.call_args
            self.assertEqual(
                call_args[1]['regions'], ["Region1", "Region2"]
            )

            # Test 3: Different output filename
            params_custom = self.params.copy()
            params_custom["Output_File"] = "custom_plots.csv"
            mock_plot_ripley.reset_mock()
            saved_files_custom = run_from_json(params_custom)
            self.assertIn("custom_plots.csv", saved_files_custom)
            self.assertTrue(
                os.path.exists(saved_files_custom["custom_plots.csv"])
            )

    def test_regions_validation_error_message(self) -> None:
        """
        Test exact error message for empty regions 
        when Plot_Specific_Regions is True.
        """
        params_bad = self.params.copy()
        params_bad["Plot_Specific_Regions"] = True
        params_bad["Regions_Label_s_"] = []

        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)

        expected_msg = (
            'Please identify at least one region in the '
            '"Regions Label(s) parameter'
        )
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    @patch('spac.templates.visualize_ripley_template.plot_ripley_l')
    def test_console_output(self, mock_plot_ripley) -> None:
        """Test that dataframe is printed to console."""
        # Mock the plot_ripley_l function
        mock_fig = MagicMock()
        mock_df = pd.DataFrame({
            'radius': [0, 50, 100],
            'ripley_l': [0, 1.2, 2.5]
        })
        mock_plot_ripley.return_value = (mock_fig, mock_df)
        
        # Capture console output
        with patch('builtins.print') as mock_print:
            run_from_json(self.params)
            
            # Verify dataframe was printed
            print_calls = mock_print.call_args_list
            # Check that to_string() output was printed
            df_printed = False
            for call in print_calls:
                if (len(call[0]) > 0 and 
                    call[0][0] == mock_df.to_string()):
                    df_printed = True
                    break
            self.assertTrue(
                df_printed, "DataFrame was not printed to console"
            )


if __name__ == "__main__":
    unittest.main()