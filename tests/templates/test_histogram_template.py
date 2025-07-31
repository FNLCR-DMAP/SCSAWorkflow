# tests/templates/test_histogram_template.py
"""Unit tests for the Histogram template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock

import matplotlib
matplotlib.use("Agg")  # Headless backend for CI

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.histogram_template import run_from_json


def mock_adata_with_features(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    
    # Simple expression data
    x_mat = rng.normal(size=(n_cells, 3))
    
    # Simple observations
    obs = pd.DataFrame({
        "cell_type": ["TypeA", "TypeB"] * (n_cells // 2)
    })
    
    # Simple var names
    var = pd.DataFrame(index=["Gene1", "Gene2", "Gene3"])
    
    return ad.AnnData(X=x_mat, obs=obs, var=var)


class TestHistogramTemplate(unittest.TestCase):
    """Unit tests for the Histogram template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "plots.csv"

        # Save minimal mock data as pickle
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata_with_features(), f)

        # Minimal parameters
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Plot_By": "Annotation",
            "Annotation": "cell_type",
            "Feature": "None",
            "Table_": "Original",
            "Group_by": "None",
            "Together": True,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.histogram_template.histogram')
    @patch('seaborn.move_legend')  # Mock seaborn to avoid legend issues
    def test_run_with_save(self, mock_move_legend, mock_histogram) -> None:
        """Test basic run with file saving."""
        # Mock the histogram function
        mock_fig = MagicMock()
        mock_fig.number = 1
        mock_fig.set_size_inches = MagicMock()
        mock_fig.set_dpi = MagicMock()
        
        mock_ax = MagicMock()
        mock_ax.get_legend.return_value = None
        mock_ax.tick_params = MagicMock()
        mock_ax.set_title = MagicMock()
        
        mock_df = pd.DataFrame({
            'category': ['TypeA', 'TypeB'],
            'count': [5, 5]
        })
        
        mock_histogram.return_value = {
            "fig": mock_fig,
            "axs": mock_ax,
            "df": mock_df
        }
        
        # Run with save_results=True (default)
        saved_files = run_from_json(self.params)
        
        # Check that file was saved
        self.assertIn(self.out_file, saved_files)
        self.assertTrue(os.path.exists(saved_files[self.out_file]))
        
        # Verify histogram was called
        mock_histogram.assert_called_once()

    @patch('spac.templates.histogram_template.histogram')
    @patch('seaborn.move_legend')
    def test_run_without_save(self, mock_move_legend, mock_histogram) -> None:
        """Test run without file saving."""
        # Mock the histogram function
        mock_fig = MagicMock()
        mock_fig.number = 1
        mock_fig.set_size_inches = MagicMock()
        mock_fig.set_dpi = MagicMock()
        
        mock_ax = MagicMock()
        mock_ax.get_legend.return_value = None
        mock_ax.tick_params = MagicMock()
        mock_ax.set_title = MagicMock()
        
        mock_df = pd.DataFrame({'category': ['A'], 'count': [10]})
        
        mock_histogram.return_value = {
            "fig": mock_fig,
            "axs": mock_ax,
            "df": mock_df
        }
        
        # Run with save_results=False
        fig, df = run_from_json(self.params, save_results=False)
        
        # Check that we got figure and dataframe
        self.assertEqual(fig, mock_fig)
        self.assertIsInstance(df, pd.DataFrame)

    def test_json_file_input(self) -> None:
        """Test that JSON file input works."""
        json_path = os.path.join(self.tmp_dir.name, "params.json")
        with open(json_path, "w") as f:
            json.dump(self.params, f)
        
        with patch('spac.templates.histogram_template.histogram') as mock_hist:
            with patch('seaborn.move_legend'):
                mock_hist.return_value = {
                    "fig": MagicMock(number=1),
                    "axs": MagicMock(),
                    "df": pd.DataFrame()
                }
                
                result = run_from_json(json_path, save_results=False)
                self.assertIsInstance(result, tuple)

    def test_error_messages(self) -> None:
        """Test exact error messages for key validations."""
        # Test 1: No annotations available
        adata_no_obs = ad.AnnData(X=np.random.rand(5, 3))
        with open(self.in_file, 'wb') as f:
            pickle.dump(adata_no_obs, f)
        
        params_bad = self.params.copy()
        params_bad["Annotation"] = "None"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        expected_msg = 'No annotations available in adata.obs to plot.'
        self.assertEqual(str(context.exception), expected_msg)
        
        # Test 2: Invalid rotation angle
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata_with_features(), f)
            
        params_bad_rotate = self.params.copy()
        params_bad_rotate["X_Axis_Label_Rotation"] = 400
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad_rotate)
        
        expected_msg = (
            'The X label rotation should fall within 0 to 360 degree. '
            'Received "400".'
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.histogram_template.histogram')
    @patch('seaborn.move_legend')
    @patch('builtins.print')
    def test_console_output(
        self, mock_print, mock_move_legend, mock_histogram
    ) -> None:
        """Test that dataframe is printed to console."""
        # Mock the histogram function
        mock_df = pd.DataFrame({
            'category': ['TypeA', 'TypeB'],
            'count': [5, 5]
        })
        
        mock_histogram.return_value = {
            "fig": MagicMock(number=1),
            "axs": MagicMock(),
            "df": mock_df
        }
        
        run_from_json(self.params, save_results=False)
        
        # Check that dataframe info was printed
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list
                      if call[0]]
        
        # Should print "Displaying top 10 rows"
        self.assertTrue(
            any("Displaying top 10 rows" in msg for msg in print_calls)
        )


if __name__ == "__main__":
    unittest.main()