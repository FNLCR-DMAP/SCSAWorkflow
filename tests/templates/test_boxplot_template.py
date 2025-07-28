# tests/templates/test_boxplot_template.py
"""Unit tests for the Boxplot template."""

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

from spac.templates.boxplot_template import run_from_json


def mock_adata_for_boxplot(n_cells: int = 20) -> ad.AnnData:
    """Return a minimal synthetic AnnData for boxplot tests."""
    rng = np.random.default_rng(0)
    
    # Create observations with annotations
    obs = pd.DataFrame({
        "cell_type": ["B cells", "T cells"] * (n_cells // 2),
        "condition": ["Control", "Treatment"] * (n_cells // 2)
    })
    
    # Create expression matrix with 3 features
    n_features = 3
    x_mat = rng.poisson(lam=5, size=(n_cells, n_features)) + 1.0
    
    # Create var dataframe with feature names
    var = pd.DataFrame(
        index=[f"Gene_{i}" for i in range(n_features)]
    )
    
    adata = ad.AnnData(X=x_mat, obs=obs, var=var)
    
    # Add a normalized layer
    adata.layers["normalized"] = np.log1p(adata.X)
    
    return adata


class TestBoxplotTemplate(unittest.TestCase):
    """Unit tests for the Boxplot template."""

    def _create_mock_boxplot_return(self, df_data=None):
        """Helper to create standard mock returns."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        if df_data is None:
            df_data = {'Gene_0': [1]}
        mock_df = pd.DataFrame(df_data)
        return mock_fig, mock_ax, mock_df

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input_data.pickle"
        )
        self.out_file = "boxplot_summary.csv"

        # Save minimal mock data as pickle
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata_for_boxplot(), f)

        # Minimal parameters
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Primary_Annotation": "cell_type",
            "Secondary_Annotation": "None",
            "Table_to_Visualize": "Original",
            "Feature_s_to_Plot": ["All"],
            "Value_Axis_Log_Scale": False,
            "Figure_Title": "BoxPlot",
            "Horizontal_Plot": False,
            "Figure_Width": 12,
            "Figure_Height": 8,
            "Figure_DPI": 300,
            "Font_Size": 10,
            "Keep_Outliers": True,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_run_with_save(self, mock_show, mock_boxplot) -> None:
        """Test boxplot with file saving."""
        # Mock the boxplot function to return figure, ax, and dataframe
        mock_fig, mock_ax, mock_df = self._create_mock_boxplot_return({
            'Gene_0': np.random.RandomState(42).rand(20),
            'Gene_1': np.random.RandomState(42).rand(20),
            'Gene_2': np.random.RandomState(42).rand(20)
        })
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test with save_results=True (default)
            saved_files = run_from_json(self.params)
            self.assertIn(self.out_file, saved_files)
            
            # Verify boxplot was called with correct parameters
            mock_boxplot.assert_called_once()
            call_args = mock_boxplot.call_args
            
            # Check keyword arguments
            self.assertEqual(call_args[1]['annotation'], "cell_type")
            self.assertEqual(call_args[1]['second_annotation'], None)
            self.assertEqual(call_args[1]['layer'], None)  # Original -> None
            self.assertEqual(call_args[1]['log_scale'], False)
            self.assertEqual(call_args[1]['orient'], "v")
            self.assertEqual(call_args[1]['showfliers'], True)

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_run_without_save(self, mock_show, mock_boxplot) -> None:
        """Test boxplot without file saving."""
        # Mock the boxplot function
        mock_fig, mock_ax, mock_df = self._create_mock_boxplot_return({
            'Gene_0': [1, 2, 3],
            'Gene_1': [4, 5, 6],
            'Gene_2': [7, 8, 9]
        })
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        # Test with save_results=False
        fig, summary_df = run_from_json(
            self.params, save_results=False
        )
        
        # Verify we got the figure and summary dataframe back
        self.assertEqual(fig, mock_fig)
        self.assertIsInstance(summary_df, pd.DataFrame)
        # Check that summary has statistics
        self.assertIn('count', summary_df['index'].values)
        self.assertIn('mean', summary_df['index'].values)
        self.assertIn('std', summary_df['index'].values)

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_all_features_plotting(self, mock_show, mock_boxplot) -> None:
        """Test plotting all features."""
        mock_fig, mock_ax, mock_df = self._create_mock_boxplot_return({
            'Gene_0': [1], 'Gene_1': [2]
        })
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(self.params, save_results=False)
        
        # Verify features parameter - should be all gene names
        call_args = mock_boxplot.call_args
        features = call_args[1]['features']
        self.assertEqual(len(features), 3)  # mock data has 3 genes
        self.assertIn('Gene_0', features)
        self.assertIn('Gene_1', features)
        self.assertIn('Gene_2', features)

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_specific_features(self, mock_show, mock_boxplot) -> None:
        """Test plotting specific features."""
        params_specific = self.params.copy()
        params_specific["Feature_s_to_Plot"] = ["Gene_0", "Gene_2"]
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1], 'Gene_2': [2]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_specific, save_results=False)
        
        # Verify specific features were passed
        call_args = mock_boxplot.call_args
        self.assertEqual(
            call_args[1]['features'], ["Gene_0", "Gene_2"]
        )

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_layer_selection(self, mock_show, mock_boxplot) -> None:
        """Test different layer selections."""
        # Test normalized layer
        params_norm = self.params.copy()
        params_norm["Table_to_Visualize"] = "normalized"
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_norm, save_results=False)
        
        call_args = mock_boxplot.call_args
        self.assertEqual(call_args[1]['layer'], "normalized")

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_horizontal_orientation(self, mock_show, mock_boxplot) -> None:
        """Test horizontal plot orientation."""
        params_horiz = self.params.copy()
        params_horiz["Horizontal_Plot"] = True
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_horiz, save_results=False)
        
        call_args = mock_boxplot.call_args
        self.assertEqual(call_args[1]['orient'], "h")

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_secondary_annotation(self, mock_show, mock_boxplot) -> None:
        """Test with secondary annotation."""
        params_second = self.params.copy()
        params_second["Secondary_Annotation"] = "condition"
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_second, save_results=False)
        
        call_args = mock_boxplot.call_args
        self.assertEqual(call_args[1]['second_annotation'], "condition")

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_log_scale(self, mock_show, mock_boxplot) -> None:
        """Test log scale option."""
        params_log = self.params.copy()
        params_log["Value_Axis_Log_Scale"] = True
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_log, save_results=False)
        
        call_args = mock_boxplot.call_args
        self.assertEqual(call_args[1]['log_scale'], True)

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_outliers_option(self, mock_show, mock_boxplot) -> None:
        """Test outliers (showfliers) option."""
        params_no_outliers = self.params.copy()
        params_no_outliers["Keep_Outliers"] = False
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_no_outliers, save_results=False)
        
        call_args = mock_boxplot.call_args
        self.assertEqual(call_args[1]['showfliers'], False)

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_figure_parameters(self, mock_subplots, mock_show, mock_boxplot) -> None:
        """Test figure configuration parameters."""
        params_fig = self.params.copy()
        params_fig.update({
            "Figure_Width": 15,
            "Figure_Height": 10,
            "Figure_DPI": 150,
            "Font_Size": 14
        })
        
        # Mock the figure and axes from plt.subplots
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock the boxplot return
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_fig, save_results=False)
        
        # Verify figure methods were called with correct values
        mock_fig.set_size_inches.assert_called_with(15, 10)
        mock_fig.set_dpi.assert_called_with(150)

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    @patch('spac.templates.boxplot_template.logging.info')
    @patch('spac.templates.boxplot_template.logging.debug')
    def test_console_output(self, mock_debug, mock_info, mock_show, mock_boxplot) -> None:
        """Test that summary statistics are logged to console."""
        mock_fig, mock_ax, mock_df = self._create_mock_boxplot_return({
            'Gene_0': [1, 2, 3],
            'Gene_1': [4, 5, 6]
        })
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(self.params, save_results=False)
        
        # Check that appropriate messages were logged
        info_calls = [str(call[0][0]) for call in mock_info.call_args_list
                     if call[0]]
        
        # Should log "Plotting All Features"
        self.assertTrue(
            any("Plotting All Features" in msg for msg in info_calls)
        )
        
        # Should log "Summary statistics of the dataset:"
        self.assertTrue(
            any("Summary statistics" in msg for msg in info_calls)
        )

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_json_file_input(self, mock_show, mock_boxplot) -> None:
        """Test with JSON file input."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        json_path = os.path.join(self.tmp_dir.name, "boxplot_params.json")
        with open(json_path, "w") as f:
            json.dump(self.params, f)
        
        result = run_from_json(json_path, save_results=False)
        
        # Should return tuple when save_results=False
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_no_annotation(self, mock_show, mock_boxplot) -> None:
        """Test with no annotation (None values)."""
        params_no_annot = self.params.copy()
        params_no_annot["Primary_Annotation"] = "None"
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        run_from_json(params_no_annot, save_results=False)
        
        call_args = mock_boxplot.call_args
        self.assertIsNone(call_args[1]['annotation'])

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_legend_handling(self, mock_show, mock_boxplot) -> None:
        """Test legend positioning handling."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_df = pd.DataFrame({'Gene_0': [1]})
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        # Test both with and without legend error
        with patch('seaborn.move_legend') as mock_move_legend:
            # Case 1: Legend exists
            run_from_json(self.params, save_results=False)
            mock_move_legend.assert_called_once()
            
            # Case 2: Legend doesn't exist (raises exception)
            mock_move_legend.side_effect = Exception("No legend")
            run_from_json(self.params, save_results=False)
            # Should not raise error, just print message

    def test_parameter_validation(self) -> None:
        """Test that missing required parameters raise errors."""
        params_missing = self.params.copy()
        del params_missing["Upstream_Analysis"]

        with self.assertRaises(KeyError) as context:
            run_from_json(params_missing)

        self.assertIn("Upstream_Analysis", str(context.exception))

    @patch('spac.templates.boxplot_template.boxplot')
    @patch('matplotlib.pyplot.show')
    def test_save_figure_option(self, mock_show, mock_boxplot) -> None:
        """Test saving figure to file."""
        params_fig_save = self.params.copy()
        params_fig_save["Figure_File"] = "boxplot_figure.png"
        
        mock_fig, mock_ax, mock_df = self._create_mock_boxplot_return()
        mock_boxplot.return_value = (mock_fig, mock_ax, mock_df)
        
        saved_files = run_from_json(params_fig_save)
        
        # Should save both summary and figure
        self.assertIn(self.out_file, saved_files)
        self.assertIn("boxplot_figure.png", saved_files)
        self.assertEqual(len(saved_files), 2)


if __name__ == "__main__":
    unittest.main()