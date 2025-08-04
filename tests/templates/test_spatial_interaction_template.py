# tests/templates/test_spatial_interaction_template.py
"""Unit tests for the Spatial Interaction template."""

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

from spac.templates.spatial_interaction_template import run_from_json


def mock_adata(n_cells: int = 100) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (
            ["TypeA", "TypeB", "TypeC"] * ((n_cells + 2) // 3)
        )[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells],
        "batch": (["Batch1", "Batch2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 5))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Marker{i}" for i in range(5)]
    # Add spatial coordinates - required for spatial analysis
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    return adata


class TestSpatialInteractionTemplate(unittest.TestCase):
    """Unit tests for the Spatial Interaction template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "spatial_interaction"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "cell_type",
            "Spatial_Analysis_Method": "Neighborhood Enrichment",
            "Stratify_By": ["None"],
            "Coordinate_Type": "None",
            "Seed": "None",
            "Radius": "None",
            "K_Nearest_Neighbors": 6,
            "Figure_Width": 15,
            "Figure_Height": 12,
            "Figure_DPI": 200,
            "Font_Size": 12,
            "Color_Bar_Range": "Automatic",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.spatial_interaction_template.'
           'spatial_interaction')
    def test_complete_io_workflow(self, mock_spatial) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the spatial_interaction function
        mock_fig = MagicMock()
        mock_fig.number = 1
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig
        mock_ax.title = MagicMock()
        mock_ax.xaxis = MagicMock()
        mock_ax.yaxis = MagicMock()
        mock_ax.xaxis.label = MagicMock()
        mock_ax.yaxis.label = MagicMock()
        mock_ax.tick_params = MagicMock()
        
        # Mock matrix data matching expected output
        mock_matrix = {
            'neighborhood_enrichment.csv': pd.DataFrame({
                'TypeA': [1.0, 0.5, 0.3],
                'TypeB': [0.5, 1.0, 0.7],
                'TypeC': [0.3, 0.7, 1.0]
            }, index=['TypeA', 'TypeB', 'TypeC'])
        }
        
        mock_spatial.return_value = {
            'Ax': mock_ax,
            'Matrix': {'annotation': mock_matrix}
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params, show_plot=False)
            self.assertIsInstance(result, dict)
            # Verify both PNG and CSV files were saved
            self.assertTrue(len(result) >= 2)  # At least 1 PNG + 1 CSV
            png_files = [f for f in result.keys() if f.endswith('.png')]
            csv_files = [f for f in result.keys() if f.endswith('.csv')]
            self.assertEqual(len(png_files), 1)
            self.assertEqual(len(csv_files), 1)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(
                self.params, save_results=False, show_plot=False
            )
            # For multi-plot template, returns None when not saving
            self.assertIsNone(result_no_save)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path, show_plot=False)
            self.assertIsInstance(result_json, dict)
            self.assertTrue(len(result_json) >= 2)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid stratify_by with None not at first position
        params_bad = self.params.copy()
        params_bad["Stratify_By"] = ["sample", "None", "batch"]
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad, show_plot=False)
        
        # Check exact error message
        expected_msg = (
            'Found string "None" in the stratify by list that is '
            'not the first entry.\n'
            'Please remove the "None" to proceed with the list of '
            'stratify by options, \n'
            'or move the "None" to start of the list to disable '
            'stratification. Thank you.'
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.spatial_interaction_template.'
           'spatial_interaction')
    def test_function_calls(self, mock_spatial) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the spatial_interaction function with stratified output
        mock_fig1 = MagicMock()
        mock_fig1.number = 1
        mock_fig2 = MagicMock()
        mock_fig2.number = 2
        
        mock_ax1 = MagicMock()
        mock_ax1.get_figure.return_value = mock_fig1
        mock_ax1.title = MagicMock()
        mock_ax1.xaxis = MagicMock()
        mock_ax1.yaxis = MagicMock()
        mock_ax1.xaxis.label = MagicMock()
        mock_ax1.yaxis.label = MagicMock()
        mock_ax1.tick_params = MagicMock()
        
        mock_ax2 = MagicMock()
        mock_ax2.get_figure.return_value = mock_fig2
        mock_ax2.title = MagicMock()
        mock_ax2.xaxis = MagicMock()
        mock_ax2.yaxis = MagicMock()
        mock_ax2.xaxis.label = MagicMock()
        mock_ax2.yaxis.label = MagicMock()
        mock_ax2.tick_params = MagicMock()
        
        # Test with stratification - should produce multiple outputs
        mock_spatial.return_value = {
            'Ax': {'sample_S1': mock_ax1, 'sample_S2': mock_ax2},
            'Matrix': {
                'sample_S1': {
                    'S1_enrichment.csv': pd.DataFrame({'A': [1, 2]})
                },
                'sample_S2': {
                    'S2_enrichment.csv': pd.DataFrame({'B': [3, 4]})
                }
            }
        }
        
        params_stratified = self.params.copy()
        params_stratified["Stratify_By"] = ["sample"]
        params_stratified["Seed"] = "42"
        params_stratified["Radius"] = "50.0"
        params_stratified["Color_Bar_Range"] = "2.5"
        params_stratified["Spatial_Analysis_Method"] = (
            "Cluster Interaction Matrix"
        )
        
        result = run_from_json(params_stratified, show_plot=False)
        
        # Verify function was called correctly
        mock_spatial.assert_called_once()
        call_args = mock_spatial.call_args
        
        # Check specific parameter conversions
        self.assertEqual(call_args[1]['annotation'], "cell_type")
        self.assertEqual(
            call_args[1]['analysis_method'],
            "Cluster Interaction Matrix"
        )
        self.assertEqual(call_args[1]['stratify_by'], ["sample"])
        self.assertEqual(call_args[1]['seed'], 42)
        self.assertEqual(call_args[1]['radius'], 50.0)
        self.assertEqual(call_args[1]['n_neighs'], 6)
        self.assertEqual(call_args[1]['vmin'], -2.5)
        self.assertEqual(call_args[1]['vmax'], 2.5)
        self.assertEqual(call_args[1]['cmap'], "seismic")
        
        # Verify multiple files were saved (2 plots + 2 matrices)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 4)
        png_files = [f for f in result.keys() if f.endswith('.png')]
        csv_files = [f for f in result.keys() if f.endswith('.csv')]
        self.assertEqual(len(png_files), 2)
        self.assertEqual(len(csv_files), 2)




if __name__ == "__main__":
    unittest.main()