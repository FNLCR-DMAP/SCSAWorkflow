# tests/templates/test_hierarchical_heatmap_template.py
"""Unit tests for the Hierarchical Heatmap template."""

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

from spac.templates.hierarchical_heatmap_template import run_from_json


def mock_adata(n_cells: int = 100) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "phenograph_k60_r1": (
            ["Cluster1", "Cluster2", "Cluster3"] * ((n_cells + 2) // 3)
        )[:n_cells],
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells]
    })
    # Create expression data with 9 markers as in the example
    x_mat = rng.normal(size=(n_cells, 9))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [
        "Hif1a", "NOS2", "COX2", "β-catenin", "vimentin", 
        "E-cadherin", "Ki67", "PIMO", "aSMA"
    ]
    # Add a z-score normalized layer for testing
    adata.layers["arcsinh_z_scores"] = (
        (x_mat - x_mat.mean(axis=0)) / x_mat.std(axis=0)
    )
    return adata


class TestHierarchicalHeatmapTemplate(unittest.TestCase):
    """Unit tests for the Hierarchical Heatmap template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "mean_intensity.csv"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - from the JSON template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "phenograph_k60_r1",
            "Table_to_Visualize": "arcsinh_z_scores",
            "Feature_s_": ["All"],
            "Standard_Scale_": "None",
            "Z_Score": "None",
            "Feature_Dendrogram": True,
            "Annotation_Dendrogram": True,
            "Figure_Title": "Hierarchical Heatmap",
            "Figure_Width": 15,
            "Figure_Height": 12,
            "Figure_DPI": 300,
            "Font_Size": 14,
            "Matrix_Plot_Ratio": 0.8,
            "Swap_Axes": False,
            "Rotate_Label_": False,
            "Horizontal_Dendrogram_Display_Ratio": 0.2,
            "Vertical_Dendrogram_Display_Ratio": 0.2,
            "Value_Min": "-3",
            "Value_Max": "3",
            "Color_Map": "seismic",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Mock the hierarchical_heatmap function
            mock_clustergrid = MagicMock()
            mock_clustergrid.fig = MagicMock()
            mock_clustergrid.ax_heatmap = MagicMock()
            mock_clustergrid.height = 12
            mock_clustergrid.width = 15
            
            mock_mean_intensity = pd.DataFrame({
                'Hif1a': [0.1, 0.2, 0.3],
                'NOS2': [0.4, 0.5, 0.6],
                'COX2': [0.7, 0.8, 0.9]
            }, index=['Cluster1', 'Cluster2', 'Cluster3'])
            
            mock_dendrogram_data = {
                'row_dendrogram': {'data': 'row_data'},
                'col_dendrogram': {'data': 'col_data'}
            }
            
            with patch(
                'spac.templates.hierarchical_heatmap_template.'
                'hierarchical_heatmap',
                return_value=(
                    mock_mean_intensity, mock_clustergrid, 
                    mock_dendrogram_data
                )
            ):
                
                # Test 1: Run with default parameters
                result = run_from_json(self.params, show_plot=False)
                self.assertIsInstance(result, dict)
                self.assertIn(self.out_file, result)
                
                # Test 2: Run without saving
                result_no_save = run_from_json(
                    self.params, save_results=False, show_plot=False
                )
                # Check appropriate return type based on template
                self.assertIsInstance(result_no_save, pd.DataFrame)
                self.assertEqual(len(result_no_save), 3)  # 3 clusters
                
                # Test 3: JSON file input
                json_path = os.path.join(
                    self.tmp_dir.name, "params.json"
                )
                with open(json_path, "w") as f:
                    json.dump(self.params, f)
                
                result_json = run_from_json(json_path, show_plot=False)
                self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid standard scale conversion
        params_bad = self.params.copy()
        params_bad["Standard_Scale_"] = "invalid_number"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad, show_plot=False)
        
        # Check exact error message
        expected_msg = (
            "Error: can't convert Standard Scale to integer. "
            "Received:\"invalid_number\""
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.hierarchical_heatmap_template.'
           'hierarchical_heatmap')
    def test_function_calls(self, mock_heatmap) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function
        mock_clustergrid = MagicMock()
        mock_clustergrid.fig = MagicMock()
        mock_clustergrid.ax_heatmap = MagicMock()
        
        # Create a proper dataframe with matching dimensions
        mock_mean_intensity = pd.DataFrame({
            'Hif1a': [0.1, 0.2, 0.3],
            'NOS2': [0.4, 0.5, 0.6],
            'COX2': [0.7, 0.8, 0.9]
        }, index=['Cluster1', 'Cluster2', 'Cluster3'])
        
        mock_heatmap.return_value = (
            mock_mean_intensity, mock_clustergrid, {}
        )
        
        # Test with swap_axes=True to verify features handling
        params_swap = self.params.copy()
        params_swap["Swap_Axes"] = True
        params_swap["Feature_s_"] = ["Hif1a", "NOS2"]
        
        run_from_json(params_swap, save_results=False, show_plot=False)
        
        # Verify function was called correctly
        mock_heatmap.assert_called_once()
        call_kwargs = mock_heatmap.call_args[1]
        
        # Check specific parameter conversions
        self.assertEqual(
            call_kwargs['annotation'], "phenograph_k60_r1"
        )
        self.assertEqual(call_kwargs['layer'], "arcsinh_z_scores")
        self.assertEqual(call_kwargs['features'], ["Hif1a", "NOS2"])
        self.assertEqual(call_kwargs['swap_axes'], True)
        self.assertEqual(call_kwargs['vmin'], -3.0)
        self.assertEqual(call_kwargs['vmax'], 3.0)
        self.assertEqual(call_kwargs['cmap'], "seismic")


if __name__ == "__main__":
    unittest.main()