# tests/templates/test_visualize_nearest_neighbor_template.py
"""Unit tests for the Visualize Nearest Neighbor template."""

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

from spac.templates.visualize_nearest_neighbor_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "renamed_phenotypes": (["Tfh", "B_cells"] * ((n_cells + 1) // 2))[:n_cells],
        "image_id": (["Image1", "Image2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Marker1", "Marker2", "Marker3"]
    
    # Add spatial coordinates
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    
    # Add mock distance matrix with proper structure
    # Create a mock distance matrix in obsm
    # For nearest neighbor, we need distances from each phenotype
    unique_phenotypes = obs["renamed_phenotypes"].unique()
    n_phenotypes = len(unique_phenotypes)
    
    # Create random distance matrix
    distance_matrix = rng.exponential(scale=10, size=(n_cells, n_phenotypes))
    distance_df = pd.DataFrame(
        distance_matrix,
        columns=[f"distance_to_{pheno}" for pheno in unique_phenotypes],
        index=adata.obs.index  # Use the same index as adata.obs
    )
    adata.obsm["spatial_distance"] = distance_df
    
    return adata


class TestVisualizeNearestNeighborTemplate(unittest.TestCase):
    """Unit tests for the Visualize Nearest Neighbor template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "nearest_neighbor_plots.csv"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - adjust based on template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "renamed_phenotypes",
            "ImageID": "None",
            "Plot_Method": "numeric",
            "Plot_Type": "boxen",
            "Source_Anchor_Cell_Label": "Tfh",
            "Target_Cell_Label": "All",
            "Nearest_Neighbor_Associated_Table": "spatial_distance",
            "Log_Scale": False,
            "Facet_Plot": False,
            "X_Axis_Label_Rotation": 0,
            "Shared_X_Axis_Title_": True,
            "X_Axis_Title_Font_Size": "None",
            "Defined_Color_Mapping": "None",
            "Figure_Width": 12,
            "Figure_Height": 6,
            "FIgure_DPI": 300,
            "Font_Size": 12,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.visualize_nearest_neighbor_template.'
           'visualize_nearest_neighbor')
    def test_complete_io_workflow(self, mock_vis_nn) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the visualize_nearest_neighbor function
        mock_fig = MagicMock()
        mock_fig.number = 1
        mock_fig.set_size_inches = MagicMock()
        mock_fig.set_dpi = MagicMock()
        mock_fig.suptitle = MagicMock()
        mock_fig.tight_layout = MagicMock()
        mock_fig.supxlabel = MagicMock()
        
        mock_ax = MagicMock()
        mock_ax.get_legend.return_value = None
        mock_ax.get_xlabel.return_value = "distance"
        mock_ax.set_xlabel = MagicMock()
        mock_ax.xaxis.get_label.return_value = MagicMock()
        
        # Create mock dataframe with expected columns
        mock_df = pd.DataFrame({
            'group': ['B_cells', 'Tfh'] * 5,
            'distance': np.random.rand(10) * 10,
            'image_id': ['Image1'] * 5 + ['Image2'] * 5
        })
        
        # Mock palette
        mock_palette = {
            'B_cells': '#1f77b4',
            'Tfh': '#ff7f0e'
        }
        
        mock_vis_nn.return_value = {
            "data": mock_df,
            "fig": mock_fig,
            "palette": mock_palette,
            "ax": mock_ax
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params, show_plot=False)
            self.assertIsInstance(result, dict)
            self.assertIn(self.out_file, result)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(
                self.params, save_results=False, show_plot=False
            )
            # Check appropriate return type - should be tuple
            self.assertIsInstance(result_no_save, tuple)
            self.assertEqual(len(result_no_save), 2)
            # First element is figure, second is dataframe
            self.assertIsInstance(result_no_save[1], pd.DataFrame)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path, show_plot=False)
            self.assertIsInstance(result_json, dict)

        # Verify visualize_nearest_neighbor was called correctly
        mock_vis_nn.assert_called()
        call_args = mock_vis_nn.call_args[1]
        self.assertEqual(call_args['annotation'], "renamed_phenotypes")
        self.assertEqual(call_args['spatial_distance'], "spatial_distance")
        self.assertEqual(call_args['distance_from'], "Tfh")
        self.assertEqual(call_args['distance_to'], None)  # "All" → None
        self.assertEqual(call_args['method'], "numeric")
        self.assertEqual(call_args['plot_type'], "boxen")
        self.assertEqual(call_args['log'], False)
        self.assertEqual(call_args['facet_plot'], False)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid integer conversion for x_axis_title_fontsize
        params_bad = self.params.copy()
        params_bad["X_Axis_Title_Font_Size"] = "invalid_size"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check exact error message from text_to_value
        expected_msg = (
            "Error: can't convert  to integer. "
            "Received:\"invalid_size\""
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.visualize_nearest_neighbor_template.'
           'visualize_nearest_neighbor')
    def test_function_calls(self, mock_vis_nn) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_legend.return_value = None
        mock_ax.get_xlabel.return_value = ""
        
        mock_df = pd.DataFrame({
            'group': ['B_cells', 'Tfh', 'Stroma'] * 2,
            'distance': [5.0, 7.0, 3.0, 6.0, 8.0, 4.0]
        })
        
        mock_vis_nn.return_value = {
            "data": mock_df,
            "fig": mock_fig,
            "palette": {'B_cells': '#000', 'Tfh': '#fff', 'Stroma': '#ccc'},
            "ax": mock_ax
        }
        
        # Test with different parameters
        params_alt = self.params.copy()
        params_alt["Target_Cell_Label"] = "B_cells,Stroma"
        params_alt["Log_Scale"] = True
        params_alt["Facet_Plot"] = True
        params_alt["ImageID"] = "image_id"
        params_alt["X_Axis_Label_Rotation"] = 45
        
        run_from_json(params_alt, save_results=False, show_plot=False)
        
        # Verify function was called correctly
        mock_vis_nn.assert_called_once()
        call_args = mock_vis_nn.call_args[1]
        
        # Check specific parameter conversions
        self.assertEqual(call_args['distance_to'], ["B_cells", "Stroma"])
        self.assertEqual(call_args['log'], True)
        self.assertEqual(call_args['facet_plot'], True)
        self.assertEqual(call_args['stratify_by'], "image_id")


if __name__ == "__main__":
    unittest.main()