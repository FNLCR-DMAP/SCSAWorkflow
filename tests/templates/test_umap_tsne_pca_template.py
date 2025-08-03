# tests/templates/test_umap_tsne_pca_template.py
"""Unit tests for the UMAP\\tSNE\\PCA Visualization template."""

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

from spac.templates.umap_tsne_pca_template import run_from_json


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
    # Add UMAP coordinates for visualization
    adata.obsm["X_umap"] = rng.random((n_cells, 2)) * 10
    adata.obsm["X_tsne"] = rng.random((n_cells, 2)) * 50
    adata.obsm["X_pca"] = rng.random((n_cells, 2)) * 5
    return adata


class TestUmapTsnePcaTemplate(unittest.TestCase):
    """Unit tests for the UMAP\\tSNE\\PCA Visualization template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "visualization"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - adjust based on template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation_to_Highlight": "cell_type",
            "Feature_to_Highlight": "None",
            "Table": "Original",
            "Dimension_Reduction_Method": "umap",
            "Figure_Width": 12,
            "Figure_Height": 12,
            "Font_Size": 12,
            "Figure_DPI": 300,
            "Legend_Location": "best",
            "Legend_Font_Size": 16,
            "Legend_Marker_Size": 5.0,
            "Color_By": "Annotation",
            "Dot_Size": 1,
            "Value_Min": "None",
            "Value_Max": "None",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Mock the dimensionality_reduction_plot function
            with patch('spac.templates.umap_tsne_pca_template.'
                      'dimensionality_reduction_plot') as mock_plot:
                # Create mock figure and axis
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_ax.get_figure.return_value = mock_fig
                mock_ax.get_legend.return_value = MagicMock()
                mock_fig.savefig = MagicMock()
                
                mock_plot.return_value = (mock_fig, mock_ax)
                
                # Test 1: Run with default parameters
                result = run_from_json(self.params)
                self.assertIsInstance(result, dict)
                # Check that output file has proper extension
                output_files = list(result.keys())
                self.assertTrue(
                    any(f.endswith('.png') for f in output_files)
                )
                
                # Test 2: Run without saving
                result_no_save = run_from_json(
                    self.params, save_results=False
                )
                # Check appropriate return type based on template
                self.assertIsInstance(result_no_save, tuple)
                fig, df = result_no_save
                self.assertEqual(fig, mock_fig)
                # This template doesn't return a dataframe
                self.assertIsNone(df)
                
                # Test 3: JSON file input
                json_path = os.path.join(
                    self.tmp_dir.name, "params.json"
                )
                with open(json_path, "w") as f:
                    json.dump(self.params, f)
                
                result_json = run_from_json(json_path)
                self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid float conversion for vmin
        params_bad = self.params.copy()
        params_bad["Value_Min"] = "invalid_number"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check exact error message
        expected_msg = (
            "Error: can't convert Value Min to float. "
            "Received:\"invalid_number\""
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.umap_tsne_pca_template.'
           'dimensionality_reduction_plot')
    def test_function_calls(self, mock_plot) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.get_figure.return_value = mock_fig
        mock_ax.get_legend.return_value = None
        mock_plot.return_value = (mock_fig, mock_ax)
        
        # Test with feature coloring instead of annotation
        params_feature = self.params.copy()
        params_feature["Color_By"] = "Feature"
        params_feature["Feature_to_Highlight"] = "Gene1"
        params_feature["Annotation_to_Highlight"] = "None"
        params_feature["Value_Min"] = "0.5"
        params_feature["Value_Max"] = "10.0"
        
        run_from_json(params_feature, save_results=False, show_plot=False)
        
        # Verify function was called correctly
        mock_plot.assert_called_once()
        call_args = mock_plot.call_args
        
        # Check that annotation is None and feature is set
        self.assertIsNone(call_args[1]['annotation'])
        self.assertEqual(call_args[1]['feature'], "Gene1")
        self.assertEqual(call_args[1]['vmin'], 0.5)
        self.assertEqual(call_args[1]['vmax'], 10.0)
        self.assertEqual(call_args[1]['method'], "umap")
        self.assertEqual(call_args[1]['point_size'], 1)


if __name__ == "__main__":
    unittest.main()