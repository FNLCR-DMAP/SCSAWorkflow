# tests/templates/test_interactive_spatial_plot_template.py
"""Unit tests for the Interactive Spatial Plot template."""

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

from spac.templates.interactive_spatial_plot_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "renamed_phenotypes": (
            ["TypeA", "TypeB"] * ((n_cells + 1) // 2)
        )[:n_cells],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    # Add spatial coordinates required for spatial plots
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 100
    # Add color mapping
    adata.uns["_spac_colors"] = {
        "TypeA": "#FF0000",
        "TypeB": "#0000FF"
    }
    return adata


class TestInteractiveSpatialPlotTemplate(unittest.TestCase):
    """Unit tests for the Interactive Spatial Plot template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "interactive_plot"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters - adjust based on template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Color_By": "Annotation",
            "Annotation_s_to_Highlight": ["renamed_phenotypes"],
            "Feature_to_Highlight": "None",
            "Table": "Original",
            "Dot_Size": 3,
            "Dot_Transparency": 0.75,
            "Feature_Color_Scale": "balance",
            "Figure_Width": 12,
            "Figure_Height": 12,
            "Figure_DPI": 200,
            "Font_Size": 12,
            "Stratify_By": "None",
            "Define_Label_Color_Mapping": "_spac_colors",
            "Lower_Colorbar_Bound": 999,
            "Upper_Colorbar_Bound": -999,
            "Flip_Vertical_Axis": False,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.interactive_spatial_plot_template.'
           'interactive_spatial_plot')
    @patch('plotly.io.to_html')
    def test_complete_io_workflow(
        self,
        mock_to_html,
        mock_spatial_plot,
    ) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the interactive_spatial_plot function
        mock_fig1 = MagicMock()
        mock_fig1.show = MagicMock()
        
        mock_spatial_plot.return_value = [
            {
                'image_name': 'plot_1',
                'image_object': mock_fig1
            }
        ]
        
        # Mock HTML conversion
        mock_to_html.return_value = "<html>Mock Plot</html>"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 1)
            expected_file = f"{self.out_file}_plot_1.html"
            self.assertIn(expected_file, result)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            self.assertIsNone(result_no_save)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

        # Verify interactive_spatial_plot was called correctly
        mock_spatial_plot.assert_called()
        call_args = mock_spatial_plot.call_args[1]
        self.assertEqual(call_args['annotations'], ["renamed_phenotypes"])
        self.assertIsNone(call_args['feature'])
        self.assertEqual(call_args['dot_size'], 3)
        self.assertEqual(call_args['reverse_y_axis'], False)

    def test_error_validation(self) -> None:
        """Test exact error messages for invalid parameters."""
        # Test missing annotation when Color_By is "Annotation"
        params_bad = self.params.copy()
        params_bad["Annotation_s_to_Highlight"] = []
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        expected_msg = (
            'Please set at least one value in the "Annotation(s) to '
            'Highlight" parameter'
        )
        self.assertEqual(str(context.exception), expected_msg)
        
        # Test missing feature when Color_By is "Feature"
        params_bad2 = self.params.copy()
        params_bad2["Color_By"] = "Feature"
        params_bad2["Feature_to_Highlight"] = "None"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad2)
        
        expected_msg = 'Please set the "Feature to Highlight" parameter.'
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.interactive_spatial_plot_template.'
           'interactive_spatial_plot')
    def test_function_calls(self, mock_spatial_plot) -> None:
        """Test that main function is called with correct parameters."""
        # Mock multiple plots with stratification
        mock_fig1 = MagicMock()
        mock_fig2 = MagicMock()
        
        mock_spatial_plot.return_value = [
            {'image_name': 'S1', 'image_object': mock_fig1},
            {'image_name': 'S2', 'image_object': mock_fig2}
        ]
        
        # Test with stratification
        params_strat = self.params.copy()
        params_strat["Stratify_By"] = "sample"
        params_strat["Color_By"] = "Feature"
        params_strat["Feature_to_Highlight"] = "Gene1"
        params_strat["Annotation_s_to_Highlight"] = [""]
        
        with patch('plotly.io.to_html', return_value="<html>Mock</html>"):
            run_from_json(params_strat)
        
        # Verify function was called correctly
        mock_spatial_plot.assert_called_once()
        call_args = mock_spatial_plot.call_args[1]
        
        # Check parameter conversions
        self.assertIsNone(call_args['annotations'])
        self.assertEqual(call_args['feature'], "Gene1")
        self.assertEqual(call_args['stratify_by'], "sample")
        self.assertEqual(call_args['defined_color_map'], "_spac_colors")


if __name__ == "__main__":
    unittest.main()