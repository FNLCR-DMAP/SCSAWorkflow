# tests/templates/test_sankey_plot_template.py
"""Unit tests for the Sankey Plot template."""

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

from spac.templates.sankey_plot_template import run_from_json


def mock_adata(n_cells: int = 20) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    
    # Create observations with source and target annotations
    # Ensure balanced distribution for sankey plot
    source_types = ["ClusterA", "ClusterB", "ClusterC"]
    target_types = ["PhenotypeX", "PhenotypeY", "PhenotypeZ"]
    
    # Create source annotations with proper repetition
    source_pattern = source_types * (
        (n_cells + len(source_types) - 1) // len(source_types)
    )
    source_annotation = source_pattern[:n_cells]
    
    # Create target annotations with some mixing
    target_annotation = []
    for i in range(n_cells):
        if i % 3 == 0:
            target_annotation.append(target_types[0])
        elif i % 3 == 1:
            target_annotation.append(target_types[1])
        else:
            target_annotation.append(target_types[2])
    
    obs = pd.DataFrame({
        "phenograph_k60_r1": source_annotation,
        "renamed_phenotypes": target_annotation
    })
    
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Marker1", "Marker2", "Marker3"]
    
    return adata


class TestSankeyPlotTemplate(unittest.TestCase):
    """Unit tests for the Sankey Plot template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "sankey"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Source_Annotation_Name": "phenograph_k60_r1",
            "Target_Annotation_Name": "renamed_phenotypes",
            "Source_Annotation_Color_Map": "tab20",
            "Target_Annotation_Color_Map": "tab20b",
            "Figure_Width_inch": 6,
            "Figure_Height_inch": 6,
            "Figure_DPI": 300,
            "Font_Size": 12,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.sankey_plot_template.sankey_plot')
    @patch('plotly.io.write_image')
    @patch('plotly.io.write_html')
    @patch('matplotlib.pyplot.imread')
    def test_complete_io_workflow(
        self, mock_imread, mock_write_html, mock_write_image, mock_sankey
    ) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the sankey_plot function to return a plotly figure
        mock_fig = MagicMock()
        mock_fig.update_layout = MagicMock()
        mock_fig.show = MagicMock()
        mock_sankey.return_value = mock_fig
        
        # Mock plt.imread to return a dummy image array
        mock_imread.return_value = np.zeros((100, 100, 3))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params, show_plot=False)
            self.assertIsInstance(result, dict)
            
            # Verify multiple files were saved
            self.assertIn("sankey_static.png", result)
            self.assertIn("sankey_interactive.html", result)
            self.assertIn("sankey_diagram.png", result)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(
                self.params, save_results=False, show_plot=False
            )
            # Should return None for multi-plot template
            self.assertIsNone(result_no_save)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path, show_plot=False)
            self.assertIsInstance(result_json, dict)

        # Verify sankey_plot was called with correct parameters
        mock_sankey.assert_called()
        call_args = mock_sankey.call_args
        self.assertEqual(
            call_args[1]['source_annotation'], "phenograph_k60_r1"
        )
        self.assertEqual(
            call_args[1]['target_annotation'], "renamed_phenotypes"
        )
        self.assertEqual(call_args[1]['source_color_map'], "tab20")
        self.assertEqual(call_args[1]['target_color_map'], "tab20b")
        self.assertEqual(call_args[1]['sankey_font'], 12)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test missing annotation column
        params_bad = self.params.copy()
        params_bad["Source_Annotation_Name"] = "nonexistent_column"
        
        # This should trigger an error in the sankey_plot function
        # For this test, we'll check that parameters are processed correctly
        with patch('spac.templates.sankey_plot_template.sankey_plot') as \
                mock_sankey:
            mock_sankey.side_effect = KeyError("nonexistent_column")
            
            with self.assertRaises(KeyError) as context:
                run_from_json(params_bad)
            
            self.assertIn("nonexistent_column", str(context.exception))

    @patch('spac.templates.sankey_plot_template.sankey_plot')
    @patch('plotly.io.write_image')
    @patch('plotly.io.write_html')
    @patch('matplotlib.pyplot.imread')
    def test_function_calls(
        self, mock_imread, mock_write_html, mock_write_image, mock_sankey
    ) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the plotly figure
        mock_fig = MagicMock()
        mock_sankey.return_value = mock_fig
        
        # Mock plt.imread to return a dummy image array
        mock_imread.return_value = np.zeros((100, 100, 3))
        
        # Test with None annotations (should use text_to_value)
        params_none = self.params.copy()
        params_none["Source_Annotation_Name"] = "None"
        params_none["Target_Annotation_Name"] = "None"
        
        run_from_json(params_none, save_results=False, show_plot=False)
        
        # Verify function was called with None values
        call_args = mock_sankey.call_args
        self.assertIsNone(call_args[1]['source_annotation'])
        self.assertIsNone(call_args[1]['target_annotation'])


if __name__ == "__main__":
    unittest.main()