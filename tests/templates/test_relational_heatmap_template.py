# tests/templates/test_relational_heatmap_template.py
"""Unit tests for the Relational Heatmap template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock, Mock

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # Headless backend for CI

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.relational_heatmap_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "phenograph_k60_r1": (["cluster1", "cluster2", "cluster3"] * 
                             ((n_cells + 2) // 3))[:n_cells],
        "renamed_phenotypes": (["phenotype_A", "phenotype_B"] * 
                              ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    return adata


class TestRelationalHeatmapTemplate(unittest.TestCase):
    """Unit tests for the Relational Heatmap template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "relational_heatmap"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Source_Annotation_Name": "phenograph_k60_r1",
            "Target_Annotation_Name": "renamed_phenotypes",
            "Colormap": "darkmint",
            "Figure_Width_inch": 8,
            "Figure_Height_inch": 10,
            "Figure_DPI": 300,
            "Font_Size": 8,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.relational_heatmap_template.relational_heatmap')
    @patch('plotly.io.write_image')
    @patch('matplotlib.pyplot.show')  # Mock plt.show()
    def test_complete_io_workflow(
        self, mock_plt_show, mock_write_image, mock_relational
    ) -> None:
        """Single I/O test covering input/output scenarios."""
        # Mock the relational_heatmap function
        mock_fig = Mock()
        mock_fig.show = Mock()  # Mock the fig.show() method
        
        mock_df = pd.DataFrame({
            'source': ['cluster1', 'cluster2'],
            'target': ['phenotype_A', 'phenotype_B'],
            'value': [5, 3]
        })
        
        mock_relational.return_value = {
            'file_name': 'relational_heatmap.csv',
            'data': mock_df,
            'figure': mock_fig
        }
        
        # Mock the plotly write_image to create a dummy image
        def create_dummy_image(fig, path, **kwargs):
            # Create a minimal PNG file
            # Ensure file path exists (for NamedTemporaryFile)
            if not os.path.exists(path):
                # Create parent directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)
            fig_dummy, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, 'test', ha='center', va='center')
            plt.savefig(path, dpi=72)
            plt.close(fig_dummy)
        
        mock_write_image.side_effect = create_dummy_image
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            # Should have both CSV and PNG files
            self.assertEqual(len(result), 2)
            csv_files = [f for f in result.keys() if f.endswith('.csv')]
            png_files = [f for f in result.keys() if f.endswith('.png')]
            self.assertEqual(len(csv_files), 1)
            self.assertEqual(len(png_files), 1)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(
                self.params, save_results=False
            )
            # Check appropriate return type - should be (figure, dataframe)
            self.assertIsInstance(result_no_save, tuple)
            self.assertEqual(len(result_no_save), 2)
            fig, df = result_no_save
            self.assertIsInstance(df, pd.DataFrame)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    @patch('matplotlib.pyplot.show')  # Mock plt.show()
    def test_error_validation(self, mock_plt_show) -> None:
        """Test exact error message for invalid parameters."""
        # Test with None annotations (should be handled by text_to_value)
        params_none = self.params.copy()
        params_none["Source_Annotation_Name"] = "None"
        params_none["Target_Annotation_Name"] = "None"
        
        with patch('spac.templates.relational_heatmap_template.'
                   'relational_heatmap') as mock_rel:
            # The template should pass None values to the function
            mock_rel.return_value = {
                'file_name': 'test.csv',
                'data': pd.DataFrame(),
                'figure': Mock(show=Mock())  # Mock fig.show()
            }
            
            # Mock write_image to create a dummy file
            def create_dummy_image(fig, path, **kwargs):
                # Create a minimal PNG file
                # Ensure file path exists (for NamedTemporaryFile)
                if not os.path.exists(path):
                    # Create parent directory if needed
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                fig_dummy, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, 'test', ha='center', va='center')
                plt.savefig(path, dpi=72)
                plt.close(fig_dummy)
            
            with patch('plotly.io.write_image', 
                      side_effect=create_dummy_image):
                run_from_json(params_none)
            
            # Verify None was passed
            call_args = mock_rel.call_args
            self.assertIsNone(call_args[1]['source_annotation'])
            self.assertIsNone(call_args[1]['target_annotation'])

    @patch('spac.templates.relational_heatmap_template.relational_heatmap')
    @patch('plotly.io.write_image')
    @patch('matplotlib.pyplot.show')  # Mock plt.show()
    def test_function_calls(
        self, mock_plt_show, mock_write_image, mock_relational
    ) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function
        mock_relational.return_value = {
            'file_name': 'test.csv',
            'data': pd.DataFrame({'a': [1, 2]}),
            'figure': Mock(show=Mock())  # Mock fig.show()
        }
        
        # Mock write_image to create a dummy file
        def create_dummy_image(fig, path, **kwargs):
            # Create a minimal PNG file
            # Ensure file path exists (for NamedTemporaryFile)
            if not os.path.exists(path):
                # Create parent directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)
            fig_dummy, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, 'test', ha='center', va='center')
            plt.savefig(path, dpi=72)
            plt.close(fig_dummy)
        
        mock_write_image.side_effect = create_dummy_image
        
        run_from_json(self.params, save_results=False)
        
        # Verify function was called correctly
        mock_relational.assert_called_once()
        call_args = mock_relational.call_args
        
        # Check specific parameters
        self.assertEqual(
            call_args[1]['source_annotation'], 'phenograph_k60_r1'
        )
        self.assertEqual(
            call_args[1]['target_annotation'], 'renamed_phenotypes'
        )
        self.assertEqual(call_args[1]['color_map'], 'darkmint')
        self.assertEqual(call_args[1]['font_size'], 8)


if __name__ == "__main__":
    unittest.main()