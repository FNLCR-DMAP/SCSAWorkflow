# tests/templates/test_spatial_plot_template.py
"""Unit tests for the Spatial Plot template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock, call

import matplotlib
matplotlib.use("Agg")  # Headless backend for CI

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.spatial_plot_template import run_from_json


def mock_adata_spatial(n_cells: int = 100) -> ad.AnnData:
    """Return a minimal synthetic AnnData with spatial coordinates."""
    rng = np.random.default_rng(0)
    
    # Create expression matrix
    n_features = 5
    X = rng.normal(size=(n_cells, n_features))
    
    # Create annotations
    obs = pd.DataFrame({
        "cell_type": rng.choice(["TypeA", "TypeB", "TypeC"], n_cells),
        "region": rng.choice(["Region1", "Region2"], n_cells),
        "slide": rng.choice(["Slide1", "Slide2"], n_cells)
    })
    
    # Create spatial coordinates
    spatial_coords = rng.random((n_cells, 2)) * 1000
    
    # Create AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["spatial"] = spatial_coords
    
    # Add feature names
    adata.var_names = [f"Feature_{i}" for i in range(n_features)]
    
    return adata


class TestSpatialPlotTemplate(unittest.TestCase):
    """Unit tests for the Spatial Plot template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "spatial_data.pickle"
        )
        self.out_file = "spatial_plot"

        # Save minimal mock data with spatial info
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata_spatial(), f)

        # Minimal parameters
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation_to_Highlight": "cell_type",
            "Feature_to_Highlight": "",
            "Table": "Original",
            "Dot_Transparency": 0.5,
            "Dot_Size": 25,
            "Figure_Height": 6,
            "Figure_Width": 12,
            "Figure_DPI": 200,
            "Font_Size": 12,
            "Lower_Colorbar_Bound": 999,
            "Upper_Colorbar_Bound": -999,
            "Color_By": "Annotation",
            "Stratify": False,
            "Stratify_By": [],
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.spatial_plot_template.spatial_plot')
    @patch('matplotlib.pyplot.show')
    def test_run_single_plot_annotation(self, mock_show, mock_spatial) -> None:
        """Test single plot with annotation coloring."""
        # Mock the spatial_plot function to return axes
        mock_ax = MagicMock()
        mock_spatial.return_value = [mock_ax]
        
        # Test with annotation coloring
        saved_files = run_from_json(self.params)
        
        # Verify spatial_plot was called correctly
        mock_spatial.assert_called_once()
        call_kwargs = mock_spatial.call_args[1]
        self.assertEqual(call_kwargs['annotation'], 'cell_type')
        self.assertIsNone(call_kwargs['feature'])
        # layer should be None because text_to_value(layer, "Original") 
        # converts "Original" to None
        self.assertIsNone(call_kwargs['layer'])
        self.assertEqual(call_kwargs['spot_size'], 25)
        self.assertEqual(call_kwargs['alpha'], 0.5)
        
        # Check saved files
        self.assertIn(f"{self.out_file}.png", saved_files)
        
        # Verify show was called
        mock_show.assert_called_once()

    @patch('spac.templates.spatial_plot_template.spatial_plot')
    @patch('matplotlib.pyplot.show')
    def test_run_single_plot_feature(self, mock_show, mock_spatial) -> None:
        """Test single plot with feature coloring."""
        # Mock the spatial_plot function
        mock_ax = MagicMock()
        mock_spatial.return_value = [mock_ax]
        
        # Update params for feature coloring
        params_feature = self.params.copy()
        params_feature["Color_By"] = "Feature"
        params_feature["Feature_to_Highlight"] = "Feature_0"
        
        result = run_from_json(params_feature)
        
        # Verify spatial_plot was called with feature
        call_kwargs = mock_spatial.call_args[1]
        self.assertIsNone(call_kwargs['annotation'])
        self.assertEqual(call_kwargs['feature'], 'Feature_0')

    @patch('spac.templates.spatial_plot_template.spatial_plot')
    @patch('spac.templates.spatial_plot_template.select_values')
    @patch('matplotlib.pyplot.show')
    @patch('builtins.print')
    def test_run_stratified_plots(
        self, mock_print, mock_show, mock_select, mock_spatial
    ) -> None:
        """Test stratified plots generation."""
        # Mock functions
        mock_ax = MagicMock()
        mock_spatial.return_value = [mock_ax]
        
        # Mock select_values to return filtered data
        mock_adata = mock_adata_spatial(50)
        mock_select.return_value = mock_adata
        
        # Update params for stratification
        params_strat = self.params.copy()
        params_strat["Stratify"] = True
        params_strat["Stratify_By"] = ["region", "slide"]
        
        saved_files = run_from_json(params_strat)
        
        # Should save multiple files
        self.assertIsInstance(saved_files, dict)
        self.assertTrue(len(saved_files) > 1)
        
        # Verify unique values were printed
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list]
        # Should print unique values array
        self.assertTrue(any('Region' in str(call) for call in print_calls))

    def test_stratify_validation_error(self) -> None:
        """Test error when stratify is True but no stratify_by provided."""
        params_bad = self.params.copy()
        params_bad["Stratify"] = True
        params_bad["Stratify_By"] = []

        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)

        expected_msg = (
            'Please set at least one annotation in the "Stratify By" '
            'option, or set the "Stratify" to False.'
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.spatial_plot_template.spatial_plot')
    @patch('matplotlib.pyplot.show')
    def test_run_without_save(self, mock_show, mock_spatial) -> None:
        """Test running without saving files."""
        # Mock the spatial_plot function
        mock_ax = MagicMock()
        mock_spatial.return_value = [mock_ax]
        
        # Run with save_results=False
        result = run_from_json(self.params, save_results=False)
        
        # Should return None when not saving
        self.assertIsNone(result)

    @patch('spac.templates.spatial_plot_template.spatial_plot')
    @patch('matplotlib.pyplot.show')
    def test_max_plots_warning(self, mock_show, mock_spatial) -> None:
        """Test warning when too many unique stratification values."""
        # Create adata with many unique combinations
        adata = mock_adata_spatial(100)
        # Add a column with many unique values
        adata.obs['many_values'] = [f'Val_{i}' for i in range(100)]
        
        with open(self.in_file, 'wb') as f:
            pickle.dump(adata, f)
        
        # Mock functions
        mock_ax = MagicMock()
        mock_spatial.return_value = [mock_ax]
        
        params_many = self.params.copy()
        params_many["Stratify"] = True
        params_many["Stratify_By"] = ["many_values"]
        
        with patch('builtins.print') as mock_print:
            saved_files = run_from_json(params_many)
            
            # Check warning was printed
            print_calls = [
                str(call[0][0]) for call in mock_print.call_args_list
            ]
            warning_printed = any(
                'displaying only the first 20 plots' in call 
                for call in print_calls
            )
            self.assertTrue(warning_printed)

    def test_json_file_input(self) -> None:
        """Test with JSON file input."""
        json_path = os.path.join(self.tmp_dir.name, "params.json")
        with open(json_path, "w") as f:
            json.dump(self.params, f)
        
        with patch('spac.templates.spatial_plot_template.spatial_plot'):
            with patch('matplotlib.pyplot.show'):
                result = run_from_json(json_path, save_results=False)
        
        # Should return None when save_results=False
        self.assertIsNone(result)

    @patch('spac.templates.spatial_plot_template.spatial_plot')
    def test_text_to_value_processing(self, mock_spatial) -> None:
        """Test that text_to_value is applied to string parameters."""
        # Mock spatial_plot
        mock_ax = MagicMock()
        mock_spatial.return_value = [mock_ax]
        
        # Test with "None" strings
        params_none = self.params.copy()
        params_none["Annotation_to_Highlight"] = "None"
        params_none["Color_By"] = "Feature"
        params_none["Feature_to_Highlight"] = "Feature_0"
        
        with patch('matplotlib.pyplot.show'):
            run_from_json(params_none, save_results=False)
        
        # Annotation should be None after text_to_value
        call_kwargs = mock_spatial.call_args[1]
        self.assertIsNone(call_kwargs['annotation'])


if __name__ == "__main__":
    unittest.main()