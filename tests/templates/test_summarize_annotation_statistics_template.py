# tests/templates/test_summarize_annotation_statistics_template.py
"""Unit tests for the Summarize Annotation's Statistics template."""

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

from spac.templates.summarize_annotation_statistics_template import (
    run_from_json
)


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "cluster": ([f"C{i % 3}" for i in range(n_cells)])[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 5))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(5)]
    # Add a layer for testing
    adata.layers["normalized"] = x_mat * 2.0
    return adata


class TestSummarizeAnnotationStatisticsTemplate(unittest.TestCase):
    """Unit tests for the Summarize Annotation's Statistics template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "annotation_summaries.csv"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "Feature_s_": ["All"],
            "Annotation": "cell_type",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            # Verify CSV file was saved
            self.assertIn(self.out_file, result)
            csv_path = result[self.out_file]
            self.assertTrue(os.path.exists(csv_path))
            
            # Verify CSV content structure
            df_saved = pd.read_csv(csv_path)
            # Should have renamed columns (no spaces/hyphens)
            for col in df_saved.columns:
                self.assertNotIn(" ", col)
                self.assertNotIn("-", col)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type
            self.assertIsInstance(result_no_save, pd.DataFrame)
            self.assertGreater(len(result_no_save), 0)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    @patch('spac.templates.summarize_annotation_statistics_template.'
           'get_cluster_info')
    def test_function_calls(self, mock_get_cluster_info) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function to return a simple DataFrame
        mock_df = pd.DataFrame({
            "cluster": ["TypeA", "TypeB"],
            "cell count": [5, 5],
            "mean-expression": [1.0, 2.0]
        })
        mock_get_cluster_info.return_value = mock_df
        
        # Test with different parameter combinations
        
        # Test 1: Default parameters
        run_from_json(self.params, save_results=False)
        mock_get_cluster_info.assert_called_once_with(
            adata=mock_get_cluster_info.call_args[1]['adata'],
            layer=None,  # "Original" → None
            annotation="cell_type",
            features=None  # ["All"] → None
        )
        
        # Test 2: With specific features and layer
        mock_get_cluster_info.reset_mock()
        params_features = self.params.copy()
        params_features["Feature_s_"] = ["Gene1", "Gene2"]
        params_features["Table_to_Process"] = "normalized"
        params_features["Annotation"] = "cluster"
        
        run_from_json(params_features, save_results=False)
        mock_get_cluster_info.assert_called_once_with(
            adata=mock_get_cluster_info.call_args[1]['adata'],
            layer="normalized",
            annotation="cluster",
            features=["Gene1", "Gene2"]
        )
        
        # Test 3: With "None" annotation
        mock_get_cluster_info.reset_mock()
        params_none = self.params.copy()
        params_none["Annotation"] = "None"
        
        run_from_json(params_none, save_results=False)
        mock_get_cluster_info.assert_called_once_with(
            adata=mock_get_cluster_info.call_args[1]['adata'],
            layer=None,
            annotation=None,  # "None" → None
            features=None
        )

    def test_column_renaming(self) -> None:
        """Test that columns with spaces and hyphens are renamed."""
        with patch('spac.templates.summarize_annotation_statistics_template.'
                   'get_cluster_info') as mock_func:
            # Create a DataFrame with problematic column names
            mock_df = pd.DataFrame({
                "cell type": ["A", "B"],
                "mean-expression": [1.0, 2.0],
                "std dev": [0.1, 0.2],
                "CD4+ count": [10, 20]
            })
            mock_func.return_value = mock_df
            
            # Run without saving to get the DataFrame directly
            result_df = run_from_json(self.params, save_results=False)
            
            # Check that columns are renamed
            expected_columns = ["cell_type", "mean_expression", "std_dev", 
                                "CD4+_count"]
            self.assertEqual(list(result_df.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()