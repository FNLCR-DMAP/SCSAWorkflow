# tests/templates/test_subset_analysis_template.py
"""Unit tests for the Subset Analysis template."""

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

from spac.templates.subset_analysis_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    
    # Create cell labels that match the example
    cell_labels = (
        ["Normal_Cells", "Cancer_Cells"] * ((n_cells + 1) // 2)
    )[:n_cells]
    
    obs = pd.DataFrame({
        "cell_labels": cell_labels,
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    return adata


class TestSubsetAnalysisTemplate(unittest.TestCase):
    """Unit tests for the Subset Analysis template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from JSON template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotation_of_interest": "cell_labels",
            "Labels": ["Normal_Cells", "Cancer_Cells"],
            "Include_Exclude": "Exclude Selected Labels",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with exclude parameters (default from setup)
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertTrue(len(result) > 0)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type based on template
            self.assertIsInstance(result_no_save, ad.AnnData)
            # Verify that cells were actually filtered
            original_adata = mock_adata()
            self.assertLess(len(result_no_save), len(original_adata))
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test with non-existent annotation
        params_bad = self.params.copy()
        params_bad["Annotation_of_interest"] = "non_existent_annotation"
        
        # This should raise an error when select_values tries to access
        # the annotation
        with self.assertRaises((KeyError, ValueError)):
            run_from_json(params_bad)

    @patch('spac.templates.subset_analysis_template.select_values')
    def test_function_calls(self, mock_select) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the select_values function to return filtered data
        filtered_adata = mock_adata(5)  # Smaller filtered dataset
        mock_select.return_value = filtered_adata
        
        # Test with "Include Selected Labels"
        params_include = self.params.copy()
        params_include["Include_Exclude"] = "Include Selected Labels"
        
        run_from_json(params_include, save_results=False)
        
        # Verify function was called correctly
        mock_select.assert_called_once()
        call_args = mock_select.call_args
        
        # Check that include mode sets values correctly
        self.assertEqual(call_args[1]['annotation'], "cell_labels")
        self.assertEqual(
            call_args[1]['values'], ["Normal_Cells", "Cancer_Cells"]
        )
        self.assertIsNone(call_args[1]['exclude_values'])
        
        # Reset mock
        mock_select.reset_mock()
        
        # Test with "Exclude Selected Labels"
        run_from_json(self.params, save_results=False)
        
        # Check that exclude mode sets values correctly
        call_args = mock_select.call_args
        self.assertIsNone(call_args[1]['values'])
        self.assertEqual(
            call_args[1]['exclude_values'], 
            ["Normal_Cells", "Cancer_Cells"]
        )


if __name__ == "__main__":
    unittest.main()