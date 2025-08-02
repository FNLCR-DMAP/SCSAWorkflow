# tests/templates/test_combine_annotations_template.py
"""Unit tests for the Combine Annotations template."""

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

from spac.templates.combine_annotations_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "detailed_cell_type": (
            ["B_cell", "T_cell", "NK_cell"] * ((n_cells + 2) // 3)
        )[:n_cells],
        "broad_cell_type": (
            ["Immune", "Immune", "Immune"] * ((n_cells + 2) // 3)
        )[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    return adata


class TestCombineAnnotationsTemplate(unittest.TestCase):
    """Unit tests for the Combine Annotations template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "transform_output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters matching NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Annotations_Names": ["detailed_cell_type", "broad_cell_type"],
            "New_Annotation_Name": "combined_annotation",
            "Separator": "_",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering all input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            # Should have 2 files: pickle and CSV
            self.assertEqual(len(result), 2)
            # Check pickle file exists
            self.assertIn(
                f"{self.out_file}.pickle", result
            )
            # Check CSV file exists
            self.assertIn("combined_annotation_counts.csv", result)
            
            # Test 2: Run without saving
            adata_result = run_from_json(self.params, save_results=False)
            # Check we got AnnData back
            self.assertIsInstance(adata_result, ad.AnnData)
            # Check new annotation was created
            self.assertIn("combined_annotation", adata_result.obs.columns)
            # Check annotation values are correct format
            sample_val = adata_result.obs["combined_annotation"].iloc[0]
            self.assertIn("_", sample_val)  # Should have separator
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)
            self.assertEqual(len(result_json), 2)
            
            # Test 4: Verify CSV content
            csv_path = result_json["combined_annotation_counts.csv"]
            df_counts = pd.read_csv(csv_path)
            # Should have the annotation name and count columns
            self.assertIn("combined_annotation", df_counts.columns)
            self.assertIn("count", df_counts.columns)
            # Should have some rows
            self.assertGreater(len(df_counts), 0)

    def test_parameter_validation(self) -> None:
        """Test exact error message for missing parameters."""
        params_bad = self.params.copy()
        del params_bad["Annotations_Names"]
        
        with self.assertRaises(KeyError) as context:
            run_from_json(params_bad)
        
        # Check that KeyError mentions the missing parameter
        self.assertIn("Annotations_Names", str(context.exception))

    @patch('spac.templates.combine_annotations_template.combine_annotations')
    def test_function_calls(self, mock_combine) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the combine_annotations function to add the expected column
        def mock_combine_side_effect(adata, **kwargs):
            # Simulate what combine_annotations does
            annotations = kwargs.get('annotations', [])
            separator = kwargs.get('separator', '_')
            new_name = kwargs.get('new_annotation_name', 'combined')
            
            # Create combined values
            combined_values = []
            for idx in range(len(adata.obs)):
                values = [str(adata.obs[col].iloc[idx]) for col in annotations]
                combined_values.append(separator.join(values))
            
            adata.obs[new_name] = combined_values
            return None
        
        mock_combine.side_effect = mock_combine_side_effect
        
        run_from_json(self.params)
        
        # Verify function was called correctly
        mock_combine.assert_called_once()
        call_args = mock_combine.call_args
        
        # Check positional arguments (adata)
        self.assertEqual(call_args[0][0].n_obs, 10)
        
        # Check keyword arguments
        self.assertEqual(
            call_args[1]['annotations'],
            ["detailed_cell_type", "broad_cell_type"]
        )
        self.assertEqual(call_args[1]['separator'], "_")
        self.assertEqual(
            call_args[1]['new_annotation_name'],
            "combined_annotation"
        )

    def test_different_separators(self) -> None:
        """Test with different separator characters."""
        # Test with hyphen separator
        params_hyphen = self.params.copy()
        params_hyphen["Separator"] = "-"
        params_hyphen["New_Annotation_Name"] = "hyphen_combined"
        
        adata = run_from_json(params_hyphen, save_results=False)
        self.assertIn("hyphen_combined", adata.obs.columns)
        # Check that values use hyphen
        sample_val = adata.obs["hyphen_combined"].iloc[0]
        self.assertIn("-", sample_val)
        
        # Test with empty separator
        params_empty = self.params.copy()
        params_empty["Separator"] = ""
        params_empty["New_Annotation_Name"] = "concat_combined"
        
        adata = run_from_json(params_empty, save_results=False)
        self.assertIn("concat_combined", adata.obs.columns)

    @patch('builtins.print')
    def test_console_output(self, mock_print) -> None:
        """Test that expected console output is produced."""
        run_from_json(self.params, save_results=False)
        
        # Check that print was called with expected messages
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list
                      if call[0]]
        
        # Should print after combining annotations
        after_combining = any(
            "After combining annotations:" in msg for msg in print_calls
        )
        self.assertTrue(after_combining)
        
        # Should print unique labels message
        unique_labels = any(
            "Unique labels in combined_annotation" in msg
            for msg in print_calls
        )
        self.assertTrue(unique_labels)


if __name__ == "__main__":
    unittest.main()