# tests/templates/test_rename_labels_template.py
"""Unit tests for the Rename Labels template."""

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

from spac.templates.rename_labels_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "phenograph_k60_r1": [str(i % 3) for i in range(n_cells)],
        "sample": (["S1", "S2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    return adata


class TestRenameLabelsTemplate(unittest.TestCase):
    """Unit tests for the Rename Labels template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.mapping_file = os.path.join(
            self.tmp_dir.name, "mapping.csv"
        )
        self.out_file = "output"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Create mapping CSV - pandas will read these as integers
        mapping_df = pd.DataFrame({
            'Original': [0, 1, 2],
            'New': ['TypeA', 'TypeB', 'TypeC']
        })
        mapping_df.to_csv(self.mapping_file, index=False)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Cluster_Mapping_Dictionary": self.mapping_file,
            "Source_Annotation": "phenograph_k60_r1",
            "New_Annotation": "renamed_phenotypes",
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
            # Verify file was saved
            self.assertTrue(len(result) > 0)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type based on template
            self.assertIsInstance(result_no_save, ad.AnnData)
            # Verify new annotation exists
            self.assertIn("renamed_phenotypes", result_no_save.obs.columns)
            # Verify mapping was applied
            unique_labels = result_no_save.obs["renamed_phenotypes"].unique()
            self.assertIn("TypeA", unique_labels)
            self.assertIn("TypeB", unique_labels)
            self.assertIn("TypeC", unique_labels)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test missing required mapping columns
        bad_mapping_df = pd.DataFrame({
            'Wrong': ['0', '1', '2'],
            'Columns': ['TypeA', 'TypeB', 'TypeC']
        })
        bad_mapping_file = os.path.join(
            self.tmp_dir.name, "bad_mapping.csv"
        )
        bad_mapping_df.to_csv(bad_mapping_file, index=False)
        
        params_bad = self.params.copy()
        params_bad["Cluster_Mapping_Dictionary"] = bad_mapping_file
        
        with self.assertRaises(KeyError) as context:
            run_from_json(params_bad)
        
        # Check that error occurs when accessing 'Original' column
        self.assertIn("Original", str(context.exception))

    @patch('spac.templates.rename_labels_template.rename_annotations')
    def test_function_calls(self, mock_rename) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the rename_annotations function to simulate its effect
        def side_effect_rename(adata, src_annotation, dest_annotation, 
                               mappings):
            # Simulate what rename_annotations does
            # Convert string values to int if mapping keys are int
            if all(isinstance(k, int) for k in mappings.keys()):
                adata.obs[dest_annotation] = (
                    adata.obs[src_annotation].astype(int).map(mappings)
                )
            else:
                adata.obs[dest_annotation] = (
                    adata.obs[src_annotation].map(mappings)
                )
            return None
        
        mock_rename.side_effect = side_effect_rename
        
        # Run the template
        result = run_from_json(self.params, save_results=False)
        
        # Verify function was called correctly
        mock_rename.assert_called_once()
        call_args = mock_rename.call_args
        
        # Check that AnnData was passed
        self.assertIsInstance(call_args[0][0], ad.AnnData)
        
        # Check keyword arguments
        self.assertEqual(
            call_args[1]['src_annotation'], "phenograph_k60_r1"
        )
        self.assertEqual(
            call_args[1]['dest_annotation'], "renamed_phenotypes"
        )
        expected_mappings = {0: 'TypeA', 1: 'TypeB', 2: 'TypeC'}
        self.assertEqual(call_args[1]['mappings'], expected_mappings)


if __name__ == "__main__":
    unittest.main()