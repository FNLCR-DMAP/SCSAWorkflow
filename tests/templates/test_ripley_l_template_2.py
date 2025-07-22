"""Unit tests for the Ripley-L template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.ripley_l_template import run_from_json, _convert_to_floats


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "phenotype": ["B cells", "CD8 T cells"] * (n_cells // 2)
    })
    x_mat = rng.normal(size=(n_cells, 2))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 50.0
    return adata


class TestRipleyLTemplate(unittest.TestCase):
    """Unit tests for the Ripley-L template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.h5ad")
        # Change to pickle format to avoid h5ad serialization issues
        self.out_file = "output.pickle"

        # Save minimal mock data
        mock_adata().write_h5ad(self.in_file)

        # Minimal parameters - use pickle output to avoid serialization issues
        self.params = {
            "Upstream_Analysis": self.in_file,
            "radii": [10, 20],
            "annotation": "phenotype",
            "center_phenotype": "B cells",
            "neighbor_phenotype": "CD8 T cells",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single comprehensive I/O test covering all input/output scenarios."""
        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Dict input with h5ad file, pickle output
            saved_files = run_from_json(self.params)
            self.assertIn(self.out_file, saved_files)
            self.assertTrue(os.path.exists(saved_files[self.out_file]))
            
            # Load and verify output structure from pickle
            with open(saved_files[self.out_file], 'rb') as f:
                adata = pickle.load(f)
            # Check that ripley_l results exist (key might vary)
            ripley_keys = [k for k in adata.uns.keys() if 'ripley' in k.lower()]
            self.assertTrue(len(ripley_keys) > 0, "No Ripley-L results found in uns")
            self.assertEqual(adata.n_obs, 10)
            
            # Test 2: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            saved_files_json = run_from_json(json_path)
            self.assertIn(self.out_file, saved_files_json)
            
            # Test 3: Pickle input file
            pickle_file = os.path.join(self.tmp_dir.name, "input.pickle")
            with open(pickle_file, "wb") as f:
                pickle.dump(mock_adata(), f)
            params_pickle = self.params.copy()
            params_pickle["Upstream_Analysis"] = pickle_file
            saved_files_pickle = run_from_json(params_pickle)
            self.assertIn(self.out_file, saved_files_pickle)
            
            # Test 4: Parameter conversion (string radii, text values)
            params_str = self.params.copy()
            params_str["radii"] = ["10", "20.5"]
            params_str["area"] = "100.0"
            params_str["stratify_by"] = "None"
            saved_files_str = run_from_json(params_str)
            self.assertIn(self.out_file, saved_files_str)

    def test_convert_to_floats_error_message(self) -> None:
        """Test exact error message for invalid radius conversion."""
        with self.assertRaises(ValueError) as context:
            _convert_to_floats(["10", "invalid", "20"])
        
        expected_msg = "Failed to convert the radius: 'invalid' to float."
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_missing_input_file_error_message(self) -> None:
        """Test exact error message for missing input file."""
        params_bad = self.params.copy()
        params_bad["Upstream_Analysis"] = "/nonexistent/file.h5ad"
        
        with self.assertRaises(FileNotFoundError) as context:
            run_from_json(params_bad)
        
        expected_msg = "Input file not found: /nonexistent/file.h5ad"
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_invalid_json_input_error_message(self) -> None:
        """Test exact error message for invalid JSON input type."""
        from spac.templates.ripley_l_template import _parse_params
        
        with self.assertRaises(TypeError) as context:
            _parse_params(123)  # Invalid type
        
        expected_msg = "json_input must be dict, JSON string, or path to JSON file"
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)


if __name__ == "__main__":
    unittest.main()