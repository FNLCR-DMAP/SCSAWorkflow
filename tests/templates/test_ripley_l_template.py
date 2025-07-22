# tests/templates/test_ripley_l_template.py
"""Unit tests for the Ripley‑L template."""

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
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.ripley_l_template import (
    run_from_json, 
    _convert_to_floats
)


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
        self.out_file = "output.h5ad"

        # Save minimal mock data
        mock_adata().write_h5ad(self.in_file)

        # Minimal parameters - match the exact parameter names from template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Radii": [10, 20],
            "Annotation": "phenotype",
            "Center_Phenotype": "B cells",
            "Neighbor_Phenotype": "CD8 T cells",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_ripley_l_analysis_workflow(self) -> None:
        """Test Ripley-L specific analysis workflow and output validation."""
        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test 1: Basic Ripley-L analysis
            saved_files = run_from_json(self.params)
            self.assertIn(self.out_file, saved_files)
            self.assertTrue(os.path.exists(saved_files[self.out_file]))

            # Load and verify Ripley-L specific output structure
            adata = ad.read_h5ad(saved_files[self.out_file])
            
            # Check that ripley_l results exist (key might vary)
            ripley_keys = [
                k for k in adata.uns.keys() if 'ripley' in k.lower()
            ]
            self.assertTrue(
                len(ripley_keys) > 0, "No Ripley-L results found in uns"
            )
            self.assertEqual(adata.n_obs, 10)

            # Test 2: JSON file input (Ripley-L specific parameters)
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            saved_files_json = run_from_json(json_path)
            self.assertIn(self.out_file, saved_files_json)

            # Test 3: Parameter conversion (Ripley-L specific string 
            # parameters)
            params_str = self.params.copy()
            params_str["Radii"] = ["10", "20.5"]  # String radii
            params_str["Area"] = "100.0"  # String area
            params_str["Stratify_By"] = "None"  # Text none value
            params_str["Number_of_Simulations"] = 100
            params_str["Seed"] = 42
            params_str["Edge_Correction"] = True
            saved_files_str = run_from_json(params_str)
            self.assertIn(self.out_file, saved_files_str)

    def test_convert_to_floats_error_message(self) -> None:
        """Test exact error message for invalid radius conversion."""
        with self.assertRaises(ValueError) as context:
            _convert_to_floats(["10", "invalid", "20"])

        expected_msg = (
            "Failed to convert the radius: 'invalid' to float."
        )
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)


if __name__ == "__main__":
    unittest.main()