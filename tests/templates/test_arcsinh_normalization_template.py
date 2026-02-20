# tests/templates/test_arcsinh_normalization_template.py
"""
Real (non-mocked) unit test for the Arcsinh Normalization template.

Validates template I/O behaviour only.
No mocking. Uses real data, real filesystem, and tempfile.
"""

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.arcsinh_normalization_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 4 cells, 2 genes with positive values."""
    rng = np.random.default_rng(42)
    X = rng.integers(1, 100, size=(4, 2)).astype(float)
    obs = pd.DataFrame({"cell_type": ["A", "B", "A", "B"]})
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestArcsinhNormalizationTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the arcsinh normalization template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Co_Factor": "5",
            "Percentile": "99.9",
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "analysis": {"type": "file", "name": "output.pickle"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_arcsinh_normalization_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run arcsinh normalization and verify outputs.

        Validates:
        1. saved_files is a dict with 'analysis' key
        2. Output pickle exists and is non-empty
        3. Output pickle contains an AnnData with 'arcsinh' layer
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("analysis", saved_files)

        pkl_path = Path(saved_files["analysis"])
        self.assertTrue(pkl_path.exists(), f"Pickle not found: {pkl_path}")
        self.assertGreater(pkl_path.stat().st_size, 0)

        with open(pkl_path, "rb") as f:
            result_adata = pickle.load(f)
        self.assertIsInstance(result_adata, ad.AnnData)
        self.assertIn("arcsinh", result_adata.layers)

        # -- save_to_disk=False returns AnnData in memory --------------
        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)
        self.assertIn("arcsinh", mem_adata.layers)


if __name__ == "__main__":
    unittest.main()
