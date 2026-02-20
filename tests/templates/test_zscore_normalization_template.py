# tests/templates/test_zscore_normalization_template.py
"""
Real (non-mocked) unit test for the Z-Score Normalization template.

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

from spac.templates.z_score_normalization_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 4 cells, 2 genes for z-score normalization."""
    rng = np.random.default_rng(42)
    X = rng.integers(1, 100, size=(4, 2)).astype(float)
    obs = pd.DataFrame({"cell_type": ["A", "B", "A", "B"]})
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestZScoreNormalizationTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the z-score normalization template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "Output_Table_Name": "zscore",
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

    def test_zscore_normalization_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run z-score normalization and verify outputs.

        Validates:
        1. saved_files dict has 'analysis' key
        2. Pickle exists, is non-empty, contains AnnData
        3. Z-score layer is present in the AnnData
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("analysis", saved_files)

        pkl_path = Path(saved_files["analysis"])
        self.assertTrue(pkl_path.exists())
        self.assertGreater(pkl_path.stat().st_size, 0)

        with open(pkl_path, "rb") as f:
            result_adata = pickle.load(f)
        self.assertIsInstance(result_adata, ad.AnnData)
        # z-score normalization creates a 'zscore' layer
        self.assertIn("zscore", result_adata.layers)

        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)
        self.assertIn("zscore", mem_adata.layers)


if __name__ == "__main__":
    unittest.main()
