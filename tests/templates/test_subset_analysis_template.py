# tests/templates/test_subset_analysis_template.py
"""
Real (non-mocked) unit test for the Subset Analysis template.

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

from spac.templates.subset_analysis_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 6 cells, 3 cell types for subset filtering."""
    rng = np.random.default_rng(42)
    X = rng.random((6, 2))
    obs = pd.DataFrame({
        "cell_type": ["A", "B", "C", "A", "B", "C"],
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestSubsetAnalysisTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the subset analysis template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation_of_interest": "cell_type",
            "Labels": ["A", "B"],
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "analysis": {"type": "file", "name": "transform_output.pickle"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_subset_analysis_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run subset analysis and verify outputs.

        Validates:
        1. saved_files dict has 'analysis' key
        2. Pickle exists, is non-empty, contains AnnData
        3. Subset has fewer cells than original (only A and B)
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
        # 6 original cells, selecting A and B = 4 cells
        self.assertEqual(result_adata.n_obs, 4)
        self.assertEqual(
            set(result_adata.obs["cell_type"].unique()), {"A", "B"}
        )

        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)
        self.assertEqual(mem_adata.n_obs, 4)


if __name__ == "__main__":
    unittest.main()
