# tests/templates/test_umap_transformation_template.py
"""
Real (non-mocked) unit test for the UMAP Transformation template.

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

from spac.templates.umap_transformation_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 20 cells, 5 genes for UMAP."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 5))
    obs = pd.DataFrame({"cell_type": ["A", "B"] * 10})
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(5)])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestUmapTransformationTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the UMAP transformation template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "Number_of_Neighbors": 5,
            "Minimum_Distance_between_Points": 0.1,
            "Target_Dimension_Number": 2,
            "Computational_Metric": "euclidean",
            "Random_State": 0,
            "Transform_Seed": 42,
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

    def test_umap_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run UMAP and verify outputs.

        Validates:
        1. saved_files dict has 'analysis' key
        2. Pickle contains AnnData with 'X_umap' in .obsm
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
        self.assertIn("X_umap", result_adata.obsm)

        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)
        self.assertIn("X_umap", mem_adata.obsm)


if __name__ == "__main__":
    unittest.main()
