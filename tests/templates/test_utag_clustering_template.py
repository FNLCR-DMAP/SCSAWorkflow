# tests/templates/test_utag_clustering_template.py
"""
Real (non-mocked) unit test for the UTAG Clustering template.

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

from spac.templates.utag_clustering_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 30 cells with spatial coords for UTAG clustering."""
    rng = np.random.default_rng(42)
    X = rng.random((30, 3))
    obs = pd.DataFrame({"cell_type": ["A", "B", "C"] * 10})
    var = pd.DataFrame(index=["Gene_0", "Gene_1", "Gene_2"])
    spatial = rng.random((30, 2)) * 100
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    return adata


class TestUTAGClusteringTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the UTAG clustering template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Table_to_Process": "Original",
            "Features": ["All"],
            "Slide_Annotation": "None",
            "Distance_Threshold": 20.0,
            "K_Nearest_Neighbors": 5,
            "Resolution_Parameter": 1,
            "PCA_Components": "None",
            "Random_Seed": 42,
            "N_Jobs": 1,
            "Leiden_Iterations": 3,
            "Parellel_Processes": False,
            "Output_Annotation_Name": "UTAG",
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

    def test_utag_clustering_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run UTAG clustering and verify outputs.

        Validates:
        1. saved_files dict has 'analysis' key
        2. Pickle contains AnnData with UTAG obs column
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
        self.assertIn("UTAG", result_adata.obs.columns)

        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)
        self.assertIn("UTAG", mem_adata.obs.columns)


if __name__ == "__main__":
    unittest.main()
