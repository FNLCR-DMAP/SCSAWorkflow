# tests/templates/test_neighborhood_profile_template.py
"""
Real (non-mocked) unit test for the Neighborhood Profile template.

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

from spac.templates.neighborhood_profile_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 20 cells with spatial coords and annotation."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 2))
    obs = pd.DataFrame({
        "cell_type": (["A"] * 10) + (["B"] * 10),
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    spatial = rng.random((20, 2)) * 100
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    return adata


class TestNeighborhoodProfileTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the neighborhood profile template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation_of_interest": "cell_type",
            "Bins": [10, 25, 50],
            "Anchor_Neighbor_List": ["A;B"],
            "Stratify_By": "None",
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "dataframe": {"type": "directory", "name": "dataframe_dir"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_neighborhood_profile_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: compute neighborhood profiles and verify.

        Validates:
        1. saved_files dict has 'dataframe' key
        2. Output directory contains CSV file(s)
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("dataframe", saved_files)

        csv_paths = saved_files["dataframe"]
        self.assertGreaterEqual(len(csv_paths), 1)
        for csv_path in csv_paths:
            csv_file = Path(csv_path)
            self.assertTrue(csv_file.exists())
            self.assertGreater(csv_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
