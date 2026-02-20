# tests/templates/test_rename_labels_template.py
"""
Real (non-mocked) unit test for the Rename Labels template.

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

from spac.templates.rename_labels_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 4 cells with cell_type annotation to rename."""
    rng = np.random.default_rng(42)
    X = rng.random((4, 2))
    obs = pd.DataFrame({"cell_type": ["A", "B", "A", "B"]})
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestRenameLabelsTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the rename labels template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        # Create mapping CSV: old_label -> new_label
        mapping_df = pd.DataFrame({
            "old_label": ["A", "B"],
            "new_label": ["Alpha", "Beta"],
        })
        self.mapping_file = os.path.join(self.tmp_dir.name, "mapping.csv")
        mapping_df.to_csv(self.mapping_file, index=False)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation_to_Rename": "cell_type",
            "Mapping_CSV": self.mapping_file,
            "New_Annotation_Name": "cell_type_renamed",
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

    def test_rename_labels_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run rename labels and verify outputs.

        Validates:
        1. saved_files dict has 'analysis' key
        2. Pickle exists, is non-empty, contains AnnData
        3. Renamed annotation column is present with new values
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
        self.assertIn("cell_type_renamed", result_adata.obs.columns)
        self.assertEqual(
            set(result_adata.obs["cell_type_renamed"].unique()),
            {"Alpha", "Beta"},
        )

        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)


if __name__ == "__main__":
    unittest.main()
