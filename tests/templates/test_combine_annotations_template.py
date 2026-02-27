# tests/templates/test_combine_annotations_template.py
"""
Real (non-mocked) unit test for the Combine Annotations template.

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

from spac.templates.combine_annotations_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 4 cells with two annotation columns to combine."""
    rng = np.random.default_rng(42)
    X = rng.random((4, 2))
    obs = pd.DataFrame({
        "tissue": ["lung", "liver", "lung", "liver"],
        "cell_type": ["B cell", "T cell", "T cell", "B cell"],
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestCombineAnnotationsTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the combine annotations template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotations_Names": ["tissue", "cell_type"],
            "Separator": "_",
            "New_Annotation_Name": "combined",
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"},
                "analysis": {"type": "file", "name": "output.pickle"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_combine_annotations_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run combine annotations and verify outputs.

        Validates:
        1. saved_files dict has 'analysis' and 'dataframe' keys
        2. Pickle contains AnnData with 'combined' obs column
        3. CSV exists and is non-empty
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("analysis", saved_files)
        self.assertIn("dataframe", saved_files)

        # -- Pickle output --
        pkl_path = Path(saved_files["analysis"])
        self.assertTrue(pkl_path.exists())
        self.assertGreater(pkl_path.stat().st_size, 0)

        with open(pkl_path, "rb") as f:
            result_adata = pickle.load(f)
        self.assertIsInstance(result_adata, ad.AnnData)
        self.assertIn("combined", result_adata.obs.columns)

        # -- CSV output --
        csv_path = Path(saved_files["dataframe"])
        self.assertTrue(csv_path.exists())
        self.assertGreater(csv_path.stat().st_size, 0)

        # -- In-memory --
        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)
        self.assertIn("combined", mem_adata.obs.columns)


if __name__ == "__main__":
    unittest.main()
