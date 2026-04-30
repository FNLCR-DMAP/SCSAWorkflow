# tests/templates/test_setup_analysis_template.py
"""
Real (non-mocked) unit test for the Setup Analysis template.

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

from spac.templates.setup_analysis_template import run_from_json


def _make_tiny_dataframe() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame simulating raw cell data.

    4 cells with spatial coordinates, features, and an annotation column.
    """
    return pd.DataFrame({
        "Gene_0": [1.0, 2.0, 3.0, 4.0],
        "Gene_1": [5.0, 6.0, 7.0, 8.0],
        "X_coord": [10.0, 20.0, 30.0, 40.0],
        "Y_coord": [11.0, 21.0, 31.0, 41.0],
        "cell_type": ["A", "B", "A", "B"],
    })


class TestSetupAnalysisTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the setup analysis template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Features_to_Analyze": ["Gene_0", "Gene_1"],
            "Annotation_s_": ["cell_type"],
            "X_Coordinate_Column": "X_coord",
            "Y_Coordinate_Column": "Y_coord",
            "Output_File": "output.pickle",
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

    def test_setup_analysis_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run setup analysis and verify outputs.

        Validates:
        1. saved_files dict has 'analysis' key
        2. Pickle exists, is non-empty, contains AnnData
        3. AnnData has correct features, obs, and spatial coords
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
        self.assertEqual(result_adata.n_obs, 4)
        self.assertIn("cell_type", result_adata.obs.columns)
        self.assertIn("spatial", result_adata.obsm)

        mem_adata = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_adata, ad.AnnData)


if __name__ == "__main__":
    unittest.main()
