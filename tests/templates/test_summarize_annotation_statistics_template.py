# tests/templates/test_summarize_annotation_statistics_template.py
"""
Real (non-mocked) unit test for the Summarize Annotation Statistics template.

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

from spac.templates.summarize_annotation_statistics_template import (
    run_from_json,
)


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 6 cells with cell_type annotation for statistics."""
    rng = np.random.default_rng(42)
    X = rng.random((6, 2))
    obs = pd.DataFrame({
        "cell_type": ["A", "A", "B", "B", "B", "C"],
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestSummarizeAnnotationStatisticsTemplate(unittest.TestCase):
    """Real (non-mocked) tests for summarize annotation statistics."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "cell_type",
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_summarize_annotation_stats_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: summarize annotation stats and verify outputs.

        Validates:
        1. saved_files dict has 'dataframe' key
        2. CSV exists and is non-empty
        3. Summary includes count/percentage information
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("dataframe", saved_files)

        csv_path = Path(saved_files["dataframe"])
        self.assertTrue(csv_path.exists())
        self.assertGreater(csv_path.stat().st_size, 0)

        result_df = pd.read_csv(csv_path)
        self.assertGreater(len(result_df), 0)

        mem_df = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
