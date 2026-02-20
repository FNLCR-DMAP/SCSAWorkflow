# tests/templates/test_append_annotation_template.py
"""
Real (non-mocked) unit test for the Append Annotation template.

Validates template I/O behaviour only:
  - Expected output files are produced on disk
  - Filenames follow the convention
  - Output artifacts are non-empty

No mocking. Uses real data, real filesystem, and tempfile.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.append_annotation_template import run_from_json


def _make_tiny_dataframe() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame for append annotation testing.

    4 rows, 2 columns -- the smallest dataset that exercises the
    template's column-append code path.
    """
    return pd.DataFrame({
        "cell_type": ["B cell", "T cell", "B cell", "T cell"],
        "marker": [1.0, 2.0, 3.0, 4.0],
    })


class TestAppendAnnotationTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the append annotation template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Annotation_Pair_List": ["batch_id:batch_1", "site:lung"],
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

    def test_append_annotation_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run append annotation template and verify
        output artifacts.

        Validates:
        1. saved_files is a dict with 'dataframe' key
        2. Output CSV exists and is non-empty
        3. New annotation columns are present in the output
        4. In-memory return is a DataFrame with the appended columns
        """
        # -- Act (save_to_disk=True) -----------------------------------
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        # -- Assert: return type ---------------------------------------
        self.assertIsInstance(saved_files, dict)

        # -- Assert: CSV file exists and is non-empty ------------------
        self.assertIn("dataframe", saved_files)
        csv_path = Path(saved_files["dataframe"])
        self.assertTrue(csv_path.exists(), f"CSV not found: {csv_path}")
        self.assertGreater(csv_path.stat().st_size, 0)

        # -- Assert: appended columns present --------------------------
        result_df = pd.read_csv(csv_path)
        self.assertIn("batch_id", result_df.columns)
        self.assertIn("site", result_df.columns)
        self.assertEqual(result_df["batch_id"].unique().tolist(), ["batch_1"])
        self.assertEqual(result_df["site"].unique().tolist(), ["lung"])

        # -- Act (save_to_disk=False) ----------------------------------
        mem_df = run_from_json(
            self.json_file,
            save_to_disk=False,
        )

        # -- Assert: in-memory return is DataFrame ---------------------
        self.assertIsInstance(mem_df, pd.DataFrame)
        self.assertIn("batch_id", mem_df.columns)
        self.assertIn("site", mem_df.columns)


if __name__ == "__main__":
    unittest.main()
