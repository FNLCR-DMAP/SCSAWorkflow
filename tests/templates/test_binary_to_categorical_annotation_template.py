# tests/templates/test_binary_to_categorical_annotation_template.py
"""
Real (non-mocked) unit test for the Binary to Categorical Annotation template.

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

from spac.templates.binary_to_categorical_annotation_template import (
    run_from_json,
)


def _make_tiny_dataframe() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame with binary one-hot columns.

    4 rows -- each row has exactly one 1 across the binary columns.
    """
    return pd.DataFrame({
        "B_cell": [1, 0, 0, 0],
        "T_cell": [0, 1, 0, 1],
        "NK_cell": [0, 0, 1, 0],
        "marker": [1.5, 2.5, 3.5, 4.5],
    })


class TestBinaryToCategoricalAnnotationTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the binary-to-categorical template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Binary_Annotation_Columns": ["B_cell", "T_cell", "NK_cell"],
            "New_Annotation_Name": "cell_labels",
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

    def test_bin2cat_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run binary-to-categorical template and verify
        output artifacts.

        Validates:
        1. saved_files is a dict with 'dataframe' key
        2. Output CSV exists and is non-empty
        3. New categorical column 'cell_labels' is present
        4. Categorical values match the original binary column names
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

        # -- Assert: categorical column present with expected values ---
        result_df = pd.read_csv(csv_path)
        self.assertIn("cell_labels", result_df.columns)
        expected_labels = {"B_cell", "T_cell", "NK_cell"}
        actual_labels = set(result_df["cell_labels"].dropna().unique())
        self.assertEqual(actual_labels, expected_labels)

        # -- Act (save_to_disk=False) ----------------------------------
        mem_df = run_from_json(
            self.json_file,
            save_to_disk=False,
        )

        # -- Assert: in-memory return is DataFrame ---------------------
        self.assertIsInstance(mem_df, pd.DataFrame)
        self.assertIn("cell_labels", mem_df.columns)


if __name__ == "__main__":
    unittest.main()
