# tests/templates/test_combine_dataframes_template.py
"""
Real (non-mocked) unit test for the Combine DataFrames template.

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

from spac.templates.combine_dataframes_template import run_from_json


def _make_tiny_dataframes():
    """Two minimal DataFrames with the same schema for concatenation."""
    df_a = pd.DataFrame({
        "cell_type": ["B cell", "T cell"],
        "marker": [1.0, 2.0],
    })
    df_b = pd.DataFrame({
        "cell_type": ["NK cell", "Monocyte"],
        "marker": [3.0, 4.0],
    })
    return df_a, df_b


class TestCombineDataFramesTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the combine dataframes template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

        df_a, df_b = _make_tiny_dataframes()
        self.file_a = os.path.join(self.tmp_dir.name, "first.csv")
        self.file_b = os.path.join(self.tmp_dir.name, "second.csv")
        df_a.to_csv(self.file_a, index=False)
        df_b.to_csv(self.file_b, index=False)

        params = {
            "First_Dataframe": self.file_a,
            "Second_Dataframe": self.file_b,
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

    def test_combine_dataframes_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run combine dataframes template and verify
        output artifacts.

        Validates:
        1. saved_files is a dict with 'dataframe' key
        2. Output CSV exists and is non-empty
        3. Combined DataFrame has all rows from both inputs
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

        # -- Assert: combined row count --------------------------------
        result_df = pd.read_csv(csv_path)
        self.assertEqual(len(result_df), 4)
        expected_types = {"B cell", "T cell", "NK cell", "Monocyte"}
        self.assertEqual(set(result_df["cell_type"]), expected_types)

        # -- Act (save_to_disk=False) ----------------------------------
        mem_df = run_from_json(
            self.json_file,
            save_to_disk=False,
        )

        # -- Assert: in-memory return is DataFrame ---------------------
        self.assertIsInstance(mem_df, pd.DataFrame)
        self.assertEqual(len(mem_df), 4)


if __name__ == "__main__":
    unittest.main()
