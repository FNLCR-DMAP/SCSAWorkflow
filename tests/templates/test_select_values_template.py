# tests/templates/test_select_values_template.py
"""
Real (non-mocked) unit test for the Select Values template.

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

from spac.templates.select_values_template import run_from_json


def _make_tiny_dataframe() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame for value filtering.

    6 rows, 3 cell types -- enough to test include-based selection.
    """
    return pd.DataFrame({
        "cell_type": ["A", "B", "C", "A", "B", "C"],
        "marker": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })


class TestSelectValuesTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the select values template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Annotation_of_Interest": "cell_type",
            "Label_s_of_Interest": ["A", "B"],
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

    def test_select_values_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run select values template and verify
        output artifacts.

        Validates:
        1. saved_files is a dict with 'dataframe' key
        2. Output CSV exists and is non-empty
        3. Only selected values (A, B) remain in the output
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

        # -- Assert: only selected values remain -----------------------
        result_df = pd.read_csv(csv_path)
        self.assertEqual(len(result_df), 4)
        self.assertEqual(
            set(result_df["cell_type"].unique()), {"A", "B"}
        )

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
