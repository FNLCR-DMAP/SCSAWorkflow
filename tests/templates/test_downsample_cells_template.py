# tests/templates/test_downsample_cells_template.py
"""
Real (non-mocked) unit test for the Downsample Cells template.

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

from spac.templates.downsample_cells_template import run_from_json


def _make_tiny_dataframe() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame for downsampling.

    8 rows, 2 groups of 4 -- enough to exercise group-based downsampling.
    """
    return pd.DataFrame({
        "cell_type": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "marker": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })


class TestDownsampleCellsTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the downsample cells template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Annotations_List": ["cell_type"],
            "Number_of_Samples": 2,
            "Stratify_Option": False,
            "Random_Selection": False,
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

    def test_downsample_cells_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run downsample cells template and verify
        output artifacts.

        Validates:
        1. saved_files is a dict with 'dataframe' key
        2. Output CSV exists and is non-empty
        3. Row count is reduced (2 per group = 4 total from 8)
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

        # -- Assert: downsampled row count -----------------------------
        result_df = pd.read_csv(csv_path)
        # 2 samples per group * 2 groups = 4 rows
        self.assertEqual(len(result_df), 4)
        # Both groups should still be present
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
