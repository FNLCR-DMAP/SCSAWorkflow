# tests/templates/test_calculate_centroid_template.py
"""
Real (non-mocked) unit test for the Calculate Centroid template.

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

from spac.templates.calculate_centroid_template import run_from_json


def _make_tiny_dataframe() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame with bounding-box coordinate columns.

    4 rows -- enough to exercise the centroid calculation.
    """
    return pd.DataFrame({
        "XMin": [0.0, 10.0, 20.0, 30.0],
        "XMax": [10.0, 20.0, 30.0, 40.0],
        "YMin": [0.0, 5.0, 10.0, 15.0],
        "YMax": [4.0, 9.0, 14.0, 19.0],
        "cell_type": ["A", "B", "A", "B"],
    })


class TestCalculateCentroidTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the calculate centroid template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Min_X_Coordinate_Column_Name": "XMin",
            "Max_X_Coordinate_Column_Name": "XMax",
            "Min_Y_Coordinate_Column_Name": "YMin",
            "Max_Y_Coordinate_Column_Name": "YMax",
            "X_Centroid_Name": "XCentroid",
            "Y_Centroid_Name": "YCentroid",
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

    def test_calculate_centroid_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run calculate centroid template and verify
        output artifacts.

        Validates:
        1. saved_files is a dict with 'dataframe' key
        2. Output CSV exists and is non-empty
        3. Centroid columns are present and correctly computed
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

        # -- Assert: centroid columns are present and correct ----------
        result_df = pd.read_csv(csv_path)
        self.assertIn("XCentroid", result_df.columns)
        self.assertIn("YCentroid", result_df.columns)

        # XCentroid = (XMin + XMax) / 2
        expected_x = [5.0, 15.0, 25.0, 35.0]
        self.assertEqual(result_df["XCentroid"].tolist(), expected_x)

        # YCentroid = (YMin + YMax) / 2
        expected_y = [2.0, 7.0, 12.0, 17.0]
        self.assertEqual(result_df["YCentroid"].tolist(), expected_y)

        # -- Act (save_to_disk=False) ----------------------------------
        mem_df = run_from_json(
            self.json_file,
            save_to_disk=False,
        )

        # -- Assert: in-memory return is DataFrame ---------------------
        self.assertIsInstance(mem_df, pd.DataFrame)
        self.assertIn("XCentroid", mem_df.columns)
        self.assertIn("YCentroid", mem_df.columns)


if __name__ == "__main__":
    unittest.main()
