# tests/templates/test_load_csv_files_with_config.py
"""
Real (non-mocked) unit test for the Load CSV Files template.

Snowball test -- validates template I/O behaviour only.
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

from spac.templates.load_csv_files_template import run_from_json


class TestLoadCSVFilesWithConfig(unittest.TestCase):
    """Real (non-mocked) tests for the load CSV files template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

        # Create two simple CSV data files
        df1 = pd.DataFrame({
            "Feature_A": [1.0, 2.0],
            "Feature_B": [3.0, 4.0],
            "ID": ["cell_1", "cell_2"],
        })
        df2 = pd.DataFrame({
            "Feature_A": [5.0, 6.0],
            "Feature_B": [7.0, 8.0],
            "ID": ["cell_3", "cell_4"],
        })

        self.csv1 = os.path.join(self.tmp_dir.name, "data1.csv")
        self.csv2 = os.path.join(self.tmp_dir.name, "data2.csv")
        df1.to_csv(self.csv1, index=False)
        df2.to_csv(self.csv2, index=False)

        # Create configuration CSV
        config_df = pd.DataFrame({
            "column_name": ["Feature_A", "Feature_B", "ID"],
            "column_type": ["feature", "feature", "string"],
        })
        self.config_file = os.path.join(self.tmp_dir.name, "config.csv")
        config_df.to_csv(self.config_file, index=False)

        params = {
            "CSV_Files": [self.csv1, self.csv2],
            "CSV_Files_Configuration": self.config_file,
            "String_Columns": ["ID"],
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

    def test_load_csv_files_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: load CSV files with config and verify.

        Validates:
        1. saved_files dict has 'dataframe' key
        2. CSV exists and is non-empty
        3. Combined data has rows from both input files
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
        self.assertEqual(len(result_df), 4)

        mem_df = run_from_json(self.json_file, save_to_disk=False)
        self.assertIsInstance(mem_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
