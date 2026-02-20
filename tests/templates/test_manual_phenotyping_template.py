#!/usr/bin/env python3
# tests/templates/test_manual_phenotyping_template.py
"""
Real (non-mocked) unit test for the Manual Phenotyping template.

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

from spac.templates.manual_phenotyping_template import run_from_json


def _make_tiny_dataframe() -> pd.DataFrame:
    """
    Minimal synthetic DataFrame with binary phenotype marker columns.

    4 rows -- each row has one positive marker matching a phenotype rule.
    """
    return pd.DataFrame({
        "cd4": [1, 0, 0, 1],
        "cd8": [0, 1, 0, 0],
        "cd20": [0, 0, 1, 0],
        "marker_intensity": [1.5, 2.5, 3.5, 4.5],
    })


def _make_phenotype_rules() -> pd.DataFrame:
    """
    Phenotype rule table: maps binary codes to phenotype names.

    Each row uses a '+' or '-' code referencing column names.
    """
    return pd.DataFrame({
        "phenotype_name": ["T_helper", "Cytotoxic_T", "B_cell"],
        "phenotype_code": ["cd4+cd8-", "cd4-cd8+", "cd20+"],
    })


class TestManualPhenotypingTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the manual phenotyping template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")
        self.rules_file = os.path.join(self.tmp_dir.name, "phenotypes.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)
        _make_phenotype_rules().to_csv(self.rules_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Phenotypes_Code": self.rules_file,
            "Classification_Column_Prefix": "",
            "Classification_Column_Suffix": "",
            "Allow_Multiple_Phenotypes": True,
            "Manual_Annotation_Name": "manual_phenotype",
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

    def test_manual_phenotyping_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run manual phenotyping template and verify
        output artifacts.

        Validates:
        1. saved_files is a dict with 'dataframe' key
        2. Output CSV exists and is non-empty
        3. Phenotype annotation column is present in output
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

        # -- Assert: phenotype column present --------------------------
        result_df = pd.read_csv(csv_path)
        self.assertIn("manual_phenotype", result_df.columns)
        # At least some rows should have assigned phenotypes
        non_null = result_df["manual_phenotype"].dropna()
        self.assertGreater(len(non_null), 0)

        # -- Act (save_to_disk=False) ----------------------------------
        mem_df = run_from_json(
            self.json_file,
            save_to_disk=False,
        )

        # -- Assert: in-memory return is DataFrame ---------------------
        self.assertIsInstance(mem_df, pd.DataFrame)
        self.assertIn("manual_phenotype", mem_df.columns)


if __name__ == "__main__":
    unittest.main()
