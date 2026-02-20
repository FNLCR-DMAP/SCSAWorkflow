# tests/templates/test_summarize_dataframe_template.py
"""
Real (non-mocked) unit test for the Summarize DataFrame template.

Validates template I/O behaviour only.
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

from spac.templates.summarize_dataframe_template import run_from_json


def _make_tiny_dataframe() -> pd.DataFrame:
    """Minimal synthetic DataFrame for summarization."""
    return pd.DataFrame({
        "cell_type": ["A", "B", "A", "B", "C", "C"],
        "marker_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "marker_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


class TestSummarizeDataFrameTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the summarize dataframe template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.csv")

        _make_tiny_dataframe().to_csv(self.in_file, index=False)

        params = {
            "Upstream_Dataset": self.in_file,
            "Columns": ["cell_type", "marker_1", "marker_2"],
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "html": {"type": "directory", "name": "html_dir"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_summarize_dataframe_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: summarize dataframe and verify outputs.

        Validates:
        1. saved_files dict has 'html' key
        2. HTML directory contains non-empty file(s)
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("html", saved_files)

        html_paths = saved_files["html"]
        self.assertGreaterEqual(len(html_paths), 1)
        for html_path in html_paths:
            html_file = Path(html_path)
            self.assertTrue(html_file.exists())
            self.assertGreater(html_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
