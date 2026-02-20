# tests/templates/test_posit_it_python_template.py
"""
Real (non-mocked) unit test for the Posit-It Python template.

Validates template I/O behaviour only.
No mocking. Uses real data, real filesystem, and tempfile.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.posit_it_python_template import run_from_json


class TestPostItPythonTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the posit-it python template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

        params = {
            "Label": "Test Note",
            "Label_font_color": "Black",
            "Label_font_size": "40",
            "Label_font_type": "normal",
            "Label_font_family": "Arial",
            "Label_Bold": "False",
            "Background_fill_color": "Yellow1",
            "Background_fill_opacity": "10",
            "Page_width": "6",
            "Page_height": "2",
            "Page_DPI": "72",
            "Output_File": os.path.join(
                self.tmp_dir.name, "postit.png"
            ),
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_posit_it_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run posit-it template and verify outputs.

        Validates:
        1. save_results=True returns a dict
        2. Output PNG file exists and is non-empty
        3. save_results=False returns a matplotlib Figure
        """
        saved_files = run_from_json(
            self.json_file,
            save_results=True,
            show_plot=False,
        )

        self.assertIsInstance(saved_files, dict)

        # The template saves to Output_File path
        output_path = Path(os.path.join(
            self.tmp_dir.name, "postit.png"
        ))
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

        import matplotlib.figure
        fig = run_from_json(
            self.json_file,
            save_results=False,
            show_plot=False,
        )
        self.assertIsInstance(fig, matplotlib.figure.Figure)


if __name__ == "__main__":
    unittest.main()
