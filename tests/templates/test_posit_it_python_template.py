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
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "figures": {"type": "directory", "name": "figures_dir"},
            },
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
        1. save_to_disk=True returns a dict with 'figures' key
        2. Figures directory contains a non-empty PNG
        3. save_to_disk=False returns a matplotlib Figure with correct text
        """
        # -- Act (save_to_disk=True): write outputs to disk ------------
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plot=False,
            output_dir=self.tmp_dir.name,
        )

        # -- Act (save_to_disk=False): get figure in memory ------------
        fig = run_from_json(
            self.json_file,
            save_to_disk=False,
            show_plot=False,
        )

        # -- Assert: return type ---------------------------------------
        self.assertIsInstance(
            saved_files, dict,
            f"Expected dict from run_from_json, got {type(saved_files)}"
        )

        # -- Assert: figures directory contains at least one PNG -------
        self.assertIn("figures", saved_files,
                       "Missing 'figures' key in saved_files")
        figure_paths = saved_files["figures"]
        self.assertGreaterEqual(
            len(figure_paths), 1, "No figure files were saved"
        )

        for fig_path in figure_paths:
            fig_file = Path(fig_path)
            self.assertTrue(
                fig_file.exists(), f"Figure not found: {fig_path}"
            )
            self.assertGreater(
                fig_file.stat().st_size, 0,
                f"Figure file is empty: {fig_path}"
            )
            self.assertEqual(
                fig_file.suffix, ".png",
                f"Expected .png extension, got {fig_file.suffix}"
            )

        # -- Assert: in-memory figure is valid -------------------------
        import matplotlib.figure
        self.assertIsInstance(
            fig, matplotlib.figure.Figure,
            f"Expected matplotlib Figure, got {type(fig)}"
        )

        # The figure text at (0.5, 0.5) should contain "Test Note"
        text_artists = fig.texts
        self.assertGreaterEqual(
            len(text_artists), 1,
            "Figure has no text artists"
        )
        # First text artist is the label placed by fig.text(0.5, 0.5, ...)
        self.assertEqual(
            text_artists[0].get_text(), "Test Note",
            f"Expected figure text 'Test Note', "
            f"got '{text_artists[0].get_text()}'"
        )


if __name__ == "__main__":
    unittest.main()
