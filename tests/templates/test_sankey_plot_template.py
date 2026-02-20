# tests/templates/test_sankey_plot_template.py
"""
Real (non-mocked) unit test for the Sankey Plot template.

Validates template I/O behaviour only.
No mocking. Uses real data, real filesystem, and tempfile.
"""

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.sankey_plot_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 8 cells with two annotation columns for Sankey."""
    rng = np.random.default_rng(42)
    X = rng.random((8, 2))
    obs = pd.DataFrame({
        "cell_type": ["A", "A", "B", "B", "A", "A", "B", "B"],
        "cluster": ["1", "2", "1", "2", "1", "2", "1", "2"],
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestSankeyPlotTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the sankey plot template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Source_Annotation_Name": "cell_type",
            "Target_Annotation_Name": "cluster",
            "Figure_Width_inch": 6,
            "Figure_Height_inch": 6,
            "Font_Size": 10,
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "figures": {"type": "directory", "name": "figures_dir"},
                "html": {"type": "directory", "name": "html_dir"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_sankey_plot_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run sankey plot and verify outputs.

        Validates:
        1. saved_files dict has 'figures' and/or 'html' keys
        2. Output files exist and are non-empty
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        # Sankey produces HTML (interactive) and/or static figures
        has_output = "html" in saved_files or "figures" in saved_files
        self.assertTrue(has_output, "No output files found")

        for key in ["html", "figures"]:
            if key in saved_files:
                paths = saved_files[key]
                self.assertGreaterEqual(len(paths), 1)
                for p in paths:
                    pf = Path(p)
                    self.assertTrue(pf.exists())
                    self.assertGreater(pf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
