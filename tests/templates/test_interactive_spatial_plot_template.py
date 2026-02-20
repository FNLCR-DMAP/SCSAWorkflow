# tests/templates/test_interactive_spatial_plot_template.py
"""
Real (non-mocked) unit test for the Interactive Spatial Plot template.

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

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.interactive_spatial_plot_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 8 cells with spatial coords."""
    rng = np.random.default_rng(42)
    X = rng.random((8, 2))
    obs = pd.DataFrame({
        "cell_type": ["A", "B", "A", "B", "A", "B", "A", "B"],
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    spatial = rng.random((8, 2)) * 100
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    return adata


class TestInteractiveSpatialPlotTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the interactive spatial plot template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Color_By": "Annotation",
            "Annotation": "cell_type",
            "Feature": "None",
            "Spot_Size": 5,
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

    def test_interactive_spatial_plot_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run interactive spatial plot and verify outputs.

        Validates:
        1. saved_files dict has 'html' key
        2. HTML directory contains non-empty file(s)
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plot=False,
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
