# tests/templates/test_spatial_plot_template.py
"""
Real (non-mocked) unit test for the Spatial Plot template.

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

from spac.templates.spatial_plot_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 8 cells with spatial coords for plotting."""
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


class TestSpatialPlotTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the spatial plot template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Color_By": "Annotation",
            "Annotation_to_Highlight": "cell_type",
            "Feature_to_Highlight": "None",
            "Stratify": False,
            "Stratify_By": [],
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 10,
            "Dot_Size": 50,
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

    def test_spatial_plot_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run spatial plot and verify outputs.

        Validates:
        1. saved_files dict has 'figures' key
        2. Figures directory contains non-empty PNG(s)
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plots=False,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("figures", saved_files)

        figure_paths = saved_files["figures"]
        self.assertGreaterEqual(len(figure_paths), 1)
        for fig_path in figure_paths:
            fig_file = Path(fig_path)
            self.assertTrue(fig_file.exists())
            self.assertGreater(fig_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
