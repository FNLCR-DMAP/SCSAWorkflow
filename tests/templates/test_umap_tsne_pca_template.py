# tests/templates/test_umap_tsne_pca_template.py
"""
Real (non-mocked) unit test for the UMAP/tSNE/PCA Visualization template.

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

from spac.templates.umap_tsne_pca_visualization_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData with pre-computed UMAP embedding for visualization."""
    rng = np.random.default_rng(42)
    X = rng.random((8, 2))
    obs = pd.DataFrame({"cell_type": ["A", "B"] * 4})
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = rng.random((8, 2)) * 10
    return adata


class TestUmapTsnePcaTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the UMAP/tSNE/PCA visualization."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Dimensionality_Reduction_Method": "UMAP",
            "Color_By": "Annotation",
            "Annotation": "cell_type",
            "Feature": "None",
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 10,
            "Spot_Size": 50,
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

    def test_umap_tsne_pca_visualization_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run dim reduction visualization and verify.

        Validates:
        1. saved_files dict has 'figures' key
        2. Figures directory contains non-empty PNG(s)
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plot=False,
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
