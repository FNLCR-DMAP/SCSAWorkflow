# tests/templates/test_spatial_interaction_template.py
"""
Real (non-mocked) unit test for the Spatial Interaction template.

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

from spac.templates.spatial_interaction_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 20 cells with spatial coords for interaction."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 2))
    obs = pd.DataFrame({
        "cell_type": (["A"] * 10) + (["B"] * 10),
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    spatial = rng.random((20, 2)) * 100
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    return adata


class TestSpatialInteractionTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the spatial interaction template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "cell_type",
            "Spatial_Analysis_Method": "Neighborhood Enrichment",
            "Stratify_By": ["None"],
            "K_Nearest_Neighbors": 6,
            "Seed": 42,
            "Coordinate_Type": "None",
            "Radius": "None",
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 10,
            "Color_Bar_Range": "Automatic",
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "figures": {"type": "directory", "name": "figures"},
                "dataframes": {"type": "directory", "name": "matrices"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_spatial_interaction_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run spatial interaction and verify outputs.

        Validates:
        1. saved_files dict has 'figures' and/or 'dataframes' keys
        2. Output files exist and are non-empty
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plot=False,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertGreater(len(saved_files), 0)

        for key in ["figures", "dataframes"]:
            if key in saved_files:
                paths = saved_files[key]
                self.assertGreaterEqual(len(paths), 1)
                for p in paths:
                    pf = Path(p)
                    self.assertTrue(pf.exists())
                    self.assertGreater(pf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
