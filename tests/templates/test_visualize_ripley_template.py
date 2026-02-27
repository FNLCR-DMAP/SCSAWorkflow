# tests/templates/test_visualize_ripley_template.py
"""
Real (non-mocked) unit test for the Visualize Ripley L template.

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

from spac.templates.visualize_ripley_l_template import run_from_json
from spac.templates.ripley_l_calculation_template import (
    run_from_json as run_ripley,
)


def _make_adata_with_ripley() -> ad.AnnData:
    """Create AnnData with pre-computed Ripley L results in .uns."""
    rng = np.random.default_rng(42)
    X = rng.random((20, 2))
    obs = pd.DataFrame({"cell_type": (["A"] * 10) + (["B"] * 10)})
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    spatial = rng.random((20, 2)) * 100
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial

    # Run actual Ripley L to populate .uns
    import tempfile as tf
    with tf.TemporaryDirectory() as td:
        pkl_in = os.path.join(td, "in.pickle")
        with open(pkl_in, "wb") as f:
            pickle.dump(adata, f)
        ripley_params = {
            "Upstream_Analysis": pkl_in,
            "Radii": [5, 10, 20],
            "Annotation": "cell_type",
            "Center_Phenotype": "A",
            "Neighbor_Phenotype": "B",
            "Number_of_Simulations": 5,
            "Seed": 42,
            "Spatial_Key": "spatial",
            "Edge_Correction": True,
        }
        json_path = os.path.join(td, "p.json")
        with open(json_path, "w") as f:
            json.dump(ripley_params, f)
        adata = run_ripley(json_path, save_to_disk=False)
    return adata


class TestVisualizeRipleyTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the visualize Ripley L template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_adata_with_ripley(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Radii": [5, 10, 20],
            "Annotation": "cell_type",
            "Center_Phenotype": "A",
            "Neighbor_Phenotype": "B",
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 10,
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "figures": {"type": "directory", "name": "figures_dir"},
                "dataframe": {"type": "file", "name": "dataframe.csv"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_visualize_ripley_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: visualize Ripley L and verify outputs.

        Validates:
        1. saved_files dict has output keys
        2. Output files exist and are non-empty
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plot=False,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        # Check that at least some output was produced
        self.assertGreater(len(saved_files), 0)

        for key, value in saved_files.items():
            if isinstance(value, list):
                for p in value:
                    pf = Path(p)
                    self.assertTrue(pf.exists())
                    self.assertGreater(pf.stat().st_size, 0)
            elif isinstance(value, str):
                pf = Path(value)
                self.assertTrue(pf.exists())
                self.assertGreater(pf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
