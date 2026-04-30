# tests/templates/test_visualize_nearest_neighbor_template.py
"""
Real (non-mocked) unit test for the Visualize Nearest Neighbor template.

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

from spac.templates.visualize_nearest_neighbor_template import run_from_json
from spac.templates.nearest_neighbor_calculation_template import (
    run_from_json as run_nn,
)


def _make_adata_with_nn() -> ad.AnnData:
    """Create AnnData with pre-computed nearest neighbor results."""
    rng = np.random.default_rng(42)
    X = rng.random((12, 2))
    obs = pd.DataFrame({
        "cell_type": ["A", "B", "C"] * 4,
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    spatial = rng.random((12, 2)) * 100
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = spatial

    # Run actual nearest neighbor to populate .obsm
    import tempfile as tf
    with tf.TemporaryDirectory() as td:
        pkl_in = os.path.join(td, "in.pickle")
        with open(pkl_in, "wb") as f:
            pickle.dump(adata, f)
        nn_params = {
            "Upstream_Analysis": pkl_in,
            "Annotation": "cell_type",
            "ImageID": "None",
        }
        json_path = os.path.join(td, "p.json")
        with open(json_path, "w") as f:
            json.dump(nn_params, f)
        adata = run_nn(json_path, save_to_disk=False)
    return adata


class TestVisualizeNearestNeighborTemplate(unittest.TestCase):
    """Real (non-mocked) tests for visualize nearest neighbor template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_adata_with_nn(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "cell_type",
            "Source_Anchor_Cell_Label": "A",
            "Nearest_Neighbor_Associated_Table": "spatial_distance",
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 10,
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "figures": {"type": "directory", "name": "figures"},
                "dataframe": {"type": "file", "name": "dataframe.csv"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_visualize_nn_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: visualize nearest neighbors and verify.

        Validates:
        1. saved_files dict has 'figures' and/or 'dataframe' keys
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

        if "figures" in saved_files:
            figure_paths = saved_files["figures"]
            self.assertGreaterEqual(len(figure_paths), 1)
            for fig_path in figure_paths:
                fig_file = Path(fig_path)
                self.assertTrue(fig_file.exists())
                self.assertGreater(fig_file.stat().st_size, 0)

        if "dataframe" in saved_files:
            csv_path = Path(saved_files["dataframe"])
            self.assertTrue(csv_path.exists())
            self.assertGreater(csv_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
