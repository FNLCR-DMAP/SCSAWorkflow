# tests/templates/test_histogram_template.py
"""
Real (non-mocked) unit test for the Histogram template.

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

from spac.templates.histogram_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 4 cells, 2 genes for histogram plotting."""
    rng = np.random.default_rng(42)
    X = rng.integers(1, 10, size=(4, 2)).astype(float)
    obs = pd.DataFrame({"cell_type": ["A", "B", "A", "B"]})
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestHistogramTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the histogram template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "cell_type",
            "Group_by": "cell_type",
            "Together": False,
            "Table_to_Visualize": "Original",
            "Feature_s_to_Plot": ["All"],
            "Figure_Title": "Test Histogram",
            "Legend_Title": "Cell Type",
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 10,
            "Number_of_Bins": 20,
            "Facet": True,
            "Facet_Ncol": 1,
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "dataframe": {"type": "file", "name": "dataframe.csv"},
                "figures": {"type": "directory", "name": "figures_dir"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_histogram_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run histogram and verify outputs.

        Validates:
        1. saved_files dict has 'figures' and 'dataframe' keys
        2. Figures directory contains non-empty PNG(s)
        3. Summary CSV exists and is non-empty
        """
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plot=False,
            output_dir=self.tmp_dir.name,
        )

        self.assertIsInstance(saved_files, dict)
        self.assertIn("figures", saved_files)
        self.assertIn("dataframe", saved_files)

        # Figures
        figure_paths = saved_files["figures"]
        self.assertGreaterEqual(len(figure_paths), 1)
        for fig_path in figure_paths:
            fig_file = Path(fig_path)
            self.assertTrue(fig_file.exists())
            self.assertGreater(fig_file.stat().st_size, 0)

        # CSV
        csv_path = Path(saved_files["dataframe"])
        self.assertTrue(csv_path.exists())
        self.assertGreater(csv_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
