# tests/templates/test_boxplot_template.py
"""
Real (non-mocked) unit test for the Boxplot template.

Snowball seed test — validates template I/O behaviour only:
  • Expected output files are produced on disk
  • Filenames follow the convention
  • Output artifacts are non-empty

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
matplotlib.use("Agg")  # Headless backend for CI

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.boxplot_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """
    Minimal synthetic AnnData for boxplot template testing.

    4 cells, 2 genes, 2 cell types — the smallest dataset that exercises
    the template's grouping, plotting, and summary-stats code paths.
    """
    rng = np.random.default_rng(42)

    # 4 cells × 2 genes — small enough to reason about,
    # large enough for describe() to return meaningful stats
    n_cells, n_genes = 4, 2
    X = rng.integers(1, 10, size=(n_cells, n_genes)).astype(float)

    obs = pd.DataFrame(
        {"cell_type": ["B cell", "T cell", "B cell", "T cell"]},
    )
    var = pd.DataFrame(index=["Gene_0", "Gene_1"])

    return ad.AnnData(X=X, obs=obs, var=var)


class TestBoxplotTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the boxplot template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        # Save minimal real data as pickle (simulates upstream analysis)
        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        # Write a JSON params file — the actual input the template receives
        # in production (from Galaxy / Code Ocean)
        params = {
            "Upstream_Analysis": self.in_file,
            "Primary_Annotation": "cell_type",
            "Secondary_Annotation": "None",
            "Table_to_Visualize": "Original",
            "Feature_s_to_Plot": ["All"],
            "Value_Axis_Log_Scale": False,
            "Figure_Title": "Test BoxPlot",
            "Horizontal_Plot": False,
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,          # low DPI for fast save
            "Font_Size": 10,
            "Keep_Outliers": True,
            "Output_Directory": self.tmp_dir.name,
            "outputs": {
                "figures": {"type": "directory", "name": "figures"},
                "dataframe": {"type": "file", "name": "output.csv"},
            },
        }

        self.json_file = os.path.join(self.tmp_dir.name, "params.json")
        with open(self.json_file, "w") as f:
            json.dump(params, f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_boxplot_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run boxplot template and verify output
        artifacts.

        Validates:
        1. saved_files is a dict with 'figures' and 'dataframe' keys
        2. A figures directory is created containing a non-empty PNG
        3. The figure title matches the "Figure_Title" param
        4. A summary CSV is created with the exact describe() rows
        """
        # -- Act (save_to_disk=True): write outputs to disk ------------
        saved_files = run_from_json(
            self.json_file,
            save_to_disk=True,
            show_plot=False,           # no GUI in CI
            output_dir=self.tmp_dir.name,
        )

        # -- Act (save_to_disk=False): get figure + df in memory -------
        fig, summary_df_mem = run_from_json(
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
            # Template saves matplotlib figures as .png
            self.assertEqual(
                fig_file.suffix, ".png",
                f"Expected .png extension, got {fig_file.suffix}"
            )

        # -- Assert: figure has the correct title ----------------------
        # The template calls ax.set_title(figure_title), so the axes
        # title must match the "Figure_Title" parameter we passed in.
        axes_title = fig.axes[0].get_title()
        self.assertEqual(
            axes_title, "Test BoxPlot",
            f"Expected figure title 'Test BoxPlot', got '{axes_title}'"
        )

        # -- Assert: summary CSV exists and is non-empty ---------------
        self.assertIn("dataframe", saved_files,
                       "Missing 'dataframe' key in saved_files")
        csv_path = Path(saved_files["dataframe"])
        self.assertTrue(
            csv_path.exists(), f"Summary CSV not found: {csv_path}"
        )
        self.assertGreater(
            csv_path.stat().st_size, 0,
            f"Summary CSV is empty: {csv_path}"
        )

        # -- Assert: CSV has the exact describe() stat rows ------------
        # The template calls df.describe().reset_index() which produces
        # exactly these 8 rows in this order.
        summary_df = pd.read_csv(csv_path)
        expected_stats = [
            "count", "mean", "std", "min",
            "25%", "50%", "75%", "max",
        ]

        # First column after reset_index() is called "index"
        actual_stats = summary_df["index"].tolist()
        self.assertEqual(
            actual_stats, expected_stats,
            f"Summary CSV stat rows don't match.\n"
            f"  Expected: {expected_stats}\n"
            f"  Actual:   {actual_stats}"
        )


if __name__ == "__main__":
    unittest.main()