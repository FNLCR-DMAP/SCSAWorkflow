# tests/templates/test_hierarchical_heatmap_template.py
"""
Real (non-mocked) unit test for the Hierarchical Heatmap template.

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

from spac.templates.hierarchical_heatmap_template import run_from_json


def _make_tiny_adata() -> ad.AnnData:
    """Minimal AnnData: 8 cells, 3 genes, 2 groups for heatmap."""
    rng = np.random.default_rng(42)
    X = rng.integers(1, 20, size=(8, 3)).astype(float)
    obs = pd.DataFrame({
        "cell_type": ["A", "A", "B", "B", "A", "A", "B", "B"],
    })
    var = pd.DataFrame(index=["Gene_0", "Gene_1", "Gene_2"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestHierarchicalHeatmapTemplate(unittest.TestCase):
    """Real (non-mocked) tests for the hierarchical heatmap template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")

        with open(self.in_file, "wb") as f:
            pickle.dump(_make_tiny_adata(), f)

        params = {
            "Upstream_Analysis": self.in_file,
            "Annotation": "cell_type",
            "Table_to_Visualize": "Original",
            "Feature_s_": ["All"],
            "Standard_Scale_": "None",
            "Z_Score": "None",
            "Feature_Dendrogram": True,
            "Annotation_Dendrogram": True,
            "Figure_Title": "Test Hierarchical Heatmap",
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 8,
            "Matrix_Plot_Ratio": 0.8,
            "Swap_Axes": False,
            "Rotate_Label_": False,
            "Horizontal_Dendrogram_Display_Ratio": 0.2,
            "Vertical_Dendrogram_Display_Ratio": 0.2,
            "Value_Min": "None",
            "Value_Max": "None",
            "Color_Map": "seismic",
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

    def test_hierarchical_heatmap_produces_expected_outputs(self) -> None:
        """
        End-to-end I/O test: run hierarchical heatmap and verify outputs.

        Validates:
        1. saved_files dict has 'figures' and 'dataframe' keys
        2. Figures directory contains non-empty PNG(s)
        3. Summary CSV exists and is non-empty
        4. Figure title matches the parameter
        """
        # -- Act (save_results_flag=True): write outputs to disk -------
        saved_files = run_from_json(
            self.json_file,
            save_results_flag=True,
            show_plot=False,
            output_dir=self.tmp_dir.name,
        )

        # -- Act (save_results_flag=False): get objects in memory ------
        clustergrid, mean_intensity_df = run_from_json(
            self.json_file,
            save_results_flag=False,
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

        # -- Assert: figure has the correct title ----------------------
        axes_title = clustergrid.ax_heatmap.get_title()
        self.assertEqual(
            axes_title, "Test Hierarchical Heatmap",
            f"Expected 'Test Hierarchical Heatmap', got '{axes_title}'"
        )

        # -- Assert: in-memory mean_intensity_df is a DataFrame --------
        self.assertIsInstance(
            mean_intensity_df, pd.DataFrame,
            f"Expected DataFrame, got {type(mean_intensity_df)}"
        )
        self.assertIn(
            "cell_type", mean_intensity_df.columns,
            "Annotation column 'cell_type' missing from mean_intensity_df"
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


if __name__ == "__main__":
    unittest.main()
