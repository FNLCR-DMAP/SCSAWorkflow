# tests/templates/test_histogram_template.py
"""
Real (non-mocked) unit test for the Histogram template.

Validates template I/O and title behaviour.
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
    """Minimal AnnData with separate plotted and grouping annotations."""
    rng = np.random.default_rng(42)
    X = rng.integers(1, 10, size=(4, 2)).astype(float)
    obs = pd.DataFrame({
        "cell_type": ["A", "B", "A", "B"],
        "batch": ["X", "X", "Y", "Y"],
    })
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
            "Table_to_Visualize": "Original",
            "Feature_s_to_Plot": ["All"],
            "Figure_Title": "Test Histogram",
            "Legend_Title": "Cell Type",
            "Figure_Width": 6,
            "Figure_Height": 4,
            "Figure_DPI": 72,
            "Font_Size": 10,
            "Number_of_Bins": 20,
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

    def _run_with_params(self, **updates):
        with open(self.json_file) as f:
            params = json.load(f)
        params.update(updates)
        return run_from_json(
            params,
            save_to_disk=False,
            show_plot=False,
            output_dir=self.tmp_dir.name,
        )

    def test_histogram_title_variants(self) -> None:
        """Cover feature, annotation, and grouping title variants."""
        # Plot the cell-type annotation, separated into one facet per batch.
        fig, _ = self._run_with_params(
            Plot_By="Annotation",
            Annotation="cell_type",
            Group_by="batch",
            Together=False,
            Facet=True,
            Max_Groups=2,
        )
        self.assertEqual(
            fig._suptitle.get_text(),
            'Histogram of "cell_type" faceted by "batch"',
        )

        fig, _ = self._run_with_params(
            Plot_By="Feature",
            Feature="Gene_0",
            Annotation="None",
            Group_by="cell_type",
            Together=True,
            Facet=False,
            Max_Groups=2,
        )
        self.assertEqual(
            fig.axes[0].get_title(),
            'Histogram of "Gene_0" grouped by "cell_type"',
        )

        fig, _ = self._run_with_params(
            Plot_By="Annotation",
            Annotation="cell_type",
            Group_by="batch",
            Facet=False,
            Together=True,
            Max_Groups=2,
        )
        self.assertEqual(
            fig.axes[0].get_title(),
            'Histogram of "cell_type" grouped by "batch"',
        )

        fig, _ = self._run_with_params(
            Plot_By="Annotation",
            Annotation="cell_type",
            Group_by="batch",
            Facet=False,
            Together=False,
            Max_Groups=2,
        )
        self.assertEqual(
            fig._suptitle.get_text(),
            'Histogram of "cell_type" grouped by "batch"',
        )

    def test_filtered_histogram_titles_include_suffixes(self) -> None:
        """Filtered grouped and faceted titles show the displayed counts."""
        # The fixture has exactly two batch annotations, X and Y.
        fig, _ = self._run_with_params(
            Plot_By="Annotation",
            Annotation="cell_type",
            Group_by="batch",
            Facet=False,
            Together=True,
            Max_Groups=1,
        )
        self.assertEqual(
            fig.axes[0].get_title(),
            'Histogram of "cell_type" grouped by "batch" '
            '(top 1 of 2 groups)',
        )

        fig, _ = self._run_with_params(
            Plot_By="Annotation",
            Annotation="cell_type",
            Group_by="batch",
            Together=False,
            Facet=True,
            Max_Groups=1,
        )
        self.assertEqual(
            fig.axes[0].get_title(),
            'Histogram of "cell_type" faceted by "batch" '
            '(top 1 of 2 facets)',
        )

    def test_unfiltered_histogram_titles_omit_suffix(self) -> None:
        """Titles omit the suffix when all available groups are shown."""
        # The fixture has exactly two batch annotations, X and Y.

        fig, _ = self._run_with_params(
            Plot_By="Annotation",
            Annotation="cell_type",
            Group_by="batch",
            Facet=False,
            Together=True,
            Max_Groups=2,
        )
        self.assertEqual(
            fig.axes[0].get_title(),
            'Histogram of "cell_type" grouped by "batch"',
        )


if __name__ == "__main__":
    unittest.main()
