import unittest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from spac.visualization import plot_qc_metrics

class TestPlotQCMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def create_test_adata(self):
        X = np.random.rand(10, 3)
        obs = pd.DataFrame({
            "nCount": np.random.randint(100, 1000, 10),
            "nFeature": np.random.randint(10, 100, 10),
            "percent.mt": np.random.rand(10) * 10,
            "group": ["A", "B"] * 5
        })
        var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
        adata = AnnData(X=X, obs=obs, var=var)
        return adata

    def test_plot_qc_metrics_returns_figure_and_axes(self):
        adata = self.create_test_adata()
        result = plot_qc_metrics(adata)
        self.assertIsInstance(result, dict)
        self.assertIn("figure", result)
        self.assertIn("axes", result)
        self.assertIsInstance(result["figure"], Figure)
        # axes can be a numpy array or a single Axes
        axes = result["axes"]
        self.assertTrue(isinstance(axes, (np.ndarray, Axes)))

    def test_plot_qc_metrics_with_annotation_column(self):
        adata = self.create_test_adata()
        result = plot_qc_metrics(adata, annotation="group")
        self.assertIsInstance(result["A"]["figure"], Figure)

    def test_plot_qc_metrics_with_log(self):
        adata = self.create_test_adata()
        result = plot_qc_metrics(adata, log=True)
        self.assertIsInstance(result["figure"], Figure)

if __name__ == "__main__":
    unittest.main()