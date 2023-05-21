import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")

from spac.visualization import hierarchical_heatmap
import anndata
import pandas as pd
import numpy as np
import unittest


class TestHierarchicalHeatmap(unittest.TestCase):
    def setUp(self):
        """Set up testing environment."""
        X = pd.DataFrame([[1, 2], [3, 4]], columns=['gene1', 'gene2'])
        obs = pd.DataFrame(['type1', 'type2'], columns=['cell_type'])
        self.adata = anndata.AnnData(X=X, obs=obs)

    def test_returns_correct_types(self):
        """Test if correct types are returned."""
        mean_intensity, matrixplot = hierarchical_heatmap(
            self.adata, 'cell_type')

        self.assertIsInstance(mean_intensity, pd.DataFrame)
        self.assertEqual(str(type(matrixplot)),
                         "<class 'scanpy.plotting._matrixplot.MatrixPlot'>")

    def test_invalid_observation_raises_error(self):
        """Test if invalid observation raises error."""
        with self.assertRaises(KeyError):
            hierarchical_heatmap(self.adata, 'invalid_observation')

    def test_nan_observation_raises_error(self):
        """Test if NaN observation raises error."""
        self.adata.obs['cell_type'] = [None, 'type2']

        with self.assertRaises(ValueError):
            hierarchical_heatmap(self.adata, 'cell_type')

    def test_invalid_layer_raises_error(self):
        """Test if invalid layer raises error."""
        with self.assertRaises(KeyError):
            hierarchical_heatmap(self.adata, 'cell_type',
                                 layer='invalid_layer')

    def create_test_anndata(self, n_cells=100, n_genes=10):
        """Create a test AnnData object."""
        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(index=np.arange(n_cells))
        obs['group'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
        var = pd.DataFrame(index=np.arange(n_genes))
        return anndata.AnnData(X=X, obs=obs, var=var)

    def test_hierarchical_heatmap(self):
        """Test hierarchical heatmap function."""
        test_adata = self.create_test_anndata()
        mean_intensity, matrixplot = hierarchical_heatmap(
            test_adata, observation='group')

        self.assertIsInstance(mean_intensity, pd.DataFrame)
        self.assertEqual(mean_intensity.shape[1], test_adata.n_vars + 1)
        self.assertEqual(list(mean_intensity.columns[1:]),
                         list(test_adata.var.index))

        self.assertEqual(str(type(matrixplot)),
                         "<class 'scanpy.plotting._matrixplot.MatrixPlot'>")


if __name__ == '__main__':
    unittest.main()
