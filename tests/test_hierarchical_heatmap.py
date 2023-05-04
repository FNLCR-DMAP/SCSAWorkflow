import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from scsaworkflow.visualization import hierarchical_heatmap
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unittest


class TestHierarchicalHeatmap(unittest.TestCase):
    
    def create_test_anndata(self, n_cells=100, n_genes=10):
        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(index=np.arange(n_cells))
        obs['group'] = np.random.choice(['A', 'B', 'C'], size=n_cells)
        var = pd.DataFrame(index=np.arange(n_genes))
        return anndata.AnnData(X=X, obs=obs, var=var)
    

    def test_hierarchical_heatmap(self):
        test_adata = self.create_test_anndata()
        mean_intensity, matrixplot = hierarchical_heatmap(test_adata, column='group')

        # Test mean_intensity object
        self.assertIsInstance(mean_intensity, pd.DataFrame)
        self.assertEqual(mean_intensity.shape[1], test_adata.n_vars + 1)
        self.assertEqual(list(mean_intensity.columns[1:]), list(test_adata.var.index))

        # Test matrixplot object
        self.assertEqual(str(type(matrixplot)), "<class 'scanpy.plotting._matrixplot.MatrixPlot'>")


if __name__ == '__main__':
    unittest.main()
