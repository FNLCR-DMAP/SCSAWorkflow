import unittest
import pandas as pd
import anndata
from scanpy.plotting._matrixplot import MatrixPlot
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
from spac.visualization import hierarchical_heatmap


class TestHierarchicalHeatmap(unittest.TestCase):
    def setUp(self):
        """Set up testing environment."""
        X = pd.DataFrame({
            'feature1': [1, 3, 5, 7],
            'feature2': [2, 4, 6, 8],
        })

        obs = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2'
            ]
        })

        self.adata = anndata.AnnData(X=X, obs=obs)

    def test_returns_correct_types(self):
        """Test if correct types are returned."""
        mean_intensity, matrixplot = hierarchical_heatmap(
            self.adata,
            'phenotype'
        )
        self.assertIsInstance(mean_intensity, pd.DataFrame)
        self.assertIsInstance(matrixplot, MatrixPlot)

    def test_invalid_observation_raises_error(self):
        """Test if invalid observation raises error."""
        err_msg = (r"The observation 'invalid_observation' does not exist "
                   r"in the provided AnnData object. Available observations "
                   r"are: \['phenotype'\]")
        with self.assertRaisesRegex(KeyError, err_msg):
            hierarchical_heatmap(self.adata, 'invalid_observation')

    def test_nan_observation_raises_error(self):
        """Test if NaN observation raises error."""
        self.adata.obs['phenotype'] = [
            None,
            'phenotype1',
            'phenotype2',
            'phenotype1'
        ]
        with self.assertRaises(ValueError):
            hierarchical_heatmap(self.adata, 'phenotype')

    def test_invalid_layer_raises_error(self):
        """Test if invalid layer raises error."""
        err_msg = ("The layer 'invalid_layer' does not exist in the "
                   "provided AnnData object")
        with self.assertRaisesRegex(KeyError, err_msg):
            hierarchical_heatmap(
                self.adata,
                'phenotype',
                layer='invalid_layer'
            )

    def test_calculates_correct_mean_intensity(self):
        """Test if correct mean intensities are calculated."""
        expected_mean_intensity = self.adata.to_df().groupby(
            self.adata.obs['phenotype']
        ).mean().reset_index()
        mean_intensity, _ = hierarchical_heatmap(self.adata, 'phenotype')
        self.assertEqual(mean_intensity.shape, (2, 3))
        pd.testing.assert_frame_equal(
            mean_intensity,
            expected_mean_intensity,
            check_exact=True
        )

    def test_matrixplot_attributes(self):
        """Test if correct matrixplot attributes are returned."""
        _, matrixplot = hierarchical_heatmap(self.adata, 'phenotype')
        self.assertEqual(len(matrixplot.var_names), 2)
        self.assertEqual(len(matrixplot.categories), 2)
        self.assertListEqual(
            sorted(list(matrixplot.var_names)),
            ['feature1', 'feature2']
        )
        self.assertListEqual(
            sorted(list(matrixplot.categories)),
            ['phenotype1', 'phenotype2']
        )


if __name__ == '__main__':
    unittest.main()
