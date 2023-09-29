import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
import unittest
import pandas as pd
import anndata
from scanpy.plotting._matrixplot import MatrixPlot
import matplotlib
from spac.visualization import hierarchical_heatmap
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestHierarchicalHeatmap(unittest.TestCase):

    def setUp(self):
        """Set up testing environment."""
        X = pd.DataFrame({
            'feature1': [1, 3, 5, 7, 9, 11, 13, 15],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16],
            'feature3': [3, 5, 7, 9, 11, 13, 15, 17]
        })

        annotation = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2',
                'phenotype3',
                'phenotype3',
                'phenotype4',
                'phenotype4'
            ]
        })

        self.adata = anndata.AnnData(X=X, obs=annotation)

    def test_returns_correct_types(self):
        """Test if correct types are returned."""
        mean_intensity, matrixplot, dendro_data = hierarchical_heatmap(
            self.adata,
            'phenotype'
        )
        self.assertIsInstance(mean_intensity, pd.DataFrame)
        self.assertIsInstance(matrixplot, MatrixPlot)
        self.assertIsNotNone(dendro_data)
        self.assertIn('dendrogram_info', dendro_data)

    def test_nan_annotation_raises_error(self):
        """Test if NaN annotation raises error."""
        self.adata.obs['phenotype'] = [
            None,
            'phenotype1',
            'phenotype2',
            'phenotype1',
            'phenotype3',
            'phenotype3',
            'phenotype4',
            'phenotype4'
        ]
        with self.assertRaises(ValueError):
            hierarchical_heatmap(self.adata, 'phenotype')

    def test_calculates_correct_mean_intensity(self):
        """Test if mean intensity is correctly calculated."""
        mean_intensity, _, _ = hierarchical_heatmap(self.adata, 'phenotype')
        expected_mean = self.adata.X.mean(axis=0)
        self.assertTrue((mean_intensity.mean(axis=0).values == expected_mean).all())

    def test_z_score_normalization_value(self):
        """Test if z-score normalization is correctly applied across features."""
        mean_intensity, _, _ = hierarchical_heatmap(self.adata, 'phenotype', z_score='var')
        self.assertTrue(abs(mean_intensity.mean().mean()) < 1e-9)  # mean close to 0
        self.assertTrue((abs(mean_intensity.std() - 1) < 1e-9).all())  # standard deviation close to 1


    def test_z_score_features(self):
        """Test if z_score normalization by features works correctly."""
        mean_intensity, _, _ = hierarchical_heatmap(self.adata, 'phenotype',
        z_score="var")
        self.assertAlmostEqual(mean_intensity['feature1'].mean(), 0)
        self.assertAlmostEqual(mean_intensity['feature1'].std(), 1)

    def test_dendrogram_attributes(self):
        """Test if dendrogram data has expected attributes."""
        _, _, dendro_data = hierarchical_heatmap(self.adata, 'phenotype', dendrogram=True)
        self.assertIn('dendrogram_info', dendro_data)
        self.assertIn('leaves', dendro_data['dendrogram_info'])


if __name__ == '__main__':
    unittest.main()
