import unittest
import pandas as pd
import anndata
import matplotlib
import seaborn as sns
from spac.visualization import hierarchical_heatmap

matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestHierarchicalHeatmap(unittest.TestCase):

    def setUp(self):
        X = pd.DataFrame({
            'feature1': [1, 3, 5, 7, 9, 12, 14, 16],
            'feature2': [2, 4, 6, 8, 10, 13, 15, 18],
            'feature3': [3, 5, 7, 9, 11, 14, 16, 19]
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
        """Test if correct mean intensities are calculated."""

        # Ensure annotation column is categorical
        mean_intensity, _, _ = hierarchical_heatmap(self.adata, 'phenotype')
        self.assertTrue(
            pd.api.types.is_categorical_dtype(self.adata.obs['phenotype'])
        )

        # Hardcoded expected mean intensities based on the input data
        expected_mean_intensity = pd.DataFrame({
            # Average values for each phenotype
            'feature1': [2.0, 6.0, 10.5, 15.0],
            'feature2': [3.0, 7.0, 11.5, 16.5],
            'feature3': [4.0, 8.0, 12.5, 17.5]
        }, index=pd.Categorical([
            'phenotype1', 'phenotype2', 'phenotype3', 'phenotype4'
        ]), dtype='float64')

        # Set the name of the index for the expected_mean_intensity DataFrame
        expected_mean_intensity.index.name = 'phenotype'

        # Ensure the calculated mean intensity matches expected mean intensity
        pd.testing.assert_frame_equal(
            mean_intensity, expected_mean_intensity, check_exact=True
        )

    def test_zscore_normalization(self):
        """Test z-score normalization."""
        _, clustergrid, _ = hierarchical_heatmap(
            self.adata, 'phenotype', z_score="feature"
        )

        # After z-score normalization, the mean of each feature should be
        # approximately 0. We round to 2 decimal places to account for
        # potential floating-point inaccuracies.
        self.assertTrue((clustergrid.data2d.mean().round(2) == 0).all())

    def test_clustergrid_attributes(self):
        """Test if correct clustergrid data attributes are returned."""
        _, clustergrid, _ = hierarchical_heatmap(self.adata, 'phenotype')
        self.assertIsInstance(clustergrid, sns.matrix.ClusterGrid)

        # Adjust the expected shape
        num_unique_phenotypes = self.adata.obs['phenotype'].nunique()
        self.assertEqual(
            clustergrid.data2d.shape,
            (num_unique_phenotypes, self.adata.n_vars)
        )

    def test_dendrogram_attribute(self):
        """Test dendrogram attributes."""
        _, _, dendrogram_data = hierarchical_heatmap(self.adata, 'phenotype')

        # Check if dendrogram_data contains the required linkage keys.
        self.assertIn('row_linkage', dendrogram_data)
        self.assertIn('col_linkage', dendrogram_data)

    def test_axes_switching(self):
        """Test axes switching."""
        mean_intensity_default, _, _ = hierarchical_heatmap(
            self.adata, 'phenotype', swap_axes=False
        )
        mean_intensity_swapped, _, _ = hierarchical_heatmap(
            self.adata, 'phenotype', swap_axes=True
        )
        self.assertNotEqual(
            mean_intensity_default.shape,
            mean_intensity_swapped.shape
        )

    def test_cell_count_labels(self):
        '''This test confirms the cell count labels are correct'''
        # Set up AnnData object with known cell counts (3 cells in 1 cluster)
        X_data = pd.DataFrame(
            {'gene1': [1, 2, 3], 'gene2': [1, 2, 3], 'gene3': [1, 2, 3]}
        )
        obs_data = pd.DataFrame({'cluster': [1, 1, 1]})
        self.adata = anndata.AnnData(X=X_data, obs=obs_data)

        # Use hierarchical_heatmap function on AnnData object and get y-axis labels
        _, clustergrid, _ = hierarchical_heatmap(self.adata, annotation='cluster')
        actual_labels = []
        for label in clustergrid.ax_heatmap.get_yticklabels():
            actual_labels.append(label.get_text())

        # Confirm actual labels are in the expected labels
        # dendrogram order varies if multiple clusters
        expected_labels = ['cluster 1\n3 cells']
        self.assertIn(expected_labels[0], actual_labels)


if __name__ == "__main__":
    unittest.main()
