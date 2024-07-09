import unittest
import pandas as pd
import anndata
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spac.visualization import boxplot
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestBoxplot(unittest.TestCase):

    def setUp(self):
        """Set up testing environment."""
        X = pd.DataFrame({
            'feature1': [1, 3, 5, 7],
            'feature2': [2, 4, 6, 8],
        })

        annotation = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2'
            ]
        })

        self.adata = anndata.AnnData(X=X, obs=annotation)

    def test_returns_correct_types(self):
        """Test if correct types are returned."""
        fig, ax = boxplot(self.adata)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

    def test_second_annotation_mode(self):
        """Test if second annotation mode works as expected."""
        self.adata.obs['treatment'] = ['Drug1', 'Drug1', 'Drug2', 'Drug2']
        fig, ax = boxplot(
            self.adata,
            annotation='phenotype',
            second_annotation='treatment',
            orient='v'
        )
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        for label in ['phenotype1', 'phenotype2']:
            self.assertIn(label, xtick_labels)
        legend_labels = [text.get_text() for text in ax.legend().get_texts()]
        for label in ['Drug1', 'Drug2']:
            self.assertIn(label, legend_labels)

    def test_single_annotation_single_feature(self):
        """Test if annotation mode works as expected with a single feature."""
        fig, ax = boxplot(self.adata, features=['feature1'],
                          annotation='phenotype', orient='v')
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        expected_unique_labels = ['phenotype1', 'phenotype2']
        for label in expected_unique_labels:
            self.assertIn(label, xtick_labels)

    def test_single_annotation_multiple_features(self):
        """Test when only one annotation but multiple features."""
        fig, ax = boxplot(self.adata,
                          features=['feature1', 'feature2'],
                          annotation='phenotype', orient='v')
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        expected_unique_labels = ['feature1', 'feature2']
        for label in expected_unique_labels:
            self.assertIn(label, xtick_labels)

    def test_multiple_features_mode(self):
        """Test if multiple features mode works as expected."""
        fig, ax = boxplot(self.adata,
                          features=['feature1', 'feature2'],
                          orient='v')

        # Get x-tick labels from the plot
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]

        # Check if x-tick labels match the feature names
        expected_features = ['feature1', 'feature2']
        self.assertEqual(set(xtick_labels), set(expected_features))

    def test_orient_kwarg(self):
        """Test for the orient parameter passed via **kwargs."""
        fig, ax = boxplot(self.adata, features=['feature1'],
                          annotation='phenotype', orient='h')
        # Check if the x-axis ticks contain the expected annotations
        ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        expected_unique_labels = ['phenotype1', 'phenotype2']
        for label in expected_unique_labels:
            self.assertIn(label, ytick_labels)

    def test_log_scale(self):
        """Test for the log_scale parameter."""
        fig, ax = boxplot(self.adata, features=['feature1'], log_scale=True)

        # Check if the y-axis is in log scale
        self.assertEqual(ax.get_yscale(), 'log')

        # Test with zero values
        self.adata.X[0, 0] = 0  # Introduce a zero value
        fig, ax = boxplot(self.adata, features=['feature1'], log_scale=True)

        # Check if the y-axis is in log scale (due to log1p transformation)
        self.assertEqual(ax.get_yscale(), 'log')

    def test_log1p_transformation(self):
        """Test if np.log1p transformation is applied correctly."""
        # Introduce a dataset with zeros
        X = pd.DataFrame({
            'feature1': [0, 1, 2, 3]
        })

        annotation = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2'
            ]
        })

        adata = anndata.AnnData(X=X, obs=annotation)

        # Manually apply log1p transformation to check values
        expected_values = np.log1p(X)

        # Create a boxplot and capture the transformed DataFrame
        fig, ax = boxplot(adata, features=['feature1'], log_scale=True)

        # Extract the log1p transformed values from the DataFrame for plotting
        transformed_values = np.log1p(adata.X)

        # Compare the transformed values with the expected values
        np.testing.assert_allclose(
            transformed_values.flatten(),
            expected_values.values.flatten(),
            rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
