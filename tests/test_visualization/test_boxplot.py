import unittest
import pandas as pd
import anndata
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

    def test_multiple_features_mode(self):
        """Test if multiple features mode works as expected."""
        fig, ax = boxplot(self.adata, features=['feature1', 'feature2'])

        # Get y-tick labels from the plot
        ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]

        # Check if y-tick labels match the feature names
        expected_features = ['feature1', 'feature2']
        self.assertEqual(set(ytick_labels), set(expected_features))

    def test_annotation_mode(self):
        """Test if annotation mode works as expected."""
        fig, ax = boxplot(self.adata, annotation='phenotype')
        ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        expected_unique_labels = ['phenotype1', 'phenotype2']
        for label in expected_unique_labels:
            self.assertIn(label, ytick_labels)

    def test_second_annotation_mode(self):
        """Test if second annotation mode works as expected."""
        self.adata.obs['treatment'] = ['Drug1', 'Drug1', 'Drug2', 'Drug2']
        fig, ax = boxplot(
            self.adata,
            annotation='phenotype',
            second_annotation='treatment'
        )
        ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        for label in ['phenotype1', 'phenotype2']:
            self.assertIn(label, ytick_labels)
        legend_labels = [text.get_text() for text in ax.legend().get_texts()]
        for label in ['Drug1', 'Drug2']:
            self.assertIn(label, legend_labels)


if __name__ == '__main__':
    unittest.main()
