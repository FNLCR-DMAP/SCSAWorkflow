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

    def test_invalid_annotation_raises_error(self):
        """Test if invalid annotation raises error."""
        err_msg = "Specified annotation 'invalid_annotation' not found " \
                  "in the provided AnnData object's .obs."
        with self.assertRaisesRegex(ValueError, err_msg):
            boxplot(self.adata, annotation='invalid_annotation')

    def test_invalid_layer_raises_error(self):
        """Test if invalid layer raises error."""
        err_msg = ("Specified layer 'invalid_layer' not found in the "
                   "provided AnnData object.")
        with self.assertRaisesRegex(ValueError, err_msg):
            boxplot(self.adata, layer='invalid_layer')

    def test_feature_filtering_works(self):
        """Test if feature filtering works."""
        fig, ax = boxplot(self.adata, features=['feature1'])
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        self.assertIn('feature1', xtick_labels)
        self.assertNotIn('feature2', xtick_labels)

    def test_invalid_features_raises_error(self):
        """Test if invalid features raise an error."""
        err_msg = "One or more provided features are not found in the data."
        with self.assertRaisesRegex(ValueError, err_msg):
            boxplot(self.adata, features=['invalid_feature'])


if __name__ == '__main__':
    unittest.main()

