import unittest
import numpy as np
import pandas as pd
from spac.visualization import visualize_2D_scatter
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestVisualize2DScatter(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(10)
        self.y = np.random.rand(10)
        self.labels_categorical = (
            pd.Series(np.random.choice(['A', 'B', 'C'], size=10))
            .astype('category')
        )
        self.labels_continuous = np.random.rand(10)

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError) as cm:
            visualize_2D_scatter(1, self.y)
        self.assertEqual(str(cm.exception), "x and y must be array-like.")

    def test_labels_length_mismatch(self):
        wrong_labels = np.random.choice(['A', 'B', 'C'], size=9)
        with self.assertRaises(ValueError) as cm:
            visualize_2D_scatter(self.x, self.y, labels=wrong_labels)
        expected_msg = "Labels length should match x and y length."
        self.assertEqual(str(cm.exception), expected_msg)

    def test_invalid_theme(self):
        with self.assertRaises(ValueError) as cm:
            visualize_2D_scatter(self.x, self.y, theme='invalid_theme')
        expected_msg = (
            "Theme 'invalid_theme' not recognized. Please use a valid theme."
        )
        self.assertEqual(str(cm.exception), expected_msg)

    def test_categorical_labels(self):
        fig, ax = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical
        )
        # Check if legend is present
        self.assertTrue(len(ax.get_legend().get_texts()) > 0)

    def test_continuous_labels(self):
        fig, ax = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_continuous
        )
        # Check if colorbar is present
        self.assertIsNotNone(ax.collections[0].colorbar)


if __name__ == '__main__':
    unittest.main()
