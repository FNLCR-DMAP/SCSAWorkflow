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

    def test_custom_point_size(self):
        custom_size = 100
        _, ax = visualize_2D_scatter(self.x, self.y, point_size=custom_size)
        self.assertEqual(ax.collections[0].get_sizes()[0], custom_size)

    def test_categorical_labels(self):
        fig, ax = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical
        )
        # Check if legend is present
        self.assertTrue(len(ax.get_legend().get_texts()) == 3)

    def test_annotate_cluster_centers(self):
        fig, ax = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical,
            annotate_centers=True
        )
        self.assertEqual(
            len(ax.texts),
            len(self.labels_categorical.cat.categories)
        )

    def test_legend_placement_categorical_labels(self):
        fig, ax = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_categorical
        )

        # Check if legend is outside the plot
        legend = ax.get_legend()
        bbox = legend.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        self.assertTrue(bbox.x0 > 1, "Legend is not placed outside the plot.")

        # Check if hardcode the expected labels match the legend's labels
        expected_labels = ['A', 'B', 'C']
        legend_labels = [text.get_text() for text in legend.get_texts()]
        self.assertTrue(
            set(legend_labels) == set(expected_labels),
            f"Expected labels {expected_labels} but got {legend_labels}."
        )

    def test_continuous_labels(self):
        fig, ax = visualize_2D_scatter(
            self.x, self.y, labels=self.labels_continuous
        )
        # Check if colorbar is present
        self.assertIsNotNone(ax.collections[0].colorbar)

    def test_equal_aspect_ratio(self):
        _, ax = visualize_2D_scatter(self.x, self.y)
        self.assertTrue(ax.get_aspect() == 'equal' or ax.get_aspect() == 1.0)


if __name__ == '__main__':
    unittest.main()
