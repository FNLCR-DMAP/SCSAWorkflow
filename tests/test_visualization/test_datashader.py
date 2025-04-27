import unittest
import numpy as np
import pandas as pd
from spac.visualization import heatmap_datashader
import matplotlib

matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window

class TestDataShaderHeatMap(unittest.TestCase):
    def setUp(self):
        """Prepare data for testing."""
        self.x = np.random.rand(10)
        self.y = np.random.rand(10)
        # Fixed categorical labels to ensure representation of each category
        fixed_labels = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
        self.labels_categorical = pd.Series(fixed_labels, dtype="category")

    def test_invalid_input_type(self):
        """Test handling of invalid input types."""
        with self.assertRaises(ValueError) as context_manager:
            heatmap_datashader(1, self.y, labels=self.labels_categorical)
        self.assertIn("x and y must be array-like", str(context_manager.exception))

    def test_labels_length_mismatch(self):
        """Test handling of mismatched lengths between data and labels."""
        wrong_labels = pd.Series(['A'] * 9)  # Shorter than x and y
        with self.assertRaises(ValueError) as context_manager:
            heatmap_datashader(self.x, self.y, labels=wrong_labels)
        self.assertIn("Labels length should match x and y length", str(context_manager.exception))

    def test_valid_input_returns_figure_basic(self):
        """Test that valid input returns a matplotlib figure with expected subplots."""
        fig = heatmap_datashader(self.x, self.y, labels=self.labels_categorical)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        num_axes = len(fig.axes)
        expected_axes = self.labels_categorical.nunique()
        self.assertEqual(num_axes, expected_axes)
        
    def test_labels_not_multiple_of_three(self):
        """Test heatmap generation when the number of labels is not a multiple of 3."""
        x = np.random.rand(7)
        y = np.random.rand(7)
        labels = pd.Series(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype="category")  # 7 labels

        fig = heatmap_datashader(x, y, labels=labels)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        num_axes = len(fig.axes)
        expected_axes = labels.nunique()
        self.assertEqual(num_axes, expected_axes)

        for ax in fig.axes:
            images = [child for child in ax.get_children() if isinstance(child, matplotlib.image.AxesImage)]
            self.assertGreater(len(images), 0, "Expected at least one image in each subplot.")

    def test_valid_input_returns_figure(self):
        """Test that valid input returns a matplotlib figure with expected subplots and images."""
        fig = heatmap_datashader(self.x, self.y, labels=self.labels_categorical)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        # Check number of axes matches number of unique labels
        num_axes = len(fig.axes)
        expected_axes = self.labels_categorical.nunique()
        self.assertEqual(num_axes, expected_axes)

        # Check that each axis has an image plotted
        for ax in fig.axes:
            images = [child for child in ax.get_children() if isinstance(child, matplotlib.image.AxesImage)]
            self.assertGreater(len(images), 0, "Expected at least one image in each subplot.")



if __name__ == "__main__":
    unittest.main()