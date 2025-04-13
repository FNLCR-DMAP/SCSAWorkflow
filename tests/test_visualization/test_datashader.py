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
        self.labels_continuous = pd.Series(np.random.rand(10))

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

    def test_valid_input_returns_figure(self):
        """Test that valid input returns a matplotlib figure with expected subplots."""
        fig = heatmap_datashader(self.x, self.y, labels=self.labels_categorical)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

        # There should be as many subplots as unique labels (3 in this case)
        num_axes = len(fig.axes)
        expected_axes = self.labels_categorical.nunique()
        self.assertEqual(num_axes, expected_axes)


if __name__ == "__main__":
    unittest.main()