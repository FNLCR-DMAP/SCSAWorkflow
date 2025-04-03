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
            heatmap_datashader(1, self.y)
        self.assertEqual(str(context_manager.exception),
                         "x and y must be array-like.")

    def test_labels_length_mismatch(self):
        """Test handling of mismatched lengths between data and labels."""
        wrong_labels = pd.Series(['A'] * 9)  # Shorter than x and y
        with self.assertRaises(ValueError) as context_manager:
            heatmap_datashader(self.x, self.y, labels=wrong_labels)
        expected_msg = "Labels length should match x and y length."
        self.assertEqual(str(context_manager.exception), expected_msg)


if __name__ == "__main__":
    unittest.main()
