import unittest
import pandas as pd
from spac.data_utils import dataframe_select_values


class TestDataFrameSelectValues(unittest.TestCase):
    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.data = pd.DataFrame({
            'column1': ['A', 'B', 'A', 'B', 'A', 'C'],
            'column2': [1, 2, 3, 4, 5, 6]
        })

    def test_select_values_typical_case(self):
        """Test selecting specified values."""
        result = dataframe_select_values(self.data, 'column1', ['A', 'B'])
        self.assertEqual(len(result), 5)
        # Assert that 'C' is not in the result
        self.assertFalse((result['column1'] == 'C').any())

    def test_select_values_all_values(self):
        """Test selecting all values when none are specified."""
        result = dataframe_select_values(self.data, 'column1')
        self.assertEqual(len(result), 6)

    def test_no_matching_values(self):
        """Test selecting with no matching values."""
        # This expects a ValueError due to validation failure
        with self.assertRaises(ValueError):
            dataframe_select_values(self.data, 'column1', ['D'])

    def test_values_not_in_annotation(self):
        """Test selecting values not present in the specified column."""
        # This test ensures that the function raises a ValueError
        # when provided values do not exist in the annotation.
        with self.assertRaises(ValueError):
            dataframe_select_values(self.data, 'column1', ['Z'])

    def test_invalid_annotation(self):
        """Test with an invalid column name."""
        with self.assertRaises(ValueError):
            dataframe_select_values(self.data, 'invalid_column', ['A'])

    def test_empty_dataframe(self):
        """Test handling an empty dataframe."""
        empty_data = pd.DataFrame()
        result = dataframe_select_values(empty_data, 'column1', ['A'])
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
