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
        """
        Test that function correctly selects specified values from an
        annotation.
        """
        result = dataframe_select_values(self.data, 'column1', ['A', 'B'])
        self.assertEqual(len(result), 5)
        # Assert that 'C' is not in the result
        self.assertFalse((result['column1'] == 'C').any())

    def test_select_values_all_values(self):
        """
        Test that function correctly selects all values
        when no specific values given.
        """
        result = dataframe_select_values(self.data, 'column1')
        self.assertEqual(len(result), 6)

    def test_no_matching_values(self):
        """
        Test the case with no matching values.
        """
        result = dataframe_select_values(self.data, 'column1', ['D'])
        self.assertEqual(len(result), 0)

    def test_invalid_annotation(self):
        """
        Test function handling when an invalid annotation name is provided.
        """
        with self.assertRaises(ValueError):
            dataframe_select_values(self.data, 'invalid_column', ['A'])

    def test_empty_dataframe(self):
        """
        Test that function correctly handles an empty dataframe.
        """
        empty_data = pd.DataFrame()
        result = dataframe_select_values(empty_data, 'column1', ['A'])
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
