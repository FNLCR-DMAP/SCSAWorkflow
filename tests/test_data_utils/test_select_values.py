import os
import sys
import unittest
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
from spac.data_utils import select_values


class TestSelectValues(unittest.TestCase):
    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.data = pd.DataFrame({
            'column1': ['A', 'B', 'A', 'B', 'A', 'C'],
            'column2': [1, 2, 3, 4, 5, 6]
        })

    def test_select_values_typical_case(self):
        """
        Test that function correctly selects specified values from a column.
        """
        result = select_values(self.data, 'column1', ['A', 'B'])
        self.assertEqual(len(result), 5)

    def test_select_values_all_values(self):
        """
        Test that function correctly selects all values
        when no specific values given.
        """
        result = select_values(self.data, 'column1')
        self.assertEqual(len(result), 6)

    def test_select_values_no_matching_values(self):
        """
        Test function handling of case where no values in the
        column match the specified values.
        """
        result = select_values(self.data, 'column1', ['D'])
        self.assertEqual(len(result), 0)

    def test_select_values_invalid_column(self):
        """Test function handling when an invalid column name is provided."""
        with self.assertRaises(ValueError):
            select_values(self.data, 'invalid_column', ['A'])

    def test_select_values_empty_dataframe(self):
        """Test that function correctly handles an empty dataframe."""
        empty_data = pd.DataFrame()
        result = select_values(empty_data, 'column1', ['A'])
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
