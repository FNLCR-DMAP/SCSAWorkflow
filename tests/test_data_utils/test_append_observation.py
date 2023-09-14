import pandas as pd
import unittest
from spac.data_utils import append_observation


class TestAppendObservation(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['apple', 'banana', 'cherry', 'date']
        })

    def test_append_observation_success_str_to_str(self):
        # Test a successful case
        result = append_observation(
            self.data, 'B', 'C', ['banana:test1', 'date:test2']
        )
        expected = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['apple', 'banana', 'cherry', 'date'],
            'C': ['Not_Mapped', 'test1', 'Not_Mapped', 'test2']
        })

        # Ensure both DataFrames have the same shape
        self.assertEqual(result.shape, expected.shape)

        # Compare each element of the DataFrames
        for col in result.columns:
            for idx in result.index:
                self.assertEqual(
                    result.at[idx, col],
                    expected.at[idx, col]
                )

    def test_append_observation_success_int_to_str(self):
        # Test a successful case
        result = append_observation(
            self.data, 'A', 'C', ['2:test1', '4:test2']
        )
        expected = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': ['apple', 'banana', 'cherry', 'date'],
            'C': ['Not_Mapped', 'test1', 'Not_Mapped', 'test2']
        })

        # Ensure both DataFrames have the same shape
        self.assertEqual(result.shape, expected.shape)

        # Compare each element of the DataFrames
        for col in result.columns:
            for idx in result.index:
                self.assertEqual(
                    result.at[idx, col],
                    expected.at[idx, col]
                )

    def test_source_column_not_in_dataframe(self):
        # Test when the source column doesn't exist in the DataFrame
        with self.assertRaises(ValueError) as context:
            append_observation(self.data, 'X', 'C', ['2:apple'])
        self.assertEqual(
            str(context.exception), "'X' does not exist in the DataFrame."
        )

    def test_new_column_already_exists(self):
        # Test when the new column already exists in the DataFrame
        with self.assertRaises(ValueError) as context:
            append_observation(self.data, 'A', 'B', ['2:apple'])
        self.assertEqual(
            str(context.exception), "'B' already exist in the DataFrame."
        )

    def test_mapping_rules_invalid_format(self):
        # Test when a mapping rule has an invalid format
        with self.assertRaises(ValueError) as context:
            append_observation(self.data, 'A', 'C', ['2:apple', 'banana'])
        self.assertEqual(
            str(context.exception),
            "Invalid mapping rule format: 'banana'. "
            "It should have the format "
            "<value in new observation>:<value in source column>."
        )

    def test_mapping_rules_not_list(self):
        # Test when mapping_rules is not a list
        with self.assertRaises(ValueError) as context:
            append_observation(self.data, 'A', 'C', '2:apple')
        self.assertEqual(
            str(context.exception), "Mapping rules must be provided as a list."
        )


if __name__ == '__main__':
    unittest.main()
