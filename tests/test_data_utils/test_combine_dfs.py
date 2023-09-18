import unittest
import pandas as pd
import numpy as np
import warnings
from spac.data_utils import combine_dfs


class TestCombineDFs(unittest.TestCase):

    def test_empty_input_list(self):
        expected_error_msg = "Input list is empty, please check."
        # Test when the input list is empty, it should raise a ValueError.
        with self.assertRaises(ValueError) as context:
            combine_dfs([])
            actual_error_message = str(context.exception)
            self.assertTrue(expected_error_msg in actual_error_message)

    def test_non_list_input(self):
        # Test when the input is not a list, it should raise a ValueError.
        expected_error_msg = "Input is not a list, please check."
        with self.assertRaises(ValueError) as context:
            combine_dfs("not_a_list")
            actual_warning = str(context.warning)
            self.assertTrue(expected_error_msg in actual_warning)

    def test_schema_warning_and_uneven_merge(self):
        # Test when the schemas of DataFrames are different,
        # it should print a warning.
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [1, 2], 'C': [5, 6]})

        expected_warning = "Schema of DataFrame 2 " + \
            "is different from the primary DataFrame."

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            combined_df = combine_dfs([df1, df2])

            # Check if a UserWarning is issued
            self.assertTrue(
                any(
                    issubclass(item.category, UserWarning) for item in w
                    )
                )

            # Check if the expected warning message
            # matches the actual warning message

            actual_warning = str(w[0].message)
            self.assertEqual(expected_warning, actual_warning)

            expected_df = pd.DataFrame(
                    {
                        'A': [1, 2, 1, 2],
                        'B': [3, 4, np.nan, np.nan],
                        'C': [np.nan, np.nan, 5, 6]
                    }
                )

            self.assertTrue(expected_df.equals(combined_df))

    def test_correct_input(self):
        # Test when the input is correct,
        # it should return the combined DataFrame.
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        expected_df = pd.DataFrame({'A': [1, 2, 5, 6], 'B': [3, 4, 7, 8]})

        combined_df = combine_dfs([df1, df2])
        self.assertTrue(expected_df.equals(combined_df))


if __name__ == '__main__':
    unittest.main()
