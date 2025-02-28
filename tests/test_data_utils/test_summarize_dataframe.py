import unittest
import pandas as pd
import numpy as np
from io import StringIO
from contextlib import redirect_stdout

from spac.data_utils import summarize_dataframe


class TestSummarizeDataFrame(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with both numeric and categorical columns and some missing values
        self.df = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', 'A', None, 'B']
        })

    def test_summary_content(self):
        # Test without printing missing locations
        result = summarize_dataframe(
            self.df, columns=['numeric_col', 'categorical_col']
        )

        # Test numeric column
        self.assertIn('numeric_col', result)
        num_info = result['numeric_col']
        self.assertEqual(num_info['data_type'], 'numeric')
        self.assertEqual(num_info['missing_indices'], [2])
        self.assertEqual(num_info['count_missing_indices'], 1)
        # Check that expected keys exist in numeric summary
        expected_numeric_keys = {'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'}
        self.assertTrue(
            expected_numeric_keys.issubset(num_info['summary'].keys())
        )

        # Test categorical column
        self.assertIn('categorical_col', result)
        cat_info = result['categorical_col']
        self.assertEqual(cat_info['data_type'], 'categorical')
        self.assertEqual(cat_info['missing_indices'], [3])
        self.assertEqual(cat_info['count_missing_indices'], 1)
        # Check that summary for categorical contains unique_values
        # and value_counts
        self.assertIn('unique_values', cat_info['summary'])
        self.assertIn('value_counts', cat_info['summary'])
        # There should be two unique non-missing values for categorical_col, 'A' and 'B'
        self.assertCountEqual(cat_info['summary']['unique_values'], ['A', 'B'])
        # Check value counts (non missing)
        self.assertEqual(cat_info['summary']['value_counts']['A'], 2)
        self.assertEqual(cat_info['summary']['value_counts']['B'], 2)

    def test_print_nan_locations_output(self):
        # Capture the standard output for the print_nan_locations option
        output = StringIO()
        with redirect_stdout(output):
            summarize_dataframe(
                self.df, columns=[
                    'numeric_col', 'categorical_col'
                ], print_nan_locations=True
            )
        printed_output = output.getvalue()
        # Check if specific printed message appears, e.g. missing indices
        # information for numeric_col
        self.assertIn(
            "Column 'numeric_col' has missing values at rows:", printed_output
        )
        # And for categorical_col as well
        self.assertIn(
            "Column 'categorical_col' has missing values at rows:",
            printed_output
        )


if __name__ == '__main__':
    unittest.main()
