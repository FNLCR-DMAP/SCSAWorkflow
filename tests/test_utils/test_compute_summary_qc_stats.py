import unittest
import numpy as np
import pandas as pd
from spac.utils import compute_summary_qc_stats

class TestComputeSummaryQCStats(unittest.TestCase):
    def setUp(self):
        # Create a simple DataFrame for testing
        self.df = pd.DataFrame({
            "nFeature": [2, 2, 2],
            "nCount": [4, 6, 11],
            "percent.mt": [25.0, 33.33333333333333, 45.45454545454545],
            "all_nan": [np.nan, np.nan, np.nan],
            "non_numeric": ["a", "b", "c"]
        })

    # Test that summary statistics are computed correctly for nFeature
    def test_basic_statistics(self):
        result = compute_summary_qc_stats(self.df, 
                                          stat_columns_list=["nFeature"])
        row = result.iloc[0]
        self.assertEqual(row["mean"], 2)
        self.assertEqual(row["median"], 2)
        self.assertEqual(row["upper_mad"], 2)
        self.assertEqual(row["lower_mad"], 2)
        self.assertEqual(row["upper_quantile"], 2)
        self.assertEqual(row["lower_quantile"], 2)

    # Test that summary statistics are computed correctly for nCount
    def test_ncount_statistics(self):
        # nCount: [4, 6, 11] -> mean 7.0, median 6.0, 95th pct 10.5, 5th pct 4.2
        result = compute_summary_qc_stats(self.df, 
                                          stat_columns_list=["nCount"])
        row = result.iloc[0]
        self.assertAlmostEqual(row["mean"], 7.0)
        self.assertAlmostEqual(row["median"], 6.0)
        self.assertAlmostEqual(row["upper_quantile"], 10.5)
        self.assertAlmostEqual(row["lower_quantile"], 4.2)

    # Test that summary statistics are computed correctly for percent.mt
    def test_percent_mt_statistics(self):
        # percent.mt: [25.0, 33.33333333333333, 45.45454545454545] ->
        # mean 34.59596, median 33.33333, upper_quantile 44.24242, 
        # lower_quantile 25.83333
        result = compute_summary_qc_stats(self.df, 
                                          stat_columns_list=["percent.mt"])
        row = result.iloc[0]
        self.assertAlmostEqual(row["mean"], 34.59596, places=5)
        self.assertAlmostEqual(row["median"], 33.33333, places=5)
        self.assertAlmostEqual(row["upper_quantile"], 44.24242, places=5)
        self.assertAlmostEqual(row["lower_quantile"], 25.83333, places=5)

    # Test that a TypeError is raised if a non-numeric column is included
    def test_non_numeric_column_raises(self):
        with self.assertRaises(TypeError) as exc_info:
            compute_summary_qc_stats(self.df, 
                                      stat_columns_list=["non_numeric"])
        expected_msg = (
            'Column "non_numeric" must be numeric to compute statistics.'
        )
        self.assertEqual(str(exc_info.exception), expected_msg)

    # Test that all-NaN columns are handled gracefully
    def test_all_nan_column_raises(self):
        with self.assertRaises(TypeError) as exc_info:
            compute_summary_qc_stats(self.df, stat_columns_list=["all_nan"])
        expected_msg = (
            'Column "all_nan" must be numeric to compute statistics. '
            'All values are NaN.'
        )
        self.assertEqual(str(exc_info.exception), expected_msg)

if __name__ == "__main__":
    unittest.main()
