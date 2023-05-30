import unittest
import pandas as pd
import numpy as np
from spac.data_utils import calculate_centroid


class TestCalculateCentroid(unittest.TestCase):

    def setUp(self):
        # Set up a sample dataframe for testing
        self.df_str = pd.DataFrame({
            'x_min': [1, 2, 3, 4],
            'x_max': [2, 3, 4, 5],
            'y_min': [1, 2, 3, 4],
            'y_max': [2, 3, 4, 5]
        })

        self.df_num = pd.DataFrame({
            1: [1, 2, 3, 4],
            2: [2, 3, 4, 5],
            3: [1, 2, 3, 4],
            4: [2, 3, 4, 5]
        })

        self.df_neg = pd.DataFrame({
            'x_min': [-3, -2, -1, 0],
            'x_max': [-2, -1, 0, 1],
            'y_min': [-3, -2, -1, 0],
            'y_max': [-2, -1, 0, 1]
        })

        self.df_non_num = pd.DataFrame({
            'x_min': ['a', 'b', 'c', 'd'],
            'x_max': ['e', 'f', 'g', 'h'],
            'y_min': ['i', 'j', 'k', 'l'],
            'y_max': ['m', 'n', 'o', 'p']
        })

        self.df_nan = pd.DataFrame({
            'x_min': [1, 2, np.nan, 4],
            'x_max': [2, 3, 4, 5],
            'y_min': [1, np.nan, 3, 4],
            'y_max': [2, 3, 4, np.nan]
        })

    def test_calculate_centroid_str(self):
        result = calculate_centroid(
            self.df_str,
            'x_min',
            'x_max',
            'y_min',
            'y_max',
            'new_x',
            'new_y'
            )
        self.assertTrue('new_x' in result.columns)
        self.assertTrue('new_y' in result.columns)
        self.assertTrue(
            np.array_equal(
                result['new_x'],
                pd.Series([1.5, 2.5, 3.5, 4.5])
                )
            )
        self.assertTrue(
            np.array_equal(
                result['new_y'],
                pd.Series([1.5, 2.5, 3.5, 4.5])
                )
            )

    def test_calculate_centroid_num(self):
        result = calculate_centroid(
            self.df_num,
            1,
            2,
            3,
            4,
            'new_x',
            'new_y'
        )
        self.assertTrue('new_x' in result.columns)
        self.assertTrue('new_y' in result.columns)
        self.assertTrue(
            np.array_equal(
                result['new_x'],
                pd.Series([1.5, 2.5, 3.5, 4.5])
            )
        )
        self.assertTrue(
            np.array_equal(
                result['new_y'],
                pd.Series([1.5, 2.5, 3.5, 4.5])
            )
        )

    def test_nonexistent_column_str(self):
        with self.assertRaises(ValueError):
            calculate_centroid(
                self.df_str,
                'nonexistent',
                'x_max',
                'y_min',
                'y_max',
                'new_x',
                'new_y'
            )

    def test_nonexistent_column_num(self):
        with self.assertRaises(ValueError):
            calculate_centroid(
                self.df_num,
                5,
                2,
                3,
                4,
                'new_x',
                'new_y'
            )

    def test_invalid_new_column_name(self):
        with self.assertRaises(ValueError):
            calculate_centroid(
                self.df_str,
                'x_min',
                'x_max',
                'y_min',
                'y_max',
                'invalid-column',
                'new_y'
            )

    def test_negative_coordinates(self):
        result = calculate_centroid(
            self.df_neg,
            'x_min',
            'x_max',
            'y_min',
            'y_max',
            'new_x',
            'new_y'
        )
        self.assertTrue('new_x' in result.columns)
        self.assertTrue('new_y' in result.columns)
        self.assertTrue(
            np.array_equal(
                result['new_x'], pd.Series([-2.5, -1.5, -0.5, 0.5])
                )
            )
        self.assertTrue(
            np.array_equal(
                result['new_y'], pd.Series([-2.5, -1.5, -0.5, 0.5])
                )
            )

    def test_non_numeric_data(self):
        with self.assertRaises(TypeError):
            calculate_centroid(
                self.df_non_num,
                'x_min',
                'x_max',
                'y_min',
                'y_max',
                'new_x',
                'new_y'
            )

    def test_nan_values(self):
        result = calculate_centroid(
            self.df_nan,
            'x_min',
            'x_max',
            'y_min',
            'y_max',
            'new_x',
            'new_y'
        )
        self.assertTrue('new_x' in result.columns)
        self.assertTrue('new_y' in result.columns)
        self.assertTrue(np.isnan(result['new_x'][2]))
        self.assertTrue(np.isnan(result['new_y'][1]))
        self.assertTrue(np.isnan(result['new_y'][3]))


if __name__ == '__main__':
    unittest.main()
