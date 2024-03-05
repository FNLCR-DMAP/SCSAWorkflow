import unittest
import pandas as pd
import anndata as ad
import numpy as np
from spac.data_utils import select_values


class TestSelectValues(unittest.TestCase):
    def setUp(self):
        """Set up for testing select_values with both DataFrame and AnnData."""
        # DataFrame setup with values 'A', 'B', 'C'
        self.df = pd.DataFrame({
            'column1': ['A', 'B', 'A', 'B', 'A', 'C'],
            'column2': [1, 2, 3, 4, 5, 6]
        })

        # AnnData setup with values 'X', 'Y', 'Z' for the same 'column1'
        self.adata = ad.AnnData(
            np.random.rand(6, 2),
            obs={'column1': ['X', 'Y', 'X', 'Y', 'X', 'Z']}
        )

    def test_select_values_dataframe_typical_case(self):
        """
        Test selecting specified values from a DataFrame column.
        """
        result_df = select_values(self.df, 'column1', ['A', 'B'])
        # Expecting 5 rows where column1 is either 'A' or 'B'
        self.assertEqual(len(result_df), 5)

    def test_select_values_adata_typical_case(self):
        """
        Test selecting specified values from an AnnData object.
        """
        result_adata = select_values(self.adata, 'column1', ['X', 'Y'])
        # Expecting 5 observations where column1 is either 'X' or 'Y'
        self.assertEqual(result_adata.n_obs, 5)

    def test_select_values_dataframe_all_values(self):
        """
        Test returning all DataFrame rows when no specific values are given.
        """
        result_df = select_values(self.df, 'column1')
        # Expecting all rows to be returned
        self.assertEqual(len(result_df), 6)

    def test_select_values_adata_all_values(self):
        """
        Test returning all AnnData values when no specific values are given.
        """
        result_adata = select_values(self.adata, 'column1')
        # Expecting all values to be returned
        self.assertEqual(result_adata.n_obs, 6)

    def test_unsupported_data_type(self):
        """
        Test handling of unsupported data types.
        """
        with self.assertRaises(TypeError):
            select_values(["not", "a", "valid", "input"], 'column1', ['A'])

    def test_select_values_dataframe_nonexistent_values(self):
        """
        Test error raised for nonexistent values from a DataFrame.
        """
        with self.assertRaises(ValueError):
            select_values(self.df, 'column1', ['Nonexistent'])

    def test_select_values_adata_nonexistent_values(self):
        """
        Test error raised for nonexistent values from an AnnData object.
        """
        with self.assertRaises(ValueError):
            select_values(self.adata, 'column1', ['Nonexistent'])


if __name__ == '__main__':
    unittest.main()
