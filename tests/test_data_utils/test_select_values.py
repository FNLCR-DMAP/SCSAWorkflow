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

    def test_dataframe_nonexistent_annotation(self):
        """
        Test error raised for a nonexistent annotation in a DataFrame.
        """
        with self.assertRaises(ValueError):
            select_values(self.df, 'nonexistent_column', ['A', 'B'])

    def test_adata_nonexistent_annotation(self):
        """
        Test error raised for a nonexistent annotation in an AnnData object.
        """
        with self.assertRaises(ValueError):
            select_values(self.adata, 'nonexistent_column', ['X', 'Y'])

    def test_select_values_dataframe_typical_case(self):
        """
        Test selecting specified values from a DataFrame column.
        """
        result_df = select_values(self.df, 'column1', ['A', 'B'])
        # Expecting 5 rows where column1 is either 'A' or 'B'
        self.assertEqual(len(result_df), 5)
        # Assert that the sets of unique values in the result and expected
        # values are identical.
        expected_values = ['A', 'B']
        unique_values_in_result = result_df['column1'].unique().tolist()
        self.assertCountEqual(unique_values_in_result, expected_values)

    def test_select_values_adata_typical_case(self):
        """
        Test selecting specified values from an AnnData object.
        """
        result_adata = select_values(self.adata, 'column1', ['X', 'Y'])
        # Expecting 5 rows where column1 is either 'X' or 'Y'
        self.assertEqual(result_adata.n_obs, 5)
        unique_values_in_result = result_adata.obs['column1'].unique().tolist()
        expected_values = ['X', 'Y']
        self.assertCountEqual(unique_values_in_result, expected_values)

    def test_select_values_dataframe_all_values(self):
        """
        Test returning all DataFrame rows when no specific values are given.
        """
        result_df = select_values(self.df, 'column1')
        # Expecting all rows to be returned
        self.assertEqual(len(result_df), 6)
        unique_values_in_result = result_df['column1'].unique().tolist()
        expected_values = ['A', 'B', 'C']
        self.assertCountEqual(unique_values_in_result, expected_values)

    def test_select_values_adata_all_values(self):
        """
        Test returning all AnnData values when no specific values are given.
        """
        result_adata = select_values(self.adata, 'column1')
        # Expecting all rows to be returned
        self.assertEqual(result_adata.n_obs, 6)
        unique_values_in_result = result_adata.obs['column1'].unique().tolist()
        expected_values = ['X', 'Y', 'Z']
        self.assertCountEqual(unique_values_in_result, expected_values)

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
