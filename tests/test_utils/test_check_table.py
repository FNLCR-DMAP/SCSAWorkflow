import unittest
import numpy as np
import pandas as pd
import anndata as ad
from spac.utils import check_table


class TestCheckTable(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object for testing
        data = {
            "feature1": [1, 3],
            "feature2": [2, 4]
        }
        self.adata = ad.AnnData(
            X=pd.DataFrame(data),
            layers={"table1": np.array([[5, 6], [7, 8]]),
                    "table2": np.array([[9, 10], [11, 12]])},
            obs={"obs1": [1, 2],
                 "obs2": [3, 4]}
        )

    def test_wrong_adata_type(self):
        with self.assertRaises(TypeError) as context:
            check_table("not an AnnData object")
        self.assertEqual(
            str(context.exception),
            "Input dataset should be an instance of anndata.AnnData, "
            "please check the input dataset source."
        )

    def test_valid_tables(self):
        # Test with valid tables
        self.assertIsNone(check_table(self.adata, tables="table1"))

    def test_valid_list_tables(self):
        # Test with list input for tables
        self.assertIsNone(check_table(self.adata, tables=["table1", "table2"]))

    def test_invalid_tables(self):
        # Test with invalid tables
        with self.assertRaises(ValueError):
            check_table(self.adata, tables="invalid_table")
        with self.assertRaises(ValueError):
            check_table(self.adata, tables=["table1", "invalid_table"])

    def test_missing_tables(self):
        with self.assertRaises(ValueError) as context:
            check_table(self.adata,
                        tables=["table1",
                                "missing_table2"])
        self.assertEqual(
            "The table 'missing_table2' "
            "does not exist in the provided dataset.\n"
            "Existing tables are:\ntable1\ntable2",
            str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            check_table(self.adata,
                        tables=["missing_table1",
                                "table2"])
        self.assertEqual(
            "The table 'missing_table1' "
            "does not exist in the provided dataset.\n"
            "Existing tables are:\ntable1\ntable2",
            str(context.exception)
        )

    def test_valid_new_tables(self):
        # Test with valid new tables
        self.assertIsNone(
            check_table(self.adata,
                        tables=["new_table1", "new_table2"],
                        should_exist=False))

    def test_invalid_new_tables(self):
        with self.assertRaises(ValueError):
            check_table(self.adata,
                        tables=["new_table1", "table2"],
                        should_exist=False)


if __name__ == "__main__":
    unittest.main()
