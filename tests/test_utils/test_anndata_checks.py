import unittest
import numpy as np
import pandas as pd
import anndata as ad
from spac.utils import anndata_checks


class TestAnndataChecks(unittest.TestCase):

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

    def test_valid_input(self):
        # Test with valid input
        self.assertIsNone(anndata_checks(self.adata))
        # No errors should be raised

    def test_invalid_adata_type(self):
        # Test with invalid adata type
        with self.assertRaises(TypeError):
            anndata_checks("not_an_anndata_object")

    def test_invalid_tables(self):
        # Test with invalid tables
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, tables="invalid_table")
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, tables=["table1", "invalid_table"])

    def test_invalid_obs(self):
        # Test with invalid obs
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, obs="invalid_observation")
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, obs=["obs1", "invalid_observation"])

    def test_invalid_features(self):
        # Test with invalid features
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, features="invalid_feature")
        with self.assertRaises(ValueError):
            anndata_checks(self.adata,
                           features=["feature1", "invalid_feature"])

    def test_none_input(self):
        # Test with None for all optional parameters
        self.assertIsNone(anndata_checks(self.adata,
                                         tables=None,
                                         obs=None,
                                         features=None))

    def test_valid_list_input(self):
        # Test with list input for tables, obs, and features
        self.assertIsNone(anndata_checks(self.adata,
                                         tables=["table1"],
                                         obs=["obs1"],
                                         features=["feature1"]))

    def test_valid_single_string_input(self):
        # Test with single string input for tables, obs, and features
        self.assertIsNone(anndata_checks(self.adata,
                                         tables="table1",
                                         obs="obs1",
                                         features="feature1"))

    def test_valid_string_and_list_input(self):
        # Test with mix of single string and
        # list input for tables, obs, and features
        self.assertIsNone(anndata_checks(self.adata,
                                         tables="table1",
                                         obs=["obs1"],
                                         features=["feature1"]))

    def test_valid_new_tables(self):
        # Test with valid new tables
        self.assertIsNone(anndata_checks(self.adata,
                                         tables="table1",
                                         obs=["obs1"],
                                         new_tables=["new_table1",
                                                     "new_table2"]))

    def test_invalid_new_tables(self):
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, new_tables=["new_table1",
                                                   "table2"])

    def test_valid_new_obs(self):
        # Test with valid new observations
        self.assertIsNone(anndata_checks(
            self.adata,
            new_obs=["new_obs1", "new_obs2"]))

    def test_invalid_new_obs(self):
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, new_obs=["new_obs1",
                                                "obs2"])

    def test_valid_new_features(self):
        # Test with valid new features
        self.assertIsNone(anndata_checks(
            self.adata, new_features="new_feature"))

    def test_invalid_new_features(self):
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, new_features=["feature1",
                                                     "new_feature"])

    def test_none_new_input(self):
        # Test with None for all new optional parameters
        self.assertIsNone(anndata_checks(self.adata,
                                         new_tables=None,
                                         new_obs=None,
                                         new_features=None))

    def test_wrong_adata_type(self):
        with self.assertRaises(TypeError) as context:
            anndata_checks("not an AnnData object")
        self.assertEqual(
            str(context.exception),
            "Input dataset should be an instance of anndata.AnnData, "
            "please check the input dataset source."
        )

    def test_missing_tables(self):
        with self.assertRaises(ValueError) as context:
            anndata_checks(self.adata,
                           tables=["table1",
                                   "missing_table2"])
        self.assertEqual(
            "The table 'missing_table2' "
            "does not exist in the provided dataset.\n"
            "Existing tables are:\ntable1\ntable2",
            str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            anndata_checks(self.adata,
                           tables=["missing_table1",
                                   "table2"])
        self.assertEqual(
            "The table 'missing_table1' "
            "does not exist in the provided dataset.\n"
            "Existing tables are:\ntable1\ntable2",
            str(context.exception)
        )

    def test_missing_observations(self):
        with self.assertRaises(ValueError) as context:
            anndata_checks(self.adata,
                           obs=["obs1",
                                "missing_obs2"])
        self.assertEqual(
            "The observation 'missing_obs2' does not "
            "exist in the provided dataset.\n"
            "Existing observations are:\nobs1\nobs2",
            str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            anndata_checks(self.adata,
                           obs=["missing_obs1",
                                "obs2"])
        self.assertEqual(
            "The observation 'missing_obs1' does not "
            "exist in the provided dataset.\n"
            "Existing observations are:\nobs1\nobs2",
            str(context.exception)
        )

    def test_missing_features(self):
        with self.assertRaises(ValueError) as context:
            anndata_checks(self.adata,
                           features=["feature1",
                                     "missing_feature2"])
        self.assertEqual(
            "The feature 'missing_feature2' does not "
            "exist in the provided dataset.\n"
            "Existing features are:\nfeature1\nfeature2",
            str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            anndata_checks(self.adata,
                           features=["missing_feature1",
                                     "feature2"])
        self.assertEqual(
            "The feature 'missing_feature1' does not "
            "exist in the provided dataset.\n"
            "Existing features are:\nfeature1\nfeature2",
            str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
