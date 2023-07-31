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
            layers={"layer1": np.array([[5, 6], [7, 8]]),
                    "layer2": np.array([[9, 10], [11, 12]])},
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

    def test_invalid_layers(self):
        # Test with invalid layers
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, layers="invalid_layer")
        with self.assertRaises(ValueError):
            anndata_checks(self.adata, layers=["layer1", "invalid_layer"])

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
                                         layers=None,
                                         obs=None,
                                         features=None))

    def test_valid_list_input(self):
        # Test with list input for layers, obs, and features
        self.assertIsNone(anndata_checks(self.adata,
                                         layers=["layer1"],
                                         obs=["obs1"],
                                         features=["feature1"]))

    def test_valid_single_string_input(self):
        # Test with single string input for layers, obs, and features
        self.assertIsNone(anndata_checks(self.adata,
                                         layers="layer1",
                                         obs="obs1",
                                         features="feature1"))

    def test_valid_string_and_list_input(self):
        # Test with mix of single string and
        # list input for layers, obs, and features
        self.assertIsNone(anndata_checks(self.adata,
                                         layers="layer1",
                                         obs=["obs1"],
                                         features=["feature1"]))


if __name__ == "__main__":
    unittest.main()
