import unittest
import numpy as np
import pandas as pd
import anndata as ad
from spac.utils import check_feature


class TestCheckFeature(unittest.TestCase):

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
            obs={"annotation1": [1, 2],
                 "annotation2": [3, 4]}
        )

    def test_wrong_adata_type(self):
        with self.assertRaises(TypeError) as context:
            check_feature("not an AnnData object")
        self.assertEqual(
            str(context.exception),
            "Input dataset should be an instance of anndata.AnnData, "
            "please check the input dataset source."
        )

    def test_valid_features(self):
        # Test with valid features
        self.assertIsNone(check_feature(self.adata, features="feature1"))

    def test_valid_list_features(self):
        # Test with list input for features
        self.assertIsNone(check_feature(self.adata,
                                        features=["feature1", "feature2"]))

    def test_invalid_features(self):
        # Test with invalid features
        with self.assertRaises(ValueError):
            check_feature(self.adata, features="invalid_feature")
        with self.assertRaises(ValueError):
            check_feature(self.adata, features=["feature1",
                                                "invalid_feature"])

    def test_missing_features(self):
        with self.assertRaises(ValueError) as context:
            check_feature(
                self.adata,
                features=["feature1", "missing_feature2"])
        self.assertEqual(
            "The feature 'missing_feature2' does not "
            "exist in the provided dataset.\n"
            "Existing features are:\nfeature1\nfeature2",
            str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            check_feature(
                self.adata,
                features=["missing_feature1", "feature2"])
        self.assertEqual(
            "The feature 'missing_feature1' does not "
            "exist in the provided dataset.\n"
            "Existing features are:\nfeature1\nfeature2",
            str(context.exception)
        )

    def test_valid_new_features(self):
        # Test with valid new features
        self.assertIsNone(check_feature(
                self.adata,
                features="new_feature",
                should_exist=False
            ))

    def test_invalid_new_features(self):
        with self.assertRaises(ValueError):
            check_feature(
                self.adata,
                features=["feature1", "new_feature"],
                should_exist=False
            )


if __name__ == "__main__":
    unittest.main()
