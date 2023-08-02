import unittest
import numpy as np
import pandas as pd
import anndata as ad
from spac.utils import check_obs


class TestCheckObs(unittest.TestCase):

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
            check_obs("not an AnnData object")
        self.assertEqual(
            str(context.exception),
            "Input dataset should be an instance of anndata.AnnData, "
            "please check the input dataset source."
        )

    def test_valid_obs(self):
        # Test with valid obs
        self.assertIsNone(check_obs(self.adata, observations="obs1"))

    def test_valid_list_obs(self):
        # Test with list input for obs
        self.assertIsNone(check_obs(self.adata, observations=["obs1", "obs2"]))

    def test_invalid_obs(self):
        # Test with invalid obs
        with self.assertRaises(ValueError):
            check_obs(self.adata, observations="invalid_observation")
        with self.assertRaises(ValueError):
            check_obs(self.adata, observations=["obs1", "invalid_observation"])

    def test_missing_observations(self):
        with self.assertRaises(ValueError) as context:
            check_obs(
                self.adata,
                observations=["obs1", "missing_obs2"])
        self.assertEqual(
            "The observation 'missing_obs2' does not "
            "exist in the provided dataset.\n"
            "Existing observations are:\nobs1\nobs2",
            str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            check_obs(
                self.adata,
                observations=["missing_obs1", "obs2"])
        self.assertEqual(
            "The observation 'missing_obs1' does not "
            "exist in the provided dataset.\n"
            "Existing observations are:\nobs1\nobs2",
            str(context.exception)
        )

    def test_valid_new_obs(self):
        # Test with valid new observations
        self.assertIsNone(check_obs(
            self.adata,
            observations=["new_obs1", "new_obs2"],
            should_exist=False))

    def test_invalid_new_obs(self):
        with self.assertRaises(ValueError):
            check_obs(
                self.adata,
                observations=["new_obs1", "obs2"],
                should_exist=False)


if __name__ == "__main__":
    unittest.main()
