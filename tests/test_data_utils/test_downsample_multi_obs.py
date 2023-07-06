import unittest
import pandas as pd
import numpy as np
from spac.data_utils import downsample_multi_obs


class TestDownsampleMultiObs(unittest.TestCase):

    def setUp(self):
        """Create a sample dataframe for testing."""
        animals = ['a1', 'a1', 'a1', 'a2', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
        regions = ['r1', 'r1', 'r2', 'r2', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7']
        values = np.random.rand(10)

        self.df = pd.DataFrame({
            'animal': animals,
            'region': regions,
            'value': values
        })
        self.observations = ['animal', 'region']

    def test_observation_existence(self):
        """
        Check if the function raises a ValueErro
        when a nonexistent column is passed.
        """
        with self.assertRaises(ValueError):
            downsample_multi_obs(self.df, ['nonexistent_column'])

    def test_stratify_not_implemented_for_multi_obs(self):
        """
        Ensure NotImplementedError is raised
        when trying to downsample without stratification
        """
        n_samples = 2
        with self.assertRaises(NotImplementedError):
            downsample_multi_obs(self.df, self.observations,
                                 n_samples=n_samples, stratify=False)

    def test_downsample_with_stratify_without_rand(self):
        """Downsample with stratify without random"""
        n_samples = 4
        df_downsampled = downsample_multi_obs(
            self.df, self.observations,
            n_samples=n_samples, stratify=True
        )
        # Ensure that the number of rows in the downsampled DataFrame
        # is greater than or equal to n_samples
        self.assertGreaterEqual(df_downsampled.shape[0], n_samples)

    def test_downsample_with_stratify_with_rand(self):
        """Downsample with stratify with random"""
        n_samples = 4
        df_downsampled = downsample_multi_obs(
            self.df, self.observations,
            n_samples=n_samples, stratify=True, rand=True
        )
        # Ensure that the number of rows in the downsampled DataFrame
        # is greater than or equal to n_samples
        self.assertGreaterEqual(df_downsampled.shape[0], n_samples)

    def test_downsample_no_n_samples(self):
        """Downsample without n_samples should return the original dataframe"""
        df_downsampled = downsample_multi_obs(self.df, self.observations)
        pd.testing.assert_frame_equal(self.df, df_downsampled)

    if __name__ == '__main__':
        unittest.main()
