import os
import sys
import unittest
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")
from spac.data_utils import downsample_cells

class TestDownsampleCells(unittest.TestCase):

    def setUp(self):
        """Create a sample dataframe for testing."""
        self.df = pd.DataFrame({
            'observation': ['obs1'] * 50 + ['obs2'] * 30 + ['obs3'] * 20,
            'value': np.random.rand(100)
        })
        self.observation_name = 'observation'

        # Additional dataframe for functional example
        self.df_example = pd.DataFrame({
            'patient': (
                ['patient_one'] * 300 + ['patient_two'] * 200 +
                ['patient_three'] * 100 + ['patient_four'] * 50
            ),
            'value': np.random.rand(650)
        })

    def test_observation_name_existence(self):
        """
        Check if the function raises a ValueError
        when a nonexistent column is passed.
        """
        with self.assertRaises(ValueError):
            downsample_cells(self.df, 'nonexistent_column')

    def test_downsample_without_stratify(self):
        """Downsample without stratify"""
        n_samples = 20
        df_downsampled = downsample_cells(
            self.df, self.observation_name,
            n_samples=n_samples, stratify=False
        )
        self.assertTrue(
            all(df_downsampled.groupby(
                self.observation_name
            ).size().values <= n_samples)
        )

    def test_downsample_with_stratify_without_rand(self):
        """Downsample with stratify without random"""
        n_samples = 20
        df_downsampled = downsample_cells(
            self.df, self.observation_name,
            n_samples=n_samples, stratify=True
        )
        self.assertEqual(df_downsampled.shape[0], n_samples)

    def test_downsample_with_stratify_with_rand(self):
        """Downsample with stratify with random"""
        n_samples = 20
        df_downsampled = downsample_cells(
            self.df, self.observation_name,
            n_samples=n_samples, stratify=True, rand=True
        )
        self.assertEqual(df_downsampled.shape[0], n_samples)

    def test_downsample_no_n_samples(self):
        """Downsample without n_samples should return the original dataframe"""
        df_downsampled = downsample_cells(self.df, self.observation_name)
        pd.testing.assert_frame_equal(self.df, df_downsampled)

    def test_functional_example(self):
        """Functional example with patient data"""
        n_samples = 200

        # Downsampling without stratifying
        df_downsampled = downsample_cells(
            self.df_example, 'patient', n_samples=n_samples, stratify=False
        )
        self.assertTrue(
            df_downsampled['patient'].value_counts()['patient_four'] == 50
        )

        # Downsampling with stratifying
        df_downsampled_stratified = downsample_cells(
            self.df_example, 'patient',
            n_samples=n_samples, stratify=True, rand=True
        )
        self.assertTrue(df_downsampled_stratified.shape[0] <= n_samples)


if __name__ == '__main__':
    unittest.main()
