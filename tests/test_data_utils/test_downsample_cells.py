import unittest
import pandas as pd
import numpy as np
from spac.data_utils import downsample_cells


class TestDownsampleCells(unittest.TestCase):

    def setUp(self):
        """Create a sample dataframe for testing."""
        self.df = pd.DataFrame({
            'annotation': ['annotation1'] * 50 +
                          ['annotation2'] * 30 +
                          ['annotation3'] * 20,
            'value': np.random.rand(100)
        })
        self.annotation = 'annotation'

    def test_annotation_existence(self):
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
            self.df, self.annotation,
            n_samples=n_samples, stratify=False
        )
        self.assertTrue(
            all(df_downsampled.groupby(
                self.annotation
            ).size().values <= n_samples)
        )

    def test_downsample_with_stratify_without_rand(self):
        """Downsample with stratify without random"""
        n_samples = 20
        df_downsampled = downsample_cells(
            self.df, self.annotation,
            n_samples=n_samples, stratify=True
        )
        self.assertEqual(df_downsampled.shape[0], n_samples)

    def test_downsample_with_stratify_with_rand(self):
        """Downsample with stratify with random"""
        n_samples = 20
        df_downsampled = downsample_cells(
            self.df, self.annotation,
            n_samples=n_samples, stratify=True, rand=True
        )
        self.assertEqual(df_downsampled.shape[0], n_samples)

    def test_downsample_no_n_samples(self):
        """Downsample without n_samples should return the original dataframe"""
        df_downsampled = downsample_cells(self.df, self.annotation)
        pd.testing.assert_frame_equal(self.df, df_downsampled)

    def test_stratification_frequency(self):
        """Test the frequency of stratification"""
        n_samples = 10

        # Downsampling with stratifying
        df_downsampled_stratified = downsample_cells(
            self.df, self.annotation,
            n_samples=n_samples, stratify=True, rand=True
        )

        # Assert the total number of samples
        num_rows = df_downsampled_stratified.shape[0]
        self.assertEqual(
            num_rows, n_samples,
            'Number of rows in df_downsampled_stratified does not match '
            'n_samples'
        )

        # Assert the number of samples in each annotation
        annotation = 'annotation'
        annotation_value_counts = \
            df_downsampled_stratified[annotation].value_counts()
        self.assertEqual(
            annotation_value_counts['annotation1'], 5,
            'Number of samples in annotation1 does not match expected count'
        )
        self.assertEqual(
            annotation_value_counts['annotation2'], 3,
            'Number of samples in annotation2 does not match expected count'
        )
        self.assertEqual(
            annotation_value_counts['annotation3'], 2,
            'Number of samples in annotation3 does not match expected count'
        )


if __name__ == '__main__':
    unittest.main()
