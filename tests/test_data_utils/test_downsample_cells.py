import unittest
import pandas as pd
from spac.data_utils import downsample_cells


class TestDownsampleCells(unittest.TestCase):

    def setUp(self):
        """Create sample dataframes for testing."""
        # Random datasets for general functionality tests
        self.df_single_random = pd.DataFrame({
            'annotations': ['annotations1'] * 50 +
                           ['annotations2'] * 30 +
                           ['annotations3'] * 20,
            'value': list(range(1, 101))
        })
        animals_random = [
            'a1', 'a1', 'a1', 'a2', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7'
        ]
        regions_random = [
            'r1', 'r1', 'r2', 'r2', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7'
        ]
        values_random = list(range(1, 11))
        self.df_multi_random = pd.DataFrame({
            'animal': animals_random,
            'region': regions_random,
            'value': values_random
        })

        # Fixed datasets for stratification frequency tests
        self.df_single_fixed = pd.DataFrame({
            'annotations': ['annotations1'] * 6 +
                           ['annotations2'] * 3 +
                           ['annotations3'],
            'value': list(range(1, 11))  # Simple values from 1 to 10
        })
        self.df_multi_fixed = pd.DataFrame({
            'animal': [
                'a1', 'a1', 'a1', 'a1', 'a1', 'a2', 'a2', 'a2', 'a3', 'a4'
            ],
            'region': [
                'r1', 'r1', 'r1', 'r2', 'r2', 'r1', 'r1', 'r2', 'r3', 'r4'
            ],
            'value': list(range(1, 11))  # Simple values from 1 to 10
        })

        self.annotations_single = 'annotations'
        self.annotations_multi = ['animal', 'region']

    def test_annotations_existence(self):
        """
        Check if the function raises a ValueError
        when a nonexistent column is passed.
        """
        with self.assertRaises(ValueError):
            downsample_cells(self.df_single_random, 'nonexistent_column')
        with self.assertRaises(ValueError):
            downsample_cells(
                self.df_multi_random,
                ['nonexistent_column', 'region']
            )

    def test_downsample_without_stratify(self):
        """Downsample without stratify"""
        n_samples = 20
        df_downsampled_single = downsample_cells(
            self.df_single_random, self.annotations_single,
            n_samples=n_samples, stratify=False
        )
        df_downsampled_multi = downsample_cells(
            self.df_multi_random, self.annotations_multi,
            n_samples=n_samples, stratify=False
        )
        self.assertTrue(
            all(df_downsampled_single.groupby(
                self.annotations_single
            ).size().values <= n_samples)
        )
        # For multi obs, group by both columns without combining them
        self.assertTrue(
            all(df_downsampled_multi.groupby(
                self.annotations_multi
            ).size().values <= n_samples)
        )

    def check_stratified_counts(
            self, downsampled_df, columns, expected_counts):
        """
        Helper method to check stratified counts for both single and
        multi annotations.
        """
        combined_col = '_'.join(columns)

        # Ensure the combined column exists in the downsampled_df
        if combined_col not in downsampled_df.columns:
            downsampled_df[combined_col] = downsampled_df[columns].apply(
                lambda row: '_'.join(row.values.astype(str)), axis=1
            )

        actual_counts = downsampled_df[combined_col].value_counts()

        for key, expected_count in expected_counts.items():
            self.assertEqual(expected_count, actual_counts.get(key, 0))

        # Remove the combined column after checks
        if combined_col in downsampled_df.columns and len(columns) > 1:
            # Only drop if it's a multi annotation
            downsampled_df.drop(columns=combined_col, inplace=True)

    def test_stratification_frequency_single_obs(self):
        """Test the frequency of stratification for single annotations."""
        n_samples = 10
        df_downsampled_stratified = downsample_cells(
            self.df_single_fixed, self.annotations_single,
            n_samples=n_samples, stratify=True, rand=True
        )
        expected_counts = {
            'annotations1': 6,
            'annotations2': 3,
            'annotations3': 1
        }
        self.check_stratified_counts(
            df_downsampled_stratified,
            [self.annotations_single],
            expected_counts
        )

    def test_stratification_frequency_multi_obs(self):
        """Test the frequency of stratification for multi annotations."""
        n_samples = 10
        df_downsampled_stratified = downsample_cells(
            self.df_multi_fixed, self.annotations_multi,
            n_samples=n_samples, stratify=True, rand=True
        )
        expected_counts = {
            'a1_r1': 3,
            'a1_r2': 2,
            'a2_r1': 2,
            'a2_r2': 1,
            'a3_r3': 1,
            'a4_r4': 1
        }
        self.check_stratified_counts(
            df_downsampled_stratified, self.annotations_multi, expected_counts
        )


if __name__ == '__main__':
    unittest.main()
