import unittest
import pandas as pd
from spac.data_utils import downsample_cells


class TestDownsampleCells(unittest.TestCase):

    def setUp(self):
        """Create sample dataframes for testing."""
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

        self.annotations_single = 'annotations'
        self.annotations_multi = ['animal', 'region']

    def test_annotations_existence(self):
        with self.assertRaises(ValueError):
            downsample_cells(self.df_single_random, 'nonexistent_column')
        with self.assertRaises(ValueError):
            downsample_cells(
                self.df_multi_random,
                ['nonexistent_column', 'region']
            )

    def test_downsample_single_without_stratify(self):
        n_samples = 20
        df_downsampled_single = downsample_cells(
            self.df_single_random, self.annotations_single,
            n_samples=n_samples, stratify=False
        )
        self.assertTrue(
            all(df_downsampled_single.groupby(
                self.annotations_single
            ).size().values <= n_samples)
        )

    def test_downsample_multi_without_stratify(self):
        n_samples = 20
        df_downsampled_multi = downsample_cells(
            self.df_multi_random, self.annotations_multi,
            n_samples=n_samples, stratify=False
        )
        self.assertTrue(
            all(df_downsampled_multi.groupby(
                self.annotations_multi
            ).size().values <= n_samples)
        )

    def test_group_with_zero_samples(self):
        """Test behavior when there's an extremely underrepresented group.

        In this test, 'group_small' makes up only 1% of the data. Despite this,
        the stratified downsampling should still ensure that at least one
        sample from 'group_small' is included in the downsampled dataset.
        """
        df_zero_sample = pd.DataFrame({
            'annotations': ['group_large'] * 990 + ['group_small'] * 10,
            'value': list(range(1, 1001))
        })
        n_samples = 20
        df_downsampled = downsample_cells(
            df_zero_sample, 'annotations',
            n_samples=n_samples, stratify=True
        )
        actual_counts = df_downsampled['annotations'].value_counts()
        self.assertTrue(actual_counts.get('group_small', 0), 1)

    def test_stratification_with_rounding(self):
        """Test stratification behavior with rounding.

        The distribution of groups in this test doesn't yield whole numbers
        when multiplied by the desired sample size (n_samples). This test
        ensures that the function correctly rounds the number of samples for
        each group and adjusts them if necessary to ensure the total number
        of samples matches the desired 'n_samples'. The expected outcome should
        closely match the original distribution of groups.
        """
        # Use Dictionary with Lists Directly
        data_dict = {
            'annotations': (
                ['group1'] * 487 +
                ['group2'] * 500 +
                ['group3'] * 13
            ),
            'value': list(range(1, 1001))
        }

        # Explicitly Specify Index
        df_rounding = pd.DataFrame(data_dict, index=range(1000))

        n_samples = 100
        df_downsampled = downsample_cells(
            df_rounding, 'annotations', n_samples=n_samples, stratify=True
        )
        self.assertEqual(len(df_downsampled), n_samples)
        counts = df_downsampled['annotations'].value_counts()
        # Expected to be 50% of n_samples
        self.assertEqual(counts['group2'], 50)
        # Expected to be rounded to 49 from 48.7% of n_samples
        self.assertEqual(counts['group1'], 49)
        # Expected to be rounded to 1 from 1.3% of n_samples
        self.assertEqual(counts['group3'], 1)

    def test_stratification_frequency_single_obs_large_dataset(self):
        """Test the frequency of stratification for single annotations
        with a larger dataset."""
        n_samples = 60  # Set a smaller number to force downsampling
        df_downsampled_stratified = downsample_cells(
            self.df_single_random, self.annotations_single,
            n_samples=n_samples, stratify=True, rand=True
        )

        # Expected counts based on stratified sampling
        # annotations1: round(50/total_cells * n_samples)
        # annotations2: round(30/total_cells * n_samples)
        # annotations3: round(20/total_cells * n_samples)
        expected_counts = {
            'annotations1': 30,  # 50% of total observations
            'annotations2': 18,  # 30% of total observations
            'annotations3': 12   # 20% of total observations
        }

        actual_counts = df_downsampled_stratified[
            self.annotations_single
        ].value_counts()
        for key, expected_count in expected_counts.items():
            self.assertEqual(expected_count, actual_counts.get(key, 0))

    def test_downsampling_effect_multi_obs(self):
        # Create a larger dataset
        df_large = pd.DataFrame({
            'animal': (
                ['a1'] * 400 + ['a2'] * 300 +
                ['a3'] * 200 + ['a4'] * 100
            ),
            'region': ['r1'] * 500 + ['r2'] * 300 + ['r3'] * 200,
            'value': list(range(1, 1001))
        })
        n_samples = 100
        df_downsampled_stratified = downsample_cells(
            df_large, ['animal', 'region'],
            n_samples=n_samples, stratify=True, rand=False,
            combined_col_name='_combined_'
        )

        """
        Expected counts based on stratified sampling:
        The chosen groups in expected_counts reflect dominant combinations.
        E.g., 'a1_r1' is significant due to high frequencies of 'a1' and 'r1'.
        'a2_r2' is dominant due to high occurrences of 'a2' and 'r2'.
        'a3_r2' and 'a3_r3' are expected: 'a3' is frequent, and 'r2', 'r3' are
        dominant. 'a4_r3' is due to 'a4' intersecting with dominant 'r3'.
        Other combinations might exist but are less frequent in the original
        data. Values for each group are scaled down to the desired sample size
        (n_samples).
        """
        expected_counts = {
            'a1_r1': 40,
            'a2_r2': 20,
            'a2_r1': 10,
            'a3_r2': 10,
            'a3_r3': 10,
            'a4_r3': 10
        }

        actual_counts = (
            df_downsampled_stratified['_combined_'].value_counts()
            .to_dict()
        )
        self.assertDictEqual(expected_counts, actual_counts)

    def test_anndata_input(self):
        """
        Test that downsample_cells accepts anndata objects as input,
        returns an anndata object and performs downsampling correctly,
        retaining features in .X and annotations in .obs.
        """
    
        # create anndata object
        X_data = pd.DataFrame({
            'feature1': [1, 3, 5, 7, 9, 12, 14, 16],
            'feature2': [2, 4, 6, 8, 10, 13, 15, 18],
            'feature3': [3, 5, 7, 9, 11, 14, 16, 19]
        })

        obs_data = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2',
                'phenotype3',
                'phenotype3',
                'phenotype4',
                'phenotype4'
            ]
        })

        anndata_obj = anndata.AnnData(X = X_data, obs = obs_data)
            
        # call downsample on the anndata object
        downsampled_adata = downsample_cells(
            input_data = anndata_obj,
            annotations = 'phenotype',
            n_samples = 1,
            stratify = False,
            rand = True,
            combined_col_name= '_combined_'
        )
    
        # confirm the downsampled_df is an anndata object
        self.assertTrue(isinstance(downsampled_adata, anndata.AnnData))

        # confirm number of samples after downsampling is correct 
        # (four groups with one sample each is four rows total)
        self.assertEqual(downsampled_adata.shape[0], 4)		

        # confirm the number of groups (phenotypes) is still four 
        self.assertEqual(downsampled_adata.obs['phenotype'].nunique(), 4)

        # confirm original annotation column is present
        self.assertIn('phenotype', downsampled_adata.obs.columns)
        
        # confirm feature columns are present in .var_names
        expected_features = X_data.columns.tolist()
        for feature in expected_features:
            self.assertIn(feature, downsampled_adata.var_names)

if __name__ == '__main__':
    unittest.main()
