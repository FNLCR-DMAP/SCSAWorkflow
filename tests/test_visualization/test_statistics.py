import unittest
import pandas as pd
import numpy as np
from scipy import stats
from spac.visualization import compute_pairwise_stats_multi 

class TestComputePairwiseStatsMulti(unittest.TestCase):

    def setUp(self):
        """Prepare sample data used for multiple tests."""
        self.df = pd.DataFrame({
            'group': ['A'] * 5 + ['B'] * 5 + ['C'] * 5,
            'feature1': [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13],
            'feature2': [2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14]
        })
        self.pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        self.value_cols = ['feature1', 'feature2']

    def test_basic_functionality(self):
        """Test basic functionality with t-test and no correction."""
        result = compute_pairwise_stats_multi(
            df=self.df,
            group_col='group',
            value_cols=self.value_cols,
            pairs=self.pairs,
            test='t-test_ind',
            comparisons_correction=None
        )
        self.assertEqual(result.shape[0], len(self.pairs) * len(self.value_cols))
        required_columns = {'Feature', 'Group 1', 'Group 2', 'Test', 'Test Statistic', 'p-value', 'Corrected p-value', 'Significance'}
        self.assertTrue(required_columns.issubset(set(result.columns)))

    def test_with_multiple_comparisons_correction(self):
        """Test functionality with multiple comparisons correction applied."""
        result = compute_pairwise_stats_multi(
            df=self.df,
            group_col='group',
            value_cols=self.value_cols,
            pairs=self.pairs,
            test='t-test_ind',
            comparisons_correction='bonferroni'
        )
        self.assertTrue((result['Corrected p-value'] <= 1).all())

    def test_invalid_test_name_raises(self):
        """Test that invalid test name raises ValueError."""
        with self.assertRaises(ValueError):
            compute_pairwise_stats_multi(
                df=self.df,
                group_col='group',
                value_cols=['feature1'],
                pairs=self.pairs,
                test='invalid_test',
                comparisons_correction=None
            )

    def test_missing_group_raises(self):
        """Test that a missing group name raises KeyError."""
        invalid_pairs = [('A', 'D')]  # 'D' does not exist
        with self.assertRaises(KeyError):
            compute_pairwise_stats_multi(
                df=self.df,
                group_col='group',
                value_cols=['feature1'],
                pairs=invalid_pairs,
                test='t-test_ind',
                comparisons_correction=None
            )

    def test_wilcoxon_with_unequal_sample_sizes(self):
        """Test Wilcoxon test with unequal group sizes returns NaNs."""
        df = pd.DataFrame({
            'group': ['A'] * 3 + ['B'] * 5,
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        result = compute_pairwise_stats_multi(
            df=df,
            group_col='group',
            value_cols=['feature1'],
            pairs=[('A', 'B')],
            test='wilcoxon',
            comparisons_correction=None
        )
        self.assertTrue(np.isnan(result['Test Statistic'].iloc[0]))
        self.assertTrue(np.isnan(result['p-value'].iloc[0]))

    def test_mannwhitneyu(self):
        """Test Mann-Whitney U test returns valid results."""
        result = compute_pairwise_stats_multi(
            df=self.df,
            group_col='group',
            value_cols=self.value_cols,
            pairs=self.pairs,
            test='mannwhitneyu',
            comparisons_correction=None
        )
        self.assertTrue((result['Test'] == 'mannwhitneyu').all())
        self.assertTrue(result['p-value'].between(0, 1).all())

    def test_wilcoxon_equal_sample_sizes(self):
        """Test Wilcoxon test with equal group sizes returns valid results."""
        df = pd.DataFrame({
            'group': ['A'] * 4 + ['B'] * 4,
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8]
        })
        result = compute_pairwise_stats_multi(
            df=df,
            group_col='group',
            value_cols=['feature1'],
            pairs=[('A', 'B')],
            test='wilcoxon',
            comparisons_correction=None
        )
        self.assertFalse(np.isnan(result['Test Statistic'].iloc[0]))
        self.assertFalse(np.isnan(result['p-value'].iloc[0]))

    def test_significance_annotation(self):
        """Test that significance stars are assigned correctly."""
        df = pd.DataFrame({
            'group': ['A'] * 10 + ['B'] * 10,
            'feature1': [1]*10 + [100]*10
        })
        result = compute_pairwise_stats_multi(
            df=df,
            group_col='group',
            value_cols=['feature1'],
            pairs=[('A', 'B')],
            test='t-test_ind',
            comparisons_correction=None
        )
        self.assertIn(result['Significance'].iloc[0], ['***', '**', '*', 'ns'])

    def test_nan_in_features(self):
        """Test that NaN values in features are handled correctly."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'feature1': [1, np.nan, 2, 3]
        })
        result = compute_pairwise_stats_multi(
            df=df,
            group_col='group',
            value_cols=['feature1'],
            pairs=[('A', 'B')],
            test='t-test_ind',
            comparisons_correction=None
        )
        self.assertEqual(result.shape[0], 1)
        self.assertIn('p-value', result.columns)

    def test_multiple_testing_correction_with_nan(self):
        """Test multiple testing correction works with some NaN p-values."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'feature1': [1, np.nan, 2, 3],
            'feature2': [np.nan, np.nan, np.nan, np.nan]
        })
        result = compute_pairwise_stats_multi(
            df=df,
            group_col='group',
            value_cols=['feature1', 'feature2'],
            pairs=[('A', 'B')],
            test='t-test_ind',
            comparisons_correction='bonferroni'
        )
        self.assertTrue((result['Corrected p-value'] <= 1).all())
        self.assertEqual(result.shape[0], 2)


if __name__ == '__main__':
    unittest.main()
