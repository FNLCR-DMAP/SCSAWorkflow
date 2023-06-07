import anndata
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from spac.visualization import threshold_heatmap


class TestThresholdHeatmap(unittest.TestCase):
    def setUp(self):
        self.obs_names = ['cell1', 'cell2', 'cell3']
        self.obs = pd.DataFrame({'phenotype': ['A', 'B', 'A']},
                                index=self.obs_names)
        self.X = np.array([[0.1, 0.3], [0.5, 0.7], [1.0, 1.5]])
        self.var_names = ['marker1', 'marker2']
        self.var = pd.DataFrame(index=self.var_names)

        self.adata = anndata.AnnData(
            X=self.X, obs=self.obs, var=self.var, dtype=self.X.dtype)
        self.adata.obs_names = self.obs_names
        self.adata.var_names = self.var_names

        self.marker_cutoffs = {
            'marker1': (0.2, 0.8),
            'marker2': (0.4, 1.0),
        }

        self.phenotype = 'phenotype'

    def test_invalid_observation_type(self):
        with self.assertRaises(TypeError):
            threshold_heatmap(
                self.adata,
                self.marker_cutoffs,
                123
            )

    def test_invalid_observation_value(self):
        with self.assertRaises(ValueError):
            threshold_heatmap(
                self.adata,
                self.marker_cutoffs,
                'non_existent_column'
            )

    def test_invalid_feature_cutoffs_type(self):
        with self.assertRaises(TypeError):
            threshold_heatmap(self.adata, [], self.phenotype)

    def test_invalid_feature_cutoffs_value(self):
        feature_cutoffs = {'marker1': (1,)}  # Tuple with one element
        with self.assertRaises(ValueError):
            threshold_heatmap(self.adata, feature_cutoffs, self.phenotype)

    def test_feature_cutoffs_values_are_nan(self):
        # Test low cutoff is NaN
        feature_cutoffs_nan_low = {
            'marker1': (float('nan'), 0.8),
            'marker2': (0.4, 1.0),
        }
        with self.assertRaises(ValueError):
            threshold_heatmap(self.adata, feature_cutoffs_nan_low, self.phenotype)

        # Test high cutoff is NaN
        feature_cutoffs_nan_high = {
            'marker1': (0.2, float('nan')),
            'marker2': (0.4, 1.0),
        }
        with self.assertRaises(ValueError):
            threshold_heatmap(self.adata, feature_cutoffs_nan_high, self.phenotype)

    def test_threshold_heatmap(self):
        ax_dict = threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype
        )

        key_list = list(ax_dict.keys())

        figure_list = []
        for item in key_list:
            figure_list.append(ax_dict[item])

        fig = figure_list[0].get_figure()

        self.assertIsInstance(fig, plt.Figure)

        self.assertEqual(self.adata.uns['feature_cutoffs'],
                         self.marker_cutoffs)

        expected_intensity_data = np.array([[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            self.adata.layers["intensity"], expected_intensity_data)

        self.assertTrue(
            pd.api.types.is_categorical_dtype(self.adata.obs[self.phenotype]))

        heatmap_ax = ax_dict.get('heatmap_ax')
        self.assertEqual(len(heatmap_ax.get_yticklabels()),
                         len(self.adata.var_names))

        groupby_ax = ax_dict.get('groupby_ax')
        self.assertEqual(len(groupby_ax.get_xticklabels()),
                         len(self.adata.obs[self.phenotype].unique()))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
