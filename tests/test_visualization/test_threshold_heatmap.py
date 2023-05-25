import anndata
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spac.visualization import threshold_heatmap


class TestThresholdHeatmap(unittest.TestCase):

    def setUp(self):

        self.obs_names = ['cell1', 'cell2', 'cell3']
        self.obs = pd.DataFrame(
            {'phenotype': ['A', 'B', 'A']},
            index=self.obs_names
            )
        self.X = np.array([[0.1, 0.3], [0.5, 0.7], [1.0, 1.5]])
        self.var_names = ['marker1', 'marker2']
        self.var = pd.DataFrame(index=self.var_names)

        self.adata = anndata.AnnData(X=self.X, obs=self.obs, var=self.var)
        self.adata.obs_names = self.obs_names
        self.adata.var_names = self.var_names

        self.marker_cutoffs = {
            'marker1': (0.2, 0.8),
            'marker2': (0.4, 1.0),
        }

        self.phenotype = 'phenotype'

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

        # Check if the figure object is returned
        self.assertIsInstance(fig, plt.Figure)

        # Check if marker_cutoffs is saved in the AnnData object
        self.assertEqual(
            self.adata.uns['marker_cutoffs'],
            self.marker_cutoffs
            )

        # Check if the feature data is saved in the AnnData object as a layer
        expected_feature_data = np.array([[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            self.adata.layers["feature"],
            expected_feature_data
            )

        # Check if the phenotype column is converted to categorical
        self.assertTrue(pd.api.types.is_categorical_dtype(
            self.adata.obs[self.phenotype])
            )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
