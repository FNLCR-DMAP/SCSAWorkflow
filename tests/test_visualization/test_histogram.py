import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
from spac.visualization import histogram


class TestHistogram(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X = np.random.rand(100, 3)
        obs_values = ['A', 'B']
        obs_types = ['cell_type_1', 'cell_type_2']
        cell_range = [f'cell_{i}' for i in range(1, 101)]
        obs = pd.DataFrame({
            'obs1': np.random.choice(obs_values, size=100),
            'obs2': np.random.choice(obs_types, size=100),
        }, index=cell_range)
        var = pd.DataFrame(index=['marker1', 'marker2', 'marker3'])
        self.adata = anndata.AnnData(X, obs=obs, var=var)

    def test_histogram_feature_name(self):
        fig, ax = histogram(self.adata, feature_name='marker1')
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_histogram_observation_name(self):
        fig, ax = histogram(self.adata, observation_name='obs1')
        total_obs = len(self.adata.obs['obs1'])
        self.assertEqual(sum(p.get_height() for p in ax.patches), total_obs)

    def test_histogram_feature_group_by(self):
        fig, axs = histogram(
            self.adata,
            feature_name='marker1',
            group_by='obs2',
            together=False
        )
        self.assertEqual(len(axs), 2)

    def test_both_feature_and_observation(self):
        err_msg = ("Cannot pass both feature_name and "
                   "observation_name, choose one")
        with self.assertRaisesRegex(ValueError, err_msg):
            histogram(
                self.adata,
                feature_name='marker1',
                observation_name='obs1'
            )

    def test_invalid_feature_name(self):
        err_msg = "feature_name not found in adata"
        with self.assertRaisesRegex(ValueError, err_msg):
            histogram(self.adata, feature_name='invalid_marker')

    def test_invalid_observation_name(self):
        err_msg = "observation_name not found in adata"
        with self.assertRaisesRegex(ValueError, err_msg):
            histogram(self.adata, observation_name='invalid_observation')

    def test_invalid_group_by(self):
        err_msg = "group_by not found in adata"
        with self.assertRaisesRegex(ValueError, err_msg):
            histogram(
                self.adata,
                observation_name='obs1',
                group_by='invalid_group_by'
            )

    def test_ax_passed_as_argument(self):
        """
        Test case to check if the function uses the passed Axes object
        instead of creating a new one. It also checks if the function
        correctly retrieves the Figure that the Axes belongs to.
        """
        fig, ax = plt.subplots()
        returned_fig, returned_ax = histogram(
            self.adata,
            feature_name='marker1',
            ax=ax
        )
        self.assertIs(ax, returned_ax)
        self.assertIs(fig, returned_fig)


if __name__ == '__main__':
    unittest.main()
