import unittest
import pandas as pd
import anndata
import numpy as np
import matplotlib.pyplot as plt
from spac.visualization import histogram


class TestHistogram(unittest.TestCase):

    def setUp(self):
        obs = pd.DataFrame({
            'obs1': np.random.normal(size=100),
            'obs2': np.random.choice(['group1', 'group2'], size=100)
        })
        self.adata = anndata.AnnData(np.random.normal(size=(100, 10)), obs=obs)

    def test_histogram_without_groupby(self):
        ax, fig = histogram(self.adata, 'obs1')
        self.assertIsInstance(ax, plt.Axes)
        self.assertIsInstance(fig, plt.Figure)

    def test_histogram_with_groupby_with_together(self):
        ax, fig = histogram(self.adata, 'obs1', 'obs2', True)
        self.assertIsInstance(ax, plt.Axes)
        self.assertIsInstance(fig, plt.Figure)

    def test_histogram_with_groupby_without_together(self):
        axs, fig = histogram(self.adata, 'obs1', 'obs2', False)
        self.assertTrue(all(isinstance(ax, plt.Axes) for ax in axs))
        self.assertIsInstance(fig, plt.Figure)

    def test_invalid_inputs(self):
        with self.assertRaises(TypeError):
            histogram(None, 'obs1')
        with self.assertRaises(ValueError):
            histogram(self.adata, 'invalid_obs')

        self.adata.obs['obs1_nan'] = self.adata.obs['obs1'].copy()
        self.adata.obs['obs1_nan'].iloc[0] = np.nan
        with self.assertRaises(ValueError):
            histogram(self.adata, 'obs1_nan')

        with self.assertRaises(ValueError):
            histogram(self.adata, 'obs1', 'invalid_group')

        self.adata.obs['obs2_nan'] = self.adata.obs['obs2'].copy()
        self.adata.obs['obs2_nan'].iloc[0] = np.nan
        with self.assertRaises(ValueError):
            histogram(self.adata, 'obs1', 'obs2_nan')

        with self.assertRaises(TypeError):
            histogram(self.adata, 'obs1', ax='invalid_ax')

    def test_functional_example_with_without_group(self):
        """
        Test that total histogram count equals the number of observations.
        """
        observations = np.random.rand(10)
        adata = anndata.AnnData(
            obs=pd.DataFrame({'observation1': observations}))
        ax, fig = histogram(adata, 'observation1')
        self.assertEqual(sum(patch.get_height() for patch in ax.patches),
                         len(observations))

    def test_functional_example_with_group(self):
        adata = anndata.AnnData(
            obs=pd.DataFrame({
                'observation1': np.random.rand(10),
                'group_by1': ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C']
            }))
        axs, fig = histogram(adata, 'observation1', group_by='group_by1',
                             together=False)
        self.assertEqual(len(axs), 3)

    def test_ax_passed_as_argument(self):
        """
        Test if function uses passed Axes and retrieves its Figure.
        """
        fig, ax = plt.subplots()
        returned_ax, returned_fig = histogram(self.adata, 'obs1', ax=ax)
        self.assertEqual(ax, returned_ax)
        self.assertEqual(fig, returned_fig)


if __name__ == '__main__':
    unittest.main()
