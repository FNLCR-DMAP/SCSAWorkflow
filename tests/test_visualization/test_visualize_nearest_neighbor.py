import unittest
import pandas as pd
import numpy as np
import anndata
import matplotlib
matplotlib.use('Agg')  # Uses a non-interactive backend for tests
import matplotlib.pyplot as plt
from spac.visualization import visualize_nearest_neighbor


class TestVisualizeNearestNeighbor(unittest.TestCase):

    def setUp(self):
        # Creates a minimal AnnData object with two cells of
        # different phenotypes
        data = np.array([[1.0], [2.0]])
        obs = pd.DataFrame(
            {
                'cell_type': ['type1', 'type2'],
                'imageid': ['img1', 'img1']
            },
            index=['CellA', 'CellB']
        )
        self.adata = anndata.AnnData(X=data, obs=obs)

        # Creates a numeric spatial_distance DataFrame
        # Each row corresponds to a cell, columns are phenotypes
        # Distances represent the nearest distance from that cell
        # to the given phenotype
        dist_df = pd.DataFrame(
            {
                'type1': [0.0, np.sqrt(2)],
                'type2': [np.sqrt(2), 0.0]
            },
            index=['CellA', 'CellB']
        )
        # Ensure numeric dtype
        dist_df = dist_df.astype(float)
        self.adata.obsm['spatial_distance'] = dist_df

    def tearDown(self):
        # Closes all figures to prevent memory issues
        plt.close('all')

    def test_missing_distance_from(self):
        """
        Tests that the function raises a ValueError when 'distance_from'
        is not provided, matching the exact error message.
        """
        expected_msg = (
            "Please specify the 'distance_from' phenotype. It indicates "
            "the reference group from which distances are measured."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            visualize_nearest_neighbor(
                adata=self.adata,
                annotation='cell_type',
                distance_from=None,
                method='numeric'
            )

    def test_invalid_method(self):
        """
        Tests that the function raises a ValueError when 'method' is invalid,
        matching the exact error message.
        """
        expected_msg = (
            "Invalid 'method'. Please choose 'numeric' or 'distribution'."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            visualize_nearest_neighbor(
                adata=self.adata,
                annotation='cell_type',
                distance_from='type1',
                method='invalid_method'
            )

    def test_simple_numeric_scenario(self):
        """
        Tests a simple numeric scenario without stratification.
        Verifies output keys and basic figure attributes.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='type1',
            distance_to='type2',
            method='numeric',
            plot_type='box'
        )

        self.assertIn('data', result)
        self.assertIn('fig', result)

        # Verifies the returned DataFrame matches expected structure
        df = result['data']
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('group', df.columns)
        self.assertIn('distance', df.columns)

        fig = result['fig']
        # Ensures there is at least one axis in the figure
        self.assertTrue(len(fig.axes) > 0)
        # Verifies axis labels
        ax = fig.axes[0]
        self.assertIn('distance', ax.get_xlabel())
        self.assertIn('group', ax.get_ylabel())

    def test_distribution_scenario(self):
        """
        Tests a distribution scenario (e.g., 'hist') without returning df.
        Verifies that a figure is produced with no errors.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='type1',
            distance_to='type2',
            method='distribution',
            plot_type='hist'
        )
        self.assertIn('data', result)
        self.assertIn('fig', result)
        df = result['data']
        fig = result['fig']
        # Data should be a DataFrame, figure should be produced
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(fig.axes) > 0)

    def test_stratify_and_facet_plot(self):
        """
        Tests scenario with stratify_by and facet_plot=True.
        Verifies multiple subplots in a single figure.
        """
        # Adds stratification column
        self.adata.obs['region'] = ['R1', 'R1']
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            stratify_by='region',
            facet_plot=True,
            method='numeric',
            plot_type='box',
            col_wrap=1
        )

        df = result['data']
        fig = result['fig']
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(fig, plt.Figure)
        # Even if only one region, it should still create a facet
        # plot structure
        self.assertTrue(len(fig.axes) >= 1)

    def test_stratify_no_facet(self):
        """
        Tests scenario with stratify_by and facet_plot=False.
        Verifies multiple figures are returned as a list when
        there are multiple groups.
        """
        # Convert CellB to type1, ensuring that after filtering by
        # distance_from='type1'both cells remain.
        self.adata.obs.at['CellB', 'cell_type'] = 'type1'

        # Adds a second region (imageid) to create multiple groups
        self.adata.obs.at['CellB', 'imageid'] = 'img2'
        # Ensures 'imageid' is categorical for proper grouping
        self.adata.obs['imageid'] = pd.Categorical(self.adata.obs['imageid'])

        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            spatial_distance='spatial_distance',
            distance_from='type1',
            stratify_by='imageid',
            facet_plot=False,
            method='distribution',
            plot_type='kde'
        )

        df = result['data']
        figs = result['fig']
        self.assertIsInstance(df, pd.DataFrame)
        # With two distinct imageids and facet_plot=False, expect
        # a list of figs
        self.assertIsInstance(figs, list)
        self.assertEqual(len(figs), 2)
        for fig in figs:
            self.assertTrue(len(fig.axes) > 0)

    def test_no_distance_to(self):
        """
        Tests scenario where distance_to is not specified.
        Verifies that the function uses all available phenotypes.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='type1',
            method='numeric',
            plot_type='box'
        )
        df = result['data']
        self.assertIsInstance(df, pd.DataFrame)
        # No distance_to means all phenotypes are considered.
        # Since type1 is distance_from, we expect type2 in 'group'
        self.assertIn('type2', df['group'].values)

    def test_log_transform(self):
        """
        Tests scenario where log transform is applied.
        Verifies that distances are log-transformed using np.log1p.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='type1',
            distance_to='type2',
            method='numeric',
            plot_type='box',
            log=True
        )

        df = result['data']
        # Original distance was sqrt(2) ~1.414
        # log1p(1.414) ~0.88 which is less than 1.414
        self.assertTrue((df['distance'] < np.sqrt(2)).all())


if __name__ == '__main__':
    unittest.main()
