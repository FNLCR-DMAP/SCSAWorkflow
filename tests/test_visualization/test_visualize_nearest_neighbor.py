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
        self.assertEqual(ax.get_xlabel(), "Nearest Neighbor Distance")
        self.assertIn('group', ax.get_ylabel())

    def test_visualize_with_log_distance(self):
        """
        Test that visualize_nearest_neighbor correctly handles log-transformed
        distances and uses the 'log_distance' column in the output.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='type1',
            spatial_distance='spatial_distance',
            log=True,
            method='numeric',
            plot_type='box'
        )

        df_long = result['data']
        fig = result['fig']

        # Ensure 'log_distance' is used
        self.assertIn('log_distance', df_long.columns)
        self.assertNotIn('distance', df_long.columns)

        # Validate the plot uses the correct label for log-transformed distance
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Log(Nearest Neighbor Distance)")


if __name__ == '__main__':
    unittest.main()
