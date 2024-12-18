import unittest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for tests
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal
from spac.visualization import _plot_spatial_distance_dispatch


class TestPlotSpatialDistanceDispatch(unittest.TestCase):

    def setUp(self):
        # Creates a minimal DataFrame for testing
        self.df_basic = pd.DataFrame({
            'cellid': ['C1', 'C2'],
            'group': ['g1', 'g1'],
            'distance': [0.5, 1.5],
            'phenotype': ['p1', 'p1']
        })

    def tearDown(self):
        # Closes all figures to prevent memory issues
        plt.close('all')

    def test_invalid_method_error(self):
        """
        Tests that providing an invalid 'method' raises the correct ValueError.
        """
        expected_msg = "`method` must be 'numeric' or 'distribution'."
        with self.assertRaisesRegex(ValueError, expected_msg):
            _plot_spatial_distance_dispatch(
                df_long=self.df_basic,
                method='invalid_method',
                plot_type='box'
            )

    def test_simple_numeric_scenario(self):
        """
        Tests a simple scenario with 'numeric' method and no stratify_by.
        Verifies output structure and basic figure attributes.
        """
        result = _plot_spatial_distance_dispatch(
            df_long=self.df_basic,
            method='numeric',
            plot_type='box'
        )

        # Checks result structure
        self.assertIn('data', result)
        self.assertIn('fig', result)

        # Verifies the data matches the input DataFrame
        assert_frame_equal(result['data'], self.df_basic)

        # Verifies figure properties
        fig = result['fig']
        # Ensures there is at least one axis in the figure
        self.assertTrue(len(fig.axes) > 0)

        # Check axis labels
        ax = fig.axes[0]
        self.assertIn('distance', ax.get_xlabel())
        self.assertIn('group', ax.get_ylabel())

    def test_distribution_scenario_with_hue(self):
        """
        Tests the 'distribution' method with a hue axis.
        Verifies output structure and figure attributes.
        """
        df_hue = self.df_basic.copy()
        df_hue['phenotype'] = ['p1', 'p2']

        result = _plot_spatial_distance_dispatch(
            df_long=df_hue,
            method='distribution',
            plot_type='kde',
            hue_axis='phenotype'
        )

        # Verifies the data matches the input DataFrame
        assert_frame_equal(result['data'], df_hue)
        fig = result['fig']
        self.assertTrue(len(fig.axes) > 0)

    def test_stratify_and_facet_plot(self):
        """
        Tests the scenario with stratify_by and facet_plot=True.
        Verifies the presence of multiple subplots in a single figure.
        """
        df_strat = pd.DataFrame({
            'cellid': ['C1', 'C2', 'C3', 'C4'],
            'group': ['g1', 'g1', 'g2', 'g2'],
            'distance': [0.5, 1.5, 0.7, 1.2],
            'phenotype': ['p1', 'p1', 'p2', 'p2'],
            'region': ['R1', 'R1', 'R2', 'R2']
        })
        result = _plot_spatial_distance_dispatch(
            df_long=df_strat,
            method='numeric',
            plot_type='violin',
            stratify_by='region',
            facet_plot=True,
            col_wrap=2
        )

        assert_frame_equal(result['data'], df_strat)
        fig = result['fig']
        # Verifies the expected number of subplots in the figure
        self.assertEqual(len(fig.axes), 2)

    def test_stratify_no_facet(self):
        """
        Tests the scenario with stratify_by and facet_plot=False.
        Verifies the presence of multiple figures, one per group.
        """
        df_strat = pd.DataFrame({
            'cellid': ['C1', 'C2', 'C3', 'C4'],
            'group': ['g1', 'g1', 'g2', 'g2'],
            'distance': [0.5, 1.5, 0.7, 1.2],
            'phenotype': ['p1', 'p1', 'p2', 'p2'],
            'region': ['R1', 'R1', 'R2', 'R2']
        })
        result = _plot_spatial_distance_dispatch(
            df_long=df_strat,
            method='distribution',
            plot_type='hist',
            stratify_by='region',
            facet_plot=False,
            bins=5
        )

        assert_frame_equal(result['data'], df_strat)
        figs = result['fig']

        # Verifies the expected number of figures
        self.assertIsInstance(figs, list)
        self.assertEqual(len(figs), 2)  # R1 and R2

        # Verifies each figure has at least one axis
        for fig in figs:
            self.assertTrue(len(fig.axes) > 0)


if __name__ == '__main__':
    unittest.main()
