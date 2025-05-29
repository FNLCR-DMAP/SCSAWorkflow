import unittest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for tests
import matplotlib.pyplot as plt
from pandas.testing import assert_frame_equal
from spac.visualization import _plot_spatial_distance_dispatch
from matplotlib.axes import Axes as MatplotlibAxes


class TestPlotSpatialDistanceDispatch(unittest.TestCase):

    def setUp(self):
        # Creates DataFrames for testing
        self.df_basic = pd.DataFrame({
            'cellid': ['C1', 'C2'],
            'group': ['g1', 'g1'],  # Single group level
            'distance': [0.5, 1.5],
            'log_distance': [0.405, 0.916], # log1p approx
            'phenotype': ['p1', 'p1'] # Single phenotype level
        })
        self.df_strat_and_hue = pd.DataFrame({
            'cellid': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
            'group': ['g1', 'g1', 'g2', 'g2', 'g1', 'g2'],
            'distance': [0.5, 1.5, 0.7, 1.2, 0.8, 1.0],
            'log_distance': [0.405, 0.916, 0.530, 0.788, 0.587, 0.693],
            'phenotype': ['p1', 'p1', 'p2', 'p2', 'p1', 'p2'],
            'region': ['R1', 'R1', 'R2', 'R2', 'R1', 'R2']
        })

    def tearDown(self):
        # Closes all figures to prevent memory issues
        plt.close('all')

    def test_simple_numeric_scenario(self):
        """
        Tests 'numeric' method, no stratify_by.
        Verifies output structure and Axes attributes.
        """
        result = _plot_spatial_distance_dispatch(
            df_long=self.df_basic,
            method='numeric',
            plot_type='box'
        )

        self.assertIn('data', result)
        self.assertIn('ax', result)
        assert_frame_equal(result['data'], self.df_basic)

        ax_obj = result['ax']
        self.assertIsInstance(ax_obj, MatplotlibAxes)
        self.assertEqual(ax_obj.get_xlabel(), "Nearest Neighbor Distance")
        self.assertEqual(ax_obj.get_ylabel(), 'group')

    def test_simple_numeric_log_distance(self):
        """Tests 'numeric' method with 'log_distance'."""
        result = _plot_spatial_distance_dispatch(
            df_long=self.df_basic,
            method='numeric',
            plot_type='box',
            distance_col='log_distance'
        )
        self.assertIn('ax', result)
        ax_obj = result['ax']
        self.assertIsInstance(ax_obj, MatplotlibAxes)
        self.assertEqual(
            ax_obj.get_xlabel(),
            "Log(Nearest Neighbor Distance)"
        )

    def test_distribution_scenario_with_explicit_hue(self):
        """
        Tests 'distribution' method with explicit hue_axis.
        Verifies output structure and Axes attributes.
        """
        result = _plot_spatial_distance_dispatch(
            df_long=self.df_strat_and_hue,
            method='distribution',
            plot_type='kde',
            hue_axis='phenotype' # 'phenotype' has multiple levels
        )

        assert_frame_equal(result['data'], self.df_strat_and_hue)
        ax_obj = result['ax']
        self.assertIsInstance(ax_obj, MatplotlibAxes)
        self.assertEqual(ax_obj.get_xlabel(), "Nearest Neighbor Distance")
        self.assertTrue(len(ax_obj.get_ylabel()) > 0) # e.g., 'Density'

    def test_stratify_and_facet_plot(self):
        """
        Tests stratify_by and facet_plot=True.
        Verifies list of Axes and their labels.
        """
        col_wrap_val = 2
        result = _plot_spatial_distance_dispatch(
            df_long=self.df_strat_and_hue,
            method='numeric',
            plot_type='violin',
            stratify_by='region', # 'region' has R1, R2 (2 unique values)
            facet_plot=True,
            col_wrap=col_wrap_val # Passed to Seaborn
        )

        assert_frame_equal(result['data'], self.df_strat_and_hue)
        ax_list = result['ax']
        self.assertIsInstance(ax_list, list)

        num_facets = self.df_strat_and_hue['region'].nunique()
        self.assertEqual(len(ax_list), num_facets)

        for i, ax_item in enumerate(ax_list):
            self.assertIsInstance(ax_item, MatplotlibAxes)
            self.assertEqual(ax_item.get_xlabel(), "Nearest Neighbor Distance")

            # Determine expected y-label based on position in the wrapped grid
            # Axes in the first column of a wrapped layout should have the y-label.
            # Others (inner columns) should have it cleared by Seaborn.
            if i % col_wrap_val == 0:
                expected_ylabel = "group"
                message = (f"Axes at index {i} (first in a wrapped row) "
                           "should have y-label 'group'")
            else:
                expected_ylabel = ""
                message = (f"Axes at index {i} (inner in a wrapped row) "
                           "should have empty y-label")
            self.assertEqual(ax_item.get_ylabel(), expected_ylabel, message)

    def test_stratify_no_facet(self):
        """
        Tests stratify_by and facet_plot=False.
        Verifies list of Axes, one per group.
        """
        result = _plot_spatial_distance_dispatch(
            df_long=self.df_strat_and_hue,
            method='distribution',
            plot_type='hist',
            stratify_by='region', # 'region' has R1, R2
            facet_plot=False,
            bins=5 # Passed to Seaborn
        )

        assert_frame_equal(result['data'], self.df_strat_and_hue)
        axes_list = result['ax']
        self.assertIsInstance(axes_list, list)
        # Expect one Axes per unique value in 'region'
        self.assertEqual(len(axes_list), self.df_strat_and_hue['region'].nunique())

        for ax_item in axes_list:
            self.assertIsInstance(ax_item, MatplotlibAxes)
            self.assertEqual(ax_item.get_xlabel(), "Nearest Neighbor Distance")
            self.assertTrue(len(ax_item.get_ylabel()) > 0) # e.g., 'Count'

    def test_no_stratify_displot_kwargs_facet(self):
        """
        Tests no stratify_by, but displot kwargs cause faceting.
        Verifies list of Axes.
        """
        # 'group' in df_strat_and_hue has 'g1', 'g2'
        result = _plot_spatial_distance_dispatch(
            df_long=self.df_strat_and_hue,
            method='distribution',
            plot_type='kde',
            # No stratify_by, facet_plot=False (default)
            # kwargs will cause faceting within _make_axes_object
            col='group'  # Facet by 'group' column using displot's 'col'
        )
        assert_frame_equal(result['data'], self.df_strat_and_hue)
        ax_list = result['ax']
        self.assertIsInstance(ax_list, list, "Expected a list of Axes due to 'col' kwarg in displot.")
        # Expect one Axes per unique value in 'group'
        self.assertEqual(len(ax_list), self.df_strat_and_hue['group'].nunique())
        for ax_item in ax_list:
            self.assertIsInstance(ax_item, MatplotlibAxes)
            self.assertEqual(ax_item.get_xlabel(), "Nearest Neighbor Distance")


    def test_invalid_method_raises_value_error(self):
        """Tests that an invalid method raises a ValueError."""
        with self.assertRaisesRegex(ValueError, "`method` must be 'numeric' or 'distribution'."):
            _plot_spatial_distance_dispatch(
                df_long=self.df_basic,
                method='invalid_method',
                plot_type='box'
            )

if __name__ == '__main__':
    unittest.main()
