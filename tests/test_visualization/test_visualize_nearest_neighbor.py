import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
import unittest
import pandas as pd
import numpy as np
import anndata
import matplotlib
matplotlib.use('Agg')  # Uses a non-interactive backend for tests
import matplotlib.pyplot as plt
from spac.visualization import visualize_nearest_neighbor


class TestVisualizeNearestNeighbor(unittest.TestCase):

    @staticmethod
    def _create_test_adata():
        """Creates a common AnnData object for testing various scenarios."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        obs = pd.DataFrame({
            'cell_type': ['typeA', 'typeB', 'typeA', 'typeC'],
            'image_id': ['img1', 'img1', 'img2', 'img2']
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        adata = anndata.AnnData(X=data, obs=obs)

        dist_df = pd.DataFrame({
            'typeA': [0.0, 1.0, 0.0, 2.0],
            'typeB': [1.0, 0.0, 4.0, 3.0],
            'typeC': [2.0, 3.0, 5.0, 0.0]
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        dist_df = dist_df.astype(float)
        adata.obsm['spatial_distance'] = dist_df
        return adata

    def setUp(self):
        """Set up basic AnnData object for tests."""
        self.adata = TestVisualizeNearestNeighbor._create_test_adata()

    def tearDown(self):
        """Close all Matplotlib figures after each test."""
        plt.close('all')

    def test_output_structure_and_types_single_plot(self):
        """
        Tests the basic output structure and types for a single plot.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',
            plot_type='box'
        )

        self.assertIsInstance(result, dict, "Result should be a dict.")
        expected_keys = ['data', 'fig', 'ax', 'palette']
        for key in expected_keys:
            self.assertIn(key, result, f"Key '{key}' missing in result.")

        self.assertIsInstance(
            result['data'], pd.DataFrame, "'data' should be a DataFrame."
        )
        self.assertIsInstance(
            result['fig'], matplotlib.figure.Figure,
            "'fig' should be a Figure."
        )
        self.assertIsInstance(
            result['ax'], matplotlib.axes.Axes,
            "'ax' should be an Axes object for a single plot."
        )
        self.assertIsInstance(
            result['palette'], dict, "'palette' should be a dictionary."
        )

        self.assertEqual(
            len(result['fig'].axes), 1,
            "Single plot figure should contain one axis."
        )
        self.assertIs(
            result['ax'], result['fig'].axes[0],
            "Returned 'ax' should be the axis in 'fig'."
        )
        self.assertIs(
            result['ax'].figure, result['fig'],
            "Returned 'ax' should belong to returned 'fig'."
        )

    def test_minimal_numeric_plot_axis_labels_and_palette(self):
        """
        Tests a minimal numeric plot, focusing on axis labels and palette.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',
            plot_type='box'
        )

        ax = result['ax']
        self.assertEqual(
            ax.get_xlabel(), "Nearest Neighbor Distance",
            "X-axis label mismatch."
        )
        self.assertEqual(
            ax.get_ylabel(), "group",
            "Y-axis label mismatch for catplot."
        )

        self.assertIn(
            'typeB', result['palette'],
            "'typeB' should be in the generated palette."
        )
        self.assertTrue(
            result['palette']['typeB'].startswith('#'),
            "Palette color should be hex."
        )

    def test_minimal_distribution_plot_axis_labels(self):
        """
        Tests a minimal distribution plot, focusing on axis labels.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='distribution',
            plot_type='kde'
        )

        self.assertTrue(
            len(result['fig'].axes) > 0,
            "Figure should have axes for displot."
        )
        ax = result['fig'].axes[0]
        self.assertEqual(
            ax.get_xlabel(), "Nearest Neighbor Distance",
            "X-axis label mismatch."
        )
        self.assertEqual(
            ax.get_ylabel(), "Density",
            "Y-axis label for KDE plot mismatch."
        )

    def test_stratify_by_facet_plot_true_output_structure(self):
        """
        Tests output structure for stratify_by with facet_plot=True.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',
            stratify_by='image_id',
            facet_plot=True
        )
        self.assertIsInstance(
            result['fig'], matplotlib.figure.Figure,
            "Should be a single Figure for facet plot."
        )
        self.assertIsInstance(
            result['ax'], list,
            "'ax' should be a list of Axes for facet plot."
        )
        expected_num_facets = self.adata.obs['image_id'].nunique()
        self.assertEqual(
            len(result['ax']), expected_num_facets,
            "Number of axes should match unique categories in stratify_by."
        )
        for ax_item in result['ax']:
            self.assertIsInstance(
                ax_item, matplotlib.axes.Axes,
                "Each item in 'ax' list should be an Axes object."
            )
            self.assertIs(
                ax_item.figure, result['fig'],
                "All facet axes should belong to the same figure."
            )

    def test_stratify_by_facet_plot_false_output_structure(self):
        """
        Tests output structure for stratify_by with facet_plot=False.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',
            stratify_by='image_id',
            facet_plot=False
        )
        num_categories = self.adata.obs['image_id'].nunique()
        self.assertIsInstance(
            result['fig'], list,
            "Should be a list of Figures for non-faceted stratified plot."
        )
        self.assertEqual(
            len(result['fig']), num_categories,
            "Number of figures should match unique categories."
        )
        for fig_item in result['fig']:
            self.assertIsInstance(
                fig_item, matplotlib.figure.Figure,
                "Each item in 'fig' list should be a Figure."
            )

        self.assertIsInstance(
            result['ax'], list, "'ax' should be a list of Axes."
        )
        self.assertEqual(
            len(result['ax']), num_categories,
            "Number of axes should match unique categories."
        )
        for i, ax_item in enumerate(result['ax']):
            self.assertIsInstance(
                ax_item, matplotlib.axes.Axes,
                "Each item in 'ax' list should be an Axes object."
            )
            self.assertIs(
                ax_item.figure, result['fig'][i],
                "Each ax should belong to its corresponding fig."
            )

    def test_defined_color_map_generates_correct_palette(self):
        """
        Tests that a defined_color_map is correctly processed.
        """
        self.adata.uns['my_colors'] = {
            'typeA': 'rgb(255,0,0)',
            'typeB': '#00FF00',
            'typeC': 'blue'
        }
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to=['typeB', 'typeC'],
            method='numeric',
            defined_color_map='my_colors'
        )
        expected_palette = {
            'typeB': '#00ff00',
            'typeC': 'blue'
        }
        self.assertEqual(
            result['palette']['typeB'], expected_palette['typeB']
        )
        self.assertEqual(
            result['palette']['typeC'], expected_palette['typeC']
        )
        self.assertNotIn(
            'typeA', result['palette'],
            "Source phenotype 'typeA' should not be in palette keys."
        )

    def test_default_plot_type_selection(self):
        """
        Tests that plot_type defaults correctly and axis labels are set.
        """
        # Test numeric default ('boxen' plot via catplot)
        res_numeric = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric'
        )
        self.assertIsInstance(
            res_numeric['fig'], matplotlib.figure.Figure,
            "Numeric default should generate a Figure object."
        )
        self.assertIsInstance(
            res_numeric['ax'], matplotlib.axes.Axes,
            "Numeric default should return an Axes object."
        )
        ax_numeric = res_numeric['ax']
        self.assertEqual(
            ax_numeric.get_xlabel(), "Nearest Neighbor Distance",
            "X-axis label for numeric default mismatch."
        )
        self.assertEqual(
            ax_numeric.get_ylabel(), "group",
            "Y-axis label for numeric default (catplot) mismatch."
        )

        # Test distribution default ('kde' plot via displot)
        res_dist = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='distribution'
        )
        self.assertIsInstance(
            res_dist['fig'], matplotlib.figure.Figure,
            "Distribution default should generate a Figure object."
        )
        self.assertTrue(
            len(res_dist['fig'].axes) > 0,
            "Distribution plot figure should have axes."
        )
        ax_dist = res_dist['fig'].axes[0]
        self.assertEqual(
            ax_dist.get_xlabel(), "Nearest Neighbor Distance",
            "X-axis label for distribution default mismatch."
        )
        self.assertEqual(
            ax_dist.get_ylabel(), "Density",
            "Y-axis label for distribution default (KDE) mismatch."
        )

    def test_legend_default_is_false_passed_to_dispatch(self):
        """
        Tests that legend=False is passed to dispatch by default.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to=['typeB', 'typeC'],
            method='numeric',
            plot_type='box'
        )
        fig = result['fig']
        self.assertEqual(
            len(fig.legends), 0,
            "Figure should not have a legend by default."
        )
        if isinstance(result['ax'], matplotlib.axes.Axes):
            self.assertIsNone(
                result['ax'].get_legend(), "Axes should not have a legend."
            )
        elif isinstance(result['ax'], list):
            for ax_item in result['ax']:
                self.assertIsNone(
                    ax_item.get_legend(),
                    "Each Axes in list should not have a legend."
                )

    def test_legend_can_be_overridden_via_kwargs(self):
        """
        Tests that the user can pass legend=True via kwargs.
        """
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to=['typeB', 'typeC'],
            method='numeric',
            plot_type='box',
            legend=True
        )
        fig = result['fig']
        self.assertEqual(
            len(fig.legends), 1,
            "Figure should have exactly one legend when legend=True."
        )
        if fig.legends: # Should be true given the assertion above
            legend_texts = [
                text.get_text() for text in fig.legends[0].get_texts()
            ]
            self.assertIn('typeB', legend_texts)
            self.assertIn('typeC', legend_texts)

    def test_error_invalid_method_in_visualize_nearest_neighbor(self):
        """
        Tests ValueError if 'method' is invalid.
        """
        expected_msg = ("Invalid 'method'. Please choose 'numeric' or "
                        "'distribution'.")
        with self.assertRaisesRegex(ValueError, expected_msg):
            visualize_nearest_neighbor(
                adata=self.adata,
                annotation='cell_type',
                distance_from='typeA',
                method='invalid_plot_method'
            )


if __name__ == '__main__':
    unittest.main()
