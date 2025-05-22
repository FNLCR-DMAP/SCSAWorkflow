import unittest
import pandas as pd
import numpy as np
import anndata
import matplotlib
import matplotlib.collections as mcoll
matplotlib.use('Agg')  # Uses a non-interactive backend for tests
import matplotlib.pyplot as plt
from spac.visualization import visualize_nearest_neighbor


class TestVisualizeNearestNeighbor(unittest.TestCase):

    @staticmethod
    def _create_test_adata():
        """
        Creates a common AnnData object for testing various scenarios.
        """
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

    def test_axis_labels_across_plot_types(self):
        """
        X/Y labels are correct for each plotting mode (no duplication).
        """
        scenarios = [
            # explicit plot_type
            ('numeric', 'box', 'group'),
            ('distribution', 'kde', 'Density'),
            # default plot_type (None)
            ('numeric', None, 'group'),
            ('distribution', None, 'Density'),
        ]
        for method, plot_type, expected_ylabel in scenarios:
            with self.subTest(method=method, plot_type=plot_type):
                result = visualize_nearest_neighbor(
                    adata=self.adata,
                    annotation='cell_type',
                    distance_from='typeA',
                    distance_to='typeB',
                    method=method,
                    plot_type=plot_type,
                )
                ax = (result['ax'] if method == 'numeric'
                      else result['fig'].axes[0])
                self.assertEqual(
                    ax.get_xlabel(), "Nearest Neighbor Distance"
                )
                self.assertEqual(ax.get_ylabel(), expected_ylabel)

    def test_output_structure_and_types_single_plot(self):
        """Dict keys and object types for a simple numeric plot."""
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',
            plot_type='box'
        )
        expected_keys = ['data', 'fig', 'ax', 'palette']
        for key in expected_keys:
            self.assertIn(key, result)
        self.assertIsInstance(result['data'], pd.DataFrame)
        self.assertIsInstance(result['fig'], matplotlib.figure.Figure)
        self.assertIsInstance(result['ax'], matplotlib.axes.Axes)
        self.assertIsInstance(result['palette'], dict)
        self.assertEqual(len(result['fig'].axes), 1)
        self.assertIs(result['ax'], result['fig'].axes[0])
        self.assertIs(result['ax'].figure, result['fig'])

    def test_minimal_numeric_palette(self):
        """ Palette filtered to target groups."""
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',
            plot_type='box'
        )
        self.assertIn('typeB', result['palette'])
        self.assertTrue(result['palette']['typeB'].startswith('#'))

    def test_minimal_distribution_has_axes(self):
        """Distribution plot returns one Axes."""
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='distribution',
            plot_type='kde'
        )
        # Under the default settings (no stratify, no facets), displot
        # produces a single subplot.
        self.assertEqual(len(result['fig'].axes), 1)

    def test_stratify_by_facet_plot_true_output_structure(self):
        """
        facet_plot=True returns one Figure and two Axes (img1, img2).
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

        self.assertIsInstance(result['fig'], matplotlib.figure.Figure)
        self.assertIsInstance(result['ax'], list)

        # hard-coded: the dummy AnnData has exactly 2 image_id levels
        self.assertEqual(len(result['ax']), 2)

        for ax in result['ax']:
            self.assertIs(ax.figure, result['fig'])

    def test_stratify_by_facet_plot_false_output_structure(self):
        """
        facet_plot=False returns two Figure/Axes pairs (img1, img2).
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
        # hard-coded: we expect exactly 2 categories → 2 figs / 2 axes
        self.assertEqual(len(result['fig']), 2)
        self.assertEqual(len(result['ax']), 2)

        for fig, ax in zip(result['fig'], result['ax']):
            self.assertIs(ax.figure, fig)

    def test_defined_color_map_generates_correct_palette(self):
        """ Only requested `distance_to` groups appear in palette. """
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
        self.assertEqual(result['palette'],
                         {'typeB': '#00ff00', 'typeC': 'blue'})

    # Default plot_type ­→ must be 'boxen' (numeric) or 'kde' (distribution)
    def test_default_plot_type_numeric_is_boxen(self):
        """method='numeric', plot_type=None should draw boxen patches."""
        res_numeric = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',      # plot_type defaults to 'boxen'
            plot_type=None,
        )

        ax = res_numeric['ax']

        # In current Seaborn/Matplotlib, each boxen is a PatchCollection
        patch_collections = [
            coll for coll in ax.collections
            if isinstance(coll, mcoll.PatchCollection)
        ]

        self.assertTrue(
            patch_collections,
            "Expected ≥1 PatchCollection from a boxenplot, found none.",
        )

    def test_default_plot_type_distribution_is_kde(self):
        """method='distribution', plot_type=None should draw KDE lines."""
        res_dist = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='distribution',  # plot_type defaults to 'kde'
            plot_type=None,
        )

        ax = res_dist['fig'].axes[0]     # displot returns a FacetGrid fig
        # KDE curves are plain Line2D objects
        kde_lines = [
            line for line in ax.lines
            if isinstance(line, matplotlib.lines.Line2D)
        ]
        self.assertGreater(len(kde_lines), 0, "No KDE lines found.")

    def test_legend_default_and_override(self):
        """legend=False by default; can be overridden via kwargs."""
        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to=['typeB', 'typeC'],
            method='numeric',
            plot_type='box',
        )
        self.assertEqual(len(result['fig'].legends), 0)

        result = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to=['typeB', 'typeC'],
            method='numeric',
            plot_type='box',
            legend=True,
        )
        self.assertEqual(len(result['fig'].legends), 1)

    def test_error_invalid_method(self):
        """Invalid `method` raises a clear ValueError."""
        expected_msg = ("Invalid 'method'. Please choose 'numeric' or "
                        "'distribution'.")
        with self.assertRaisesRegex(ValueError, expected_msg):
            visualize_nearest_neighbor(
                adata=self.adata,
                annotation='cell_type',
                distance_from='typeA',
                method='invalid_plot_method'
            )

    # log=True → x-axis label must be "Log(Nearest Neighbor Distance)"
    def test_log_distance_axis_label(self):
        """X-axis title switches to log scale wording when log=True."""
        res_log = visualize_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            distance_from='typeA',
            distance_to='typeB',
            method='numeric',
            plot_type='box',
            log=True,
        )

        ax = res_log['ax']            # numeric mode returns a single Axes
        self.assertEqual(
            ax.get_xlabel(),
            'Log(Nearest Neighbor Distance)',
        )


if __name__ == '__main__':
    unittest.main()
