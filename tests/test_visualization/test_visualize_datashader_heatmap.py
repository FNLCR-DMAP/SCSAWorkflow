import unittest
import numpy as np
import pandas as pd
from datashader_heatmap import visualize_2D_datashader_heatmap
import matplotlib
matplotlib.use('Agg')
 
 
class TestVisualize2DDatashaderHeatmap(unittest.TestCase):
 
    def setUp(self):
        """Prepare data for testing."""
        self.x = np.random.rand(500)
        self.y = np.random.rand(500)
        fixed_labels = (['A'] * 170) + (['B'] * 165) + (['C'] * 165)
        self.labels_categorical = pd.Categorical(fixed_labels)
 
    def test_density_heatmap_returns_fig_and_ax(self):
        """Test that density mode returns a figure and axes."""
        figure, axis = visualize_2D_datashader_heatmap(self.x, self.y)
        self.assertIsNotNone(figure)
        self.assertIsNotNone(axis)
 
    def test_density_heatmap_has_image(self):
        """Test that density mode renders an image on the axes."""
        figure, axis = visualize_2D_datashader_heatmap(self.x, self.y)
        self.assertTrue(len(axis.images) > 0)
 
    def test_density_heatmap_colorbar(self):
        """Test that a colorbar is present in density mode."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, show_colorbar=True
        )
        # colorbar adds a second axes to the figure
        self.assertTrue(len(figure.axes) > 1)
 
    def test_density_heatmap_no_colorbar(self):
        """Test that colorbar is suppressed when show_colorbar=False."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, show_colorbar=False
        )
        self.assertEqual(len(figure.axes), 1)
 
    def test_density_heatmap_log_scale(self):
        """Test that log-scale normalization is applied."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, log_scale=True
        )
        im = axis.images[0]
        self.assertIsInstance(im.norm, matplotlib.colors.LogNorm)
 
    def test_density_heatmap_linear_scale(self):
        """Test that linear normalization is used when log_scale=False."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, log_scale=False
        )
        im = axis.images[0]
        self.assertNotIsInstance(im.norm, matplotlib.colors.LogNorm)
 
    def test_custom_theme(self):
        """Test specifying a custom colormap theme."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, theme='inferno'
        )
        im = axis.images[0]
        self.assertEqual(im.cmap.name, 'inferno')
 
    def test_default_theme_is_viridis(self):
        """Test that the default colormap is viridis."""
        figure, axis = visualize_2D_datashader_heatmap(self.x, self.y)
        im = axis.images[0]
        self.assertEqual(im.cmap.name, 'viridis')
 
    def test_custom_bins(self):
        """Test specifying a custom bin count."""
        bins = 50
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, bins=bins
        )
        im = axis.images[0]
        data = im.get_array()
        self.assertEqual(data.shape, (bins, bins))
 
    def test_axis_titles(self):
        """Test if axis titles are set correctly."""
        x_title = 'Test X Axis'
        y_title = 'Test Y Axis'
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y,
            x_axis_title=x_title, y_axis_title=y_title
        )
        self.assertEqual(axis.get_xlabel(), x_title)
        self.assertEqual(axis.get_ylabel(), y_title)
 
    def test_plot_title(self):
        """Test if plot title is set correctly."""
        title = 'Test Plot Title'
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, plot_title=title
        )
        self.assertEqual(axis.get_title(), title)
 
    def test_no_plot_title_by_default(self):
        """Test that no title is set when plot_title is None."""
        figure, axis = visualize_2D_datashader_heatmap(self.x, self.y)
        self.assertEqual(axis.get_title(), '')
 
    def test_existing_axes(self):
        """Test passing an existing axes object."""
        import matplotlib.pyplot as plt
        fig, ext_ax = plt.subplots()
        ret_fig, ret_ax = visualize_2D_datashader_heatmap(
            self.x, self.y, ax=ext_ax
        )
        self.assertIs(ret_ax, ext_ax)
        self.assertIs(ret_fig, fig)
 
    def test_categorical_labels_returns_fig_and_ax(self):
        """Test that categorical mode returns a figure and axes."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, labels=self.labels_categorical
        )
        self.assertIsNotNone(figure)
        self.assertIsNotNone(axis)
 
    def test_categorical_labels_has_image(self):
        """Test that categorical mode renders an image on the axes."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, labels=self.labels_categorical
        )
        self.assertTrue(len(axis.images) > 0)
 
    def test_categorical_labels_legend(self):
        """Test that a legend is present with categorical labels."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, labels=self.labels_categorical
        )
        legend = axis.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(len(legend.get_texts()), 3)
 
    def test_categorical_labels_legend_labels(self):
        """Test that legend labels match the categories."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, labels=self.labels_categorical
        )
        legend = axis.get_legend()
        expected = sorted(['A', 'B', 'C'])
        actual = sorted(t.get_text() for t in legend.get_texts())
        self.assertEqual(actual, expected)
 
    def test_categorical_labels_no_legend(self):
        """Test that legend is suppressed when show_colorbar=False."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y,
            labels=self.labels_categorical, show_colorbar=False
        )
        self.assertIsNone(axis.get_legend())
 
    def test_categorical_color_representation(self):
        """Test that color_representation appears in the legend title."""
        color_repr = 'broad_cell_type'
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y,
            labels=self.labels_categorical,
            color_representation=color_repr
        )
        legend = axis.get_legend()
        self.assertIn(color_repr, legend.get_title().get_text())
 
    def test_categorical_legend_placement(self):
        """Test that the legend is placed outside the plot area."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, labels=self.labels_categorical
        )
        legend = axis.get_legend()
        bbox = legend.get_bbox_to_anchor().transformed(
            axis.transAxes.inverted()
        )
        self.assertTrue(bbox.x0 >= 1, "Legend is not outside the plot.")
 
    def test_fig_width_and_height(self):
        """Test custom figure dimensions via kwargs."""
        figure, axis = visualize_2D_datashader_heatmap(
            self.x, self.y, fig_width=12, fig_height=6
        )
        width, height = figure.get_size_inches()
        self.assertAlmostEqual(width, 12)
        self.assertAlmostEqual(height, 6)
 
 
if __name__ == '__main__':
    unittest.main()