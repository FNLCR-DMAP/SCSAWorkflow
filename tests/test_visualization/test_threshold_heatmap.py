import anndata
import unittest
import numpy as np
import pandas as pd
import matplotlib
from spac.visualization import threshold_heatmap
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestThresholdHeatmap(unittest.TestCase):
    def setUp(self):
        self.annotation_names = ['cell1', 'cell2', 'cell3']
        self.annotation = pd.DataFrame({'phenotype': ['A', 'B', 'A']},
                                       index=self.annotation_names)
        self.X = np.array([[0.1, 0.3], [0.5, 0.7], [1.0, 1.5]])
        self.var_names = ['marker1', 'marker2']
        self.var = pd.DataFrame(index=self.var_names)

        self.adata = anndata.AnnData(
            X=self.X, obs=self.annotation, var=self.var, dtype=self.X.dtype)
        self.adata.obs_names = self.annotation_names
        self.adata.var_names = self.var_names

        self.marker_cutoffs = {
            'marker1': (0.2, 0.8),
            'marker2': (0.4, 1.0),
        }

        self.phenotype = 'phenotype'

    def test_feature_cutoffs_values_are_nan(self):
        # Test low cutoff is NaN
        feature_cutoffs_nan_low = {
            'marker1': (float('nan'), 0.8),
            'marker2': (0.4, 1.0),
        }
        with self.assertRaises(ValueError):
            threshold_heatmap(
                self.adata,
                feature_cutoffs_nan_low,
                self.phenotype
            )

        # Test high cutoff is NaN
        feature_cutoffs_nan_high = {
            'marker1': (0.2, float('nan')),
            'marker2': (0.4, 1.0),
        }
        with self.assertRaises(ValueError):
            threshold_heatmap(
                self.adata,
                feature_cutoffs_nan_high,
                self.phenotype
            )

    def test_threshold_heatmap(self):
        heatmap_plot = threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype
        )

        self.assertEqual(
            self.adata.uns['feature_cutoffs'], self.marker_cutoffs
        )

        expected_intensity_data = np.array([[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            self.adata.layers["intensity"], expected_intensity_data
        )

        # Ensure that 'heatmap_ax' and 'groupby_ax' are in the returned
        # dictionary
        self.assertIn('heatmap_ax', heatmap_plot)
        self.assertIn('groupby_ax', heatmap_plot)

    def test_swap_axes(self):
        # Generate heatmap without swapping axe
        heatmap_plot = threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype,
            swap_axes=False
        )
        heatmap_ax = heatmap_plot['heatmap_ax']
        x_labels_before = heatmap_ax.get_xticks()
        y_labels_before = heatmap_ax.get_yticks()

        # Generate heatmap with swapping axes
        heatmap_plot_swapped = threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype,
            swap_axes=True
        )
        heatmap_ax_swapped = heatmap_plot_swapped['heatmap_ax']
        x_labels_after = heatmap_ax_swapped.get_xticks()
        y_labels_after = heatmap_ax_swapped.get_yticks()

        # Check that the axes have been swapped
        self.assertEqual(len(x_labels_before), len(y_labels_after))
        self.assertEqual(len(y_labels_before), len(x_labels_after))

    def test_threshold_heatmap_with_layer(self):
        # Add a new layer "new_layer" to adata for this specific test
        new_layer_data = np.array([[0.2, 0.4], [0.6, 0.8], [1.1, 1.6]])
        self.adata.layers["new_layer"] = new_layer_data

        # Using the new layer "new_layer"
        threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype,
            layer="new_layer"
        )

        # Check if the intensities are set correctly based on the new layer
        expected_intensity_data = np.array([[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            self.adata.layers["intensity"], expected_intensity_data
        )

    def test_heatmap_axes_and_colorbar(self):
        # Generate the heatmap
        heatmap_plot = threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype
        )

        # Check if 'heatmap_ax' is in the returned dictionary
        self.assertIn('heatmap_ax', heatmap_plot)

        # Check if 'groupby_ax' is in the returned dictionary
        self.assertIn('groupby_ax', heatmap_plot)

        # Set the expected ticks and labels
        new_ticks = [0, 1, 2]
        new_labels = ['Low', 'Medium', 'High']

        # Find the colorbar by iterating through the figure's axes and their
        # children
        colorbar = None
        for ax in plt.gcf().axes:
            for child in ax.get_children():
                if hasattr(child, 'colorbar') and child.colorbar is not None:
                    colorbar = child.colorbar
                    break
            if colorbar is not None:
                break

        # Ensure a colorbar was found
        self.assertIsNot(colorbar, None, "No colorbar found in the plot.")

        # Check the ticks and labels of the colorbar
        self.assertListEqual(colorbar.get_ticks().tolist(), new_ticks)
        self.assertListEqual(
            [tick.get_text() for tick in colorbar.ax.get_yticklabels()],
            new_labels
        )


if __name__ == '__main__':
    unittest.main()
