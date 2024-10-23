import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
from spac.visualization import histogram
mpl.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestHistogram(unittest.TestCase):
    def setUp(self):
        # Number of cells and genes
        n_cells, n_genes = 100, 3

        # Create a data matrix with fixed values: [1, 2, 3], [2, 3, 4], ...
        X = np.array([[(j+1) + i for j in range(n_genes)]
                      for i in range(n_cells)])

        annotation_values = ['A', 'B']
        annotation_types = ['cell_type_1', 'cell_type_2']
        cell_range = [f'cell_{i}' for i in range(1, n_cells+1)]

        # Create annotations with a repetitive pattern:
        # ['A', 'B', 'A', 'B', ...]
        annotation = pd.DataFrame({
            'annotation1': [annotation_values[i % 2] for i in range(n_cells)],
            'annotation2': [annotation_types[i % 2] for i in range(n_cells)],
        }, index=cell_range)

        var = pd.DataFrame(index=['marker1', 'marker2', 'marker3'])

        self.adata = anndata.AnnData(
            X.astype(np.float32), obs=annotation, var=var
        )

    def test_both_feature_and_annotation(self):
        err_msg = ("Cannot pass both feature and annotation,"
                   " choose one.")
        with self.assertRaisesRegex(ValueError, err_msg):
            histogram(
                self.adata,
                feature='marker1',
                annotation='annotation1'
            )

    def test_histogram_feature(self):
        # Define bin edges so that each bin covers exactly one integer value
        bin_edges = np.linspace(0.5, 100.5, 101)

        fig, ax = histogram(self.adata, feature='marker1', bins=bin_edges)

        # Check the histogram bars
        bars = ax.patches
        # Expecting 100 bars for 100 distinct values
        self.assertEqual(len(bars), 100)

        # Check the height and position of each bar
        for i, bar in enumerate(bars):
            # Each bar should have a height of 1
            self.assertAlmostEqual(bar.get_height(), 1)

            # Bar centers should match the values from 1 to 100
            expected_center = i + 1
            self.assertAlmostEqual(
                bar.get_x() + bar.get_width() / 2, expected_center
            )

        # Check that ax is an Axes object
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_histogram_annotation(self):
        fig, ax = histogram(self.adata, annotation='annotation1')
        total_annotation = len(self.adata.obs['annotation1'])
        # Assuming the y-axis of the histogram represents counts
        # (not frequencies or probabilities)
        self.assertEqual(sum(p.get_height() for p in ax.patches),
                         total_annotation)
        # Check that ax is an Axes object
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_histogram_feature_group_by(self):
        fig, axs = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=False
        )
        self.assertEqual(len(axs), 2)
        self.assertIsInstance(axs[0], mpl.axes.Axes)
        self.assertIsInstance(axs[1], mpl.axes.Axes)
        # Check the number of axes returned
        unique_annotations = self.adata.obs['annotation2'].nunique()
        self.assertEqual(len(axs), unique_annotations)

    def test_histogram_feature_group_by_one_label_annotation(self):

        # Setup an annotation with one value
        n_cells, n_genes = 100, 3
        # Create a data matrix with fixed values: [1, 2, 3], [2, 3, 4], ...
        X = np.array([[(j+1) + i for j in range(n_genes)]
                      for i in range(n_cells)])

        cell_range = [f'cell_{i}' for i in range(1, n_cells+1)]

        # Create annotations with one label 'A'
        annotation = pd.DataFrame({
            'annotation1': ['A' for i in range(n_cells)],
        }, index=cell_range)

        var = pd.DataFrame(index=['marker1', 'marker2', 'marker3'])

        adata = anndata.AnnData(
            X.astype(np.float32), obs=annotation, var=var
        )

        fig, axs = histogram(
            adata,
            feature='marker1',
            group_by='annotation1',
            together=False
        )
        # Since only one plot is created, axs is an Axes object
        self.assertIsInstance(axs, mpl.axes.Axes)
        # Wrap axs in a list for consistent handling
        axs = [axs]

        # Check the number of returned axes matches the unique group count
        unique_annotations = adata.obs['annotation1'].nunique()
        self.assertEqual(len(axs), unique_annotations)

    def test_log_scale(self):
        fig, ax = histogram(self.adata, feature='marker1', log_scale=True)
        self.assertTrue(ax.get_xscale() == 'log')

    def test_overlay_options(self):
        fig, ax = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=True,
            multiple="layer",
            element="step"
        )
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_layer(self):
        # Create synthetic data with known values for a subset of the data
        subset_size = 3
        synthetic_data = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])

        # Update a subset of the main matrix with different values to ensure
        # the layer is used
        self.adata.X[:subset_size, :] = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30]
        ])

        # Add the synthetic data as a layer for the subset
        self.adata.layers['layer1'] = np.zeros_like(self.adata.X)
        self.adata.layers['layer1'][:subset_size, :] = synthetic_data

        # Plot the histogram using the layer data for the subset
        fig, ax = histogram(
            self.adata[:subset_size],
            feature='marker1',
            layer='layer1',
            bins=[0.5, 1.5, 2.5, 3.5])

        # Check the histogram bars
        bars = ax.patches
        self.assertEqual(len(bars), 3)  # Expecting 3 bars for synthetic data

        # Check the height and position of each bar to match the synthetic data
        for i, value in enumerate([1, 2, 3]):
            # Each bar should have a height of 1
            self.assertAlmostEqual(bars[i].get_height(), 1)
            # Bar centers should match the synthetic values
            self.assertAlmostEqual(
                bars[i].get_x() + bars[i].get_width() / 2, value
            )

    def test_ax_passed_as_argument(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = histogram(
            self.adata,
            feature='marker1',
            ax=ax
        )
        # Check that the passed ax is the one that is returned
        self.assertEqual(id(returned_ax), id(ax))

        # Check that the passed fig is the one that is returned
        self.assertIs(fig, returned_fig)

        # Check that returned_ax is an Axes object
        self.assertIsInstance(returned_ax, mpl.axes.Axes)

    def test_default_first_feature(self):
        with self.assertWarns(UserWarning) as warning:
            bin_edges = np.linspace(0.5, 100.5, 101)
            fig, ax = histogram(self.adata, bins=bin_edges)

        warning_message = ("No feature or annotation specified. "
                           "Defaulting to the first feature: 'marker1'.")
        self.assertEqual(str(warning.warning), warning_message)

        # Check that ax is an Axes object
        self.assertIsInstance(ax, mpl.axes.Axes)

        # Check the number of bars matches the number of cells
        bars = ax.patches
        self.assertEqual(len(bars), 100)


if __name__ == '__main__':
    unittest.main()
