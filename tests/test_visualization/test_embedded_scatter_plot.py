import unittest
from unittest.mock import patch
import anndata
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from spac.visualization import embedded_scatter_plot
matplotlib.use('Agg')


class TestStaticScatterPlot(unittest.TestCase):

    def setUp(self):
        self.adata = anndata.AnnData(X=np.random.rand(10, 10))
        self.adata.obsm['X_tsne'] = np.random.rand(10, 2)
        self.adata.obsm['X_umap'] = np.random.rand(10, 2)
        self.adata.obsm['X_pca'] = np.random.rand(10, 2)
        self.adata.obsm['sumap'] = np.random.rand(10, 2)
        self.adata.obsm['3dsumap'] = np.random.rand(10, 3)
        self.adata.obs['annotation_column'] = np.random.choice(
            ['A', 'B', 'C'], size=10
        )
        self.adata.var_names = ['gene_' + str(i) for i in range(10)]

    def test_missing_umap_coordinates(self):
        del self.adata.obsm['X_umap']
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(self.adata, 'umap')
        expected_msg = (
            "X_umap coordinates not found in adata.obsm. "
            "Please run UMAP before calling this function."
        )
        self.assertEqual(str(cm.exception), expected_msg)

    def test_missing_tsne_coordinates(self):
        del self.adata.obsm['X_tsne']
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(self.adata, 'tsne')
        expected_msg = (
            "X_tsne coordinates not found in adata.obsm. "
            "Please run TSNE before calling this function."
        )
        self.assertEqual(str(cm.exception), expected_msg)

    def test_annotation_and_feature(self):
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(
                self.adata, 'tsne',
                annotation='annotation_column',
                feature='feature_column'
            )
        expected_msg = (
            "Please specify either an annotation or a feature for coloring, "
            "not both."
        )
        self.assertEqual(str(cm.exception), expected_msg)

    def test_annotation_column(self):
        fig, ax = embedded_scatter_plot(
            self.adata, 'tsne', annotation='annotation_column'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), 't-SNE 1')
        self.assertEqual(ax.get_ylabel(), 't-SNE 2')
        self.assertEqual(ax.get_title(), 'TSNE-annotation_column')

    def test_associated_table(self):
        fig, ax = embedded_scatter_plot(
            self.adata,
            annotation='annotation_column',
            associated_table='sumap'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), 'sumap 1')
        self.assertEqual(ax.get_ylabel(), 'sumap 2')
        self.assertEqual(ax.get_title(), 'sumap-annotation_column')

    def test_feature_column(self):
        fig, ax = embedded_scatter_plot(
            self.adata, 'tsne', feature='gene_1'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), 't-SNE 1')
        self.assertEqual(ax.get_ylabel(), 't-SNE 2')
        self.assertEqual(ax.get_title(), 'TSNE-gene_1')

    def test_ax_provided(self):
        fig, ax_provided = plt.subplots()
        fig_returned, ax_returned = embedded_scatter_plot(
            self.adata, 'tsne', ax=ax_provided
        )
        self.assertIs(fig, fig_returned)
        self.assertIs(ax_provided, ax_returned)

    def test_real_tsne_plot(self):
        fig, ax = embedded_scatter_plot(
            self.adata, 'tsne', annotation='annotation_column'
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlabel(), 't-SNE 1')
        self.assertEqual(ax.get_ylabel(), 't-SNE 2')
        self.assertEqual(ax.get_title(), 'TSNE-annotation_column')

    def test_real_umap_plot(self):
        fig, ax = embedded_scatter_plot(
            self.adata, 'umap', feature='gene_1'
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlabel(), 'UMAP 1')
        self.assertEqual(ax.get_ylabel(), 'UMAP 2')
        self.assertEqual(ax.get_title(), 'UMAP-gene_1')

    def test_real_pca_plot(self):
        fig, ax = embedded_scatter_plot(
            self.adata, 'pca', annotation='annotation_column'
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlabel(), 'PCA 1')
        self.assertEqual(ax.get_ylabel(), 'PCA 2')
        self.assertEqual(ax.get_title(), 'PCA-annotation_column')

    def test_invalid_method(self):
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(self.adata, 'invalid_method')
        expected_msg = ("Method should be one of {'tsne', 'umap', 'pca',"
                        " 'spatial'}."
                        ' Got:"invalid_method"'
                        )
        self.assertEqual(str(cm.exception), expected_msg)

    def test_input_derived_feature_3d(self):
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(
                self.adata,
                associated_table='3dsumap')
        expected_msg = ('The associated table:"3dsumap" does not have'
                        ' two dimensions. It shape is:"(10, 3)"')

        self.assertEqual(str(cm.exception), expected_msg)

    def test_conflicting_kwargs(self):
        # This test ensures conflicting keys are removed from kwargs
        fig, ax = embedded_scatter_plot(
            self.adata,
            'tsne',
            annotation='annotation_column',
            x_axis_title='Conflict X',
            y_axis_title='Conflict Y',
            plot_title='Conflict Title',
            color_representation='Conflict Color'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), 't-SNE 1')
        self.assertEqual(ax.get_ylabel(), 't-SNE 2')
        self.assertEqual(ax.get_title(), 'TSNE-annotation_column')


class SpatialPlotTestCase(unittest.TestCase):
    def setUp(self):
        # Set up test data
        num_rows = 100
        num_cols = 100

        # Generate data matrix
        features = np.random.randint(0, 100, size=(num_rows, num_cols))
        feature_names = [f'Intensity_{i}' for i in range(num_cols)]

        # Generate annotation metadata
        annotation_data = {
            'annotation1': np.random.choice(['A', 'B', 'C'], size=num_rows),
            'annotation2': np.random.normal(0, 1, size=num_rows),
            'annotation3': np.random.uniform(0, 1, size=num_rows)
        }

        # Create the AnnData object
        self.adata = anndata.AnnData(
            X=features,
            obs=annotation_data
        )

        numpy_array = np.random.uniform(0, 100, size=(num_rows, 2))
        self.adata.obsm["spatial"] = numpy_array
        self.adata.var_names = feature_names

        # Generate layer data
        layer_data = np.random.randint(0, 100, size=(num_rows, num_cols))
        self.adata.layers['Normalized'] = layer_data
        self.adata.layers['Standardized'] = layer_data

        self.spot_size = 10
        self.alpha = 0.5

    def test_invalid_adata(self):
        # Test when adata is not an instance of anndata.AnnData
        with self.assertRaises(ValueError):
            embedded_scatter_plot(adata=None,
                                  method='spatial',
                                  spot_size=self.spot_size,
                                  alpha=self.alpha
                                  )

    def test_invalid_layer(self):
        # Test when layer is not a string
        with self.assertRaises(ValueError):
            embedded_scatter_plot(adata=self.adata,
                                  method='spatial',
                                  layer=123,
                                  spot_size=self.spot_size,
                                  alpha=self.alpha
                                  )

    def test_invalid_feature(self):
        # Test when feature is not a string
        with self.assertRaises(ValueError):
            embedded_scatter_plot(adata=self.adata,
                                  method='spatial',
                                  feature=123,
                                  spot_size=self.spot_size,
                                  alpha=self.alpha
                                  )

    def test_invalid_annotation(self):
        # Test when annotation is not a string
        with self.assertRaises(ValueError):
            embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                annotation=123,
                spot_size=self.spot_size,
                alpha=self.alpha
                )

    def test_invalid_spot_size(self):
        # Test when spot_size is not an integer
        with self.assertRaises(ValueError):
            embedded_scatter_plot(adata=self.adata, method='spatial',
                                  spot_size=10.5, alpha=self.alpha)

    def test_invalid_alpha(self):
        # Test when alpha is not a float
        with self.assertRaises(ValueError):
            embedded_scatter_plot(adata=self.adata, method='spatial',
                                  spot_size=self.spot_size, alpha="0.5")

    def test_invalid_alpha_range(self):
        # Test when alpha is outside the range of 0 to 1
        with self.assertRaises(ValueError):
            embedded_scatter_plot(adata=self.adata, method='spatial',
                                  spot_size=self.spot_size, alpha=-0.5)

    def test_invalid_theme(self):
        # Should raise ValueError for invalid theme
        with self.assertRaises(ValueError):
            embedded_scatter_plot(self.adata, 'umap', annotation='cat_anno',
                                  theme='not_a_theme')

    def test_missing_annotation(self):
        # Test when annotation is None and feature is None
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(adata=self.adata, method='spatial',
                                  spot_size=self.spot_size, alpha=self.alpha)
        error_msg = str(cm.exception)
        err_msg_exp = "Both annotation and feature are None, " + \
            "please provide single input."
        self.assertEqual(error_msg, err_msg_exp)

    def test_invalid_annotation_name(self):
        # Test when annotation name is not found in the dataset
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                annotation='annotation4',
                spot_size=self.spot_size,
                alpha=self.alpha
            )
        error_msg = str(cm.exception)
        err_msg_exp = ("The annotation 'annotation4' does not exist "
                       "in the provided dataset.\n"
                       "Existing annotations are:\n"
                       "annotation1\nannotation2\nannotation3"
                       )
        self.assertEqual(error_msg, err_msg_exp)

    def test_invalid_feature_name(self):
        # Test when feature name is not found in the layer
        with self.assertRaises(ValueError) as cm:
            embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                feature='feature1',
                spot_size=self.spot_size,
                alpha=self.alpha
            )
        error_msg = str(cm.exception)
        target_features = "\n".join(self.adata.var_names)
        err_msg_exp = ("The feature 'feature1' does not exist "
                       "in the provided dataset.\n"
                       f"Existing features are:\n{target_features}"
                       )
        self.assertEqual(error_msg, err_msg_exp)

    def test_spatial_plot_annotation(self):
        # This test verifies that the spatial_plot function
        # correctly interacts with the spatial function,
        # and returns the expected Axes object when
        # the annotation parameter is used.

        # Mock the spatial function
        # This mock the spatial function to replace the
        # original implementation.The mock function, mock_spatial,
        # checks if the inputs match the expected values.
        # The purpose of mocking the spatial function with the
        # mock_spatial function is to simulate the behavior of the
        # original spatial function during testing and verify if the
        # inputs passed to it match the expected values.
        def mock_spatial(
                adata,
                color,
                spot_size,
                alpha,
                ax,
                show,
                layer,
                vmin,
                vmax,
                **kwargs):
            # Assert that the inputs match the expected values
            assert layer is None
            self.assertEqual(color, 'annotation1')
            self.assertEqual(spot_size, self.spot_size)
            self.assertEqual(alpha, self.alpha)
            # self.assertEqual(vmin, None)
            assert vmax is None
            assert vmin is None
            # self.assertEqual(vmax, None)
            self.assertIsInstance(ax, plt.Axes)
            self.assertFalse(show)
            # Return a list containing the ax object to mimic
            # the behavior of the spatial function
            return [ax]

        # Mock the spatial function with the mock_spatial function
        # spatial_plot.__globals__['sc.pl.spatial'] = mock_spatial

        with patch('scanpy.pl.spatial', new=mock_spatial):
            # Create an instance of Axes
            ax = plt.Axes(
                plt.figure(),
                rect=[0, 0, 1, 1]
            )

            # Call the spatial_plot function with the ax object
            fig, returned_ax = embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                annotation='annotation1',
                layer=None,
                ax=ax,
                spot_size=self.spot_size,
                alpha=self.alpha
            )

        # Assert that the spatial_plot function returned a list
        # containing an Axes object with the same properties
        self.assertEqual(returned_ax.get_title(), ax.get_title())
        self.assertEqual(returned_ax.get_xlabel(), ax.get_xlabel())
        self.assertEqual(returned_ax.get_ylabel(), ax.get_ylabel())

        # Restore the original spatial function
        # del spatial_plot.__globals__['sc.pl.spatial']

    def test_spatial_plot_feature(self):
        # Mock the spatial function
        def mock_spatial(
                adata,
                layer,
                color,
                spot_size,
                alpha,
                vmin,
                vmax,
                ax,
                show,
                **kwargs):
            # Assert that the inputs match the expected values
            assert layer is None
            self.assertEqual(color, 'Intensity_10spatial_plot')
            self.assertEqual(spot_size, self.spot_size)
            self.assertEqual(alpha, self.alpha)
            self.assertEqual(vmin, 0)
            self.assertEqual(vmax, 100)
            self.assertIsInstance(ax, plt.Axes)
            self.assertFalse(show)
            # Return a list containing the ax object to mimic
            # the behavior of the spatial function
            return [ax]

        # Mock the spatial function with the mock_spatial function
        # spatial_plot.__globals__['sc.pl.spatial'] = mock_spatial
        with patch('scanpy.pl.spatial', new=mock_spatial):

            # Create an instance of Axes
            ax = plt.Axes(
                plt.figure(),
                rect=[0, 0, 1, 1]
            )

            # Call the spatial_plot function with the ax object
            fig, returned_ax = embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                feature='Intensity_10',
                ax=ax,
                spot_size=self.spot_size,
                alpha=self.alpha,
                vmin=0,
                vmax=100
            )

        # Assert that the spatial_plot function returned a list
        # containing an Axes object with the same properties
        self.assertEqual(returned_ax.get_title(), ax.get_title())
        self.assertEqual(returned_ax.get_xlabel(), ax.get_xlabel())
        self.assertEqual(returned_ax.get_ylabel(), ax.get_ylabel())

        # Restore the original spatial function
        # del spatial_plot.__globals__['sc.pl.spatial']

    def test_spatial_plot_combos_feature(self):
        # Define the parameter combinations to test
        spot_sizes = [10, 20]
        alphas = [0.5, 0.8]
        vmins = [-999, 0, 5]
        vmaxs = [-999, 10, 20]
        features = ['Intensity_20', 'Intensity_80']
        layers = [None, 'Standardized']

        # Generate all combinations of parameters
        # excluding both None values for features and annotations

        parameter_combinations = list(itertools.product(
            spot_sizes, alphas, vmins, vmaxs, features, layers
        ))

        parameter_combinations = [
            params for params in parameter_combinations if not (
                params[4] is None and params[5] is None
            )
        ]

        for params in parameter_combinations:
            spot_size, alpha, vmin, vmax, feature, layer = params
            print(layer)
            # Test the spatial_plot function with the
            # given parameter combination

            returned_fig, ax = embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                feature=feature,
                layer=layer,
                spot_size=spot_size,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax
            )

            # Perform assertions on the spatial plot
            # Check if ax has data plotted
            self.assertTrue(ax.has_data())

    def test_spatial_plot_combos_annotation(self):
        # Define the parameter combinations to test
        spot_sizes = [10, 20]
        alphas = [0.5, 0.8]
        vmins = [-999, 0, 5]
        vmaxs = [-999, 10, 20]
        annotation = ['annotation1', 'annotation2']
        layers = [None, 'Normalized']

        # Generate all combinations of parameters
        # excluding both None values for features and annotations

        parameter_combinations = list(itertools.product(
            spot_sizes, alphas, vmins, vmaxs, annotation, layers
        ))

        parameter_combinations = [
            params for params in parameter_combinations if not (
                params[4] is None and params[5] is None
            )
        ]

        for params in parameter_combinations:
            spot_size, alpha, vmin, vmax, annotation, layer = params
            # Test the spatial_plot function with the
            # given parameter combination

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            returned_fig, ax = embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                annotation=annotation,
                layer=layer,
                ax=ax,
                spot_size=spot_size,
                alpha=alpha,
                vmin=vmin,
                vmax=vmax
            )

            plt.close(fig)
            # Perform assertions on the spatial plot
            # Check if ax has data plotted
            self.assertTrue(ax.has_data())

    def test_color_map_from_uns(self):
        # Should use color map from adata.uns
        self.adata.uns['anno_colors'] = {'A': '#111111', 'B': '#222222',
                                         'C': '#333333', 'D': '#444444'}
        returned_fig, ax = embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                annotation='annotation1',
                color_map='anno_colors'
        )
        self.assertTrue(ax.has_data())

    def test_color_map_input(self):
        # Should raise failure as it accepts colormap as str not dict
        anno_colors = {'A': '#111111', 'B': '#222222',
                       'C': '#333333', 'D': '#444444'}
        with self.assertRaises(TypeError):
            embedded_scatter_plot(
                adata=self.adata,
                method='spatial',
                annotation='annotation1',
                color_map=anno_colors)


if __name__ == '__main__':
    unittest.main()
