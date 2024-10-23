import unittest
import anndata
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spac.visualization import dimensionality_reduction_plot
matplotlib.use('Agg')


class TestDimensionalityReductionPlot(unittest.TestCase):

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
            dimensionality_reduction_plot(self.adata, 'umap')
        expected_msg = (
            "X_umap coordinates not found in adata.obsm. "
            "Please run UMAP before calling this function."
        )
        self.assertEqual(str(cm.exception), expected_msg)

    def test_missing_tsne_coordinates(self):
        del self.adata.obsm['X_tsne']
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(self.adata, 'tsne')
        expected_msg = (
            "X_tsne coordinates not found in adata.obsm. "
            "Please run TSNE before calling this function."
        )
        self.assertEqual(str(cm.exception), expected_msg)

    def test_annotation_and_feature(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(
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
        fig, ax = dimensionality_reduction_plot(
            self.adata, 'tsne', annotation='annotation_column'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), 't-SNE 1')
        self.assertEqual(ax.get_ylabel(), 't-SNE 2')
        self.assertEqual(ax.get_title(), 'TSNE-annotation_column')

    def test_associated_table(self):
        fig, ax = dimensionality_reduction_plot(
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
        fig, ax = dimensionality_reduction_plot(
            self.adata, 'tsne', feature='gene_1'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_xlabel(), 't-SNE 1')
        self.assertEqual(ax.get_ylabel(), 't-SNE 2')
        self.assertEqual(ax.get_title(), 'TSNE-gene_1')

    def test_ax_provided(self):
        fig, ax_provided = plt.subplots()
        fig_returned, ax_returned = dimensionality_reduction_plot(
            self.adata, 'tsne', ax=ax_provided
        )
        self.assertIs(fig, fig_returned)
        self.assertIs(ax_provided, ax_returned)

    def test_real_tsne_plot(self):
        fig, ax = dimensionality_reduction_plot(
            self.adata, 'tsne', annotation='annotation_column'
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlabel(), 't-SNE 1')
        self.assertEqual(ax.get_ylabel(), 't-SNE 2')
        self.assertEqual(ax.get_title(), 'TSNE-annotation_column')

    def test_real_umap_plot(self):
        fig, ax = dimensionality_reduction_plot(
            self.adata, 'umap', feature='gene_1'
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlabel(), 'UMAP 1')
        self.assertEqual(ax.get_ylabel(), 'UMAP 2')
        self.assertEqual(ax.get_title(), 'UMAP-gene_1')

    def test_real_pca_plot(self):
        fig, ax = dimensionality_reduction_plot(
            self.adata, 'pca', annotation='annotation_column'
        )
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertEqual(ax.get_xlabel(), 'PCA 1')
        self.assertEqual(ax.get_ylabel(), 'PCA 2')
        self.assertEqual(ax.get_title(), 'PCA-annotation_column')

    def test_invalid_method(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(self.adata, 'invalid_method')
        expected_msg = ("Method should be one of {'tsne', 'umap', 'pca'}."
                        ' Got:"invalid_method"')
        self.assertEqual(str(cm.exception), expected_msg)

    def test_input_derived_feature_3d(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(
                self.adata,
                associated_table='3dsumap')
        expected_msg = ('The associated table:"3dsumap" does not have'
                        ' two dimensions. It shape is:"(10, 3)"')

        self.assertEqual(str(cm.exception), expected_msg)

    def test_conflicting_kwargs(self):
        # This test ensures conflicting keys are removed from kwargs
        fig, ax = dimensionality_reduction_plot(
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


if __name__ == '__main__':
    unittest.main()
