import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
import unittest
from unittest.mock import patch
import anndata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from spac.visualization import dimensionality_reduction_plot
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestDimensionalityReductionPlot(unittest.TestCase):

    def setUp(self):
        self.adata = anndata.AnnData(X=np.random.rand(10, 10))
        self.adata.obsm['X_tsne'] = np.random.rand(10, 2)
        self.adata.obsm['X_umap'] = np.random.rand(10, 2)
        self.adata.obs['annotation_column'] = np.random.choice(
            ['A', 'B', 'C'], size=10
        )
        self.adata.var_names = ['gene_' + str(i) for i in range(10)]

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(1, 'tsne')
        self.assertEqual(str(cm.exception), "adata must be an AnnData object.")

    def test_invalid_method(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(self.adata, 'invalid_method')
        expected_msg = "Method should be one of {'tsne', 'umap'}."
        self.assertEqual(str(cm.exception), expected_msg)

    def test_annotation_and_feature(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(
                self.adata, 'tsne',
                annotation='annotation_column',
                feature='feature_column'
            )

        expected_msg = ("Please specify either an annotation or a feature "
                        "for coloring, not both.")
        self.assertEqual(str(cm.exception), expected_msg)

    def test_annotation_column(self):
        fig, ax = dimensionality_reduction_plot(
            self.adata, 'tsne', annotation='annotation_column'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_feature_column(self):
        fig, ax = dimensionality_reduction_plot(
            self.adata, 'tsne', feature='gene_1'
        )
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_ax_provided(self):
        fig, ax_provided = plt.subplots()
        fig_returned, ax_returned = dimensionality_reduction_plot(
            self.adata, 'tsne', ax=ax_provided
        )
        self.assertIs(fig, fig_returned)
        self.assertIs(ax_provided, ax_returned)

    def test_annotation_column_invalid(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(
                self.adata, 'tsne', annotation='invalid_column'
            )
        err_msg = "Annotation 'invalid_column' not found in adata.obs."
        self.assertTrue(err_msg in str(cm.exception))

    def test_feature_column_invalid(self):
        with self.assertRaises(ValueError) as cm:
            dimensionality_reduction_plot(
                self.adata, 'tsne', feature='invalid_gene'
            )
        err_msg = "Feature 'invalid_gene' not found in adata.var_names."
        self.assertTrue(err_msg in str(cm.exception))

    @patch('scanpy.pl.tsne')
    def test_tsne_kwargs_color(self, mock_tsne):
        dimensionality_reduction_plot(
            self.adata, 'tsne', annotation='annotation_column'
        )
        mock_tsne.assert_called_once()
        args, kwargs = mock_tsne.call_args
        self.assertEqual(kwargs['color'], 'annotation_column')

    @patch('scanpy.pl.umap')
    def test_umap_kwargs_color(self, mock_umap):
        dimensionality_reduction_plot(self.adata, 'umap', feature='gene_1')
        mock_umap.assert_called_once()
        args, kwargs = mock_umap.call_args
        self.assertEqual(kwargs['color'], 'gene_1')


if __name__ == '__main__':
    unittest.main()
