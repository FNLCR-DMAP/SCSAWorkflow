import unittest
import anndata
import numpy as np
import matplotlib.pyplot as plt
from spac.visualization import tsne_plot
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestTsnePlot(unittest.TestCase):

    def setUp(self):
        self.adata = anndata.AnnData(X=np.random.rand(10, 10))
        self.adata.obsm['X_tsne'] = np.random.rand(10, 2)
        self.adata.obs['color_column'] = np.random.choice(
            ['A', 'B', 'C'], size=10)

    def test_invalid_input_type(self):
        with self.assertRaises(ValueError) as cm:
            tsne_plot(1)
        self.assertEqual(str(cm.exception),
                         "adata must be an AnnData object.")

    def test_no_tsne_data(self):
        del self.adata.obsm['X_tsne']
        with self.assertRaises(ValueError) as cm:
            tsne_plot(self.adata)
        self.assertEqual(str(cm.exception),
                         "adata.obsm does not contain 'X_tsne',"
                         " perform t-SNE transformation first.")

    def test_color_column(self):
        fig, ax = tsne_plot(self.adata, color_column='color_column')
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    def test_ax_provided(self):
        fig, ax_provided = plt.subplots()
        fig_returned, ax_returned = tsne_plot(self.adata, ax=ax_provided)
        self.assertIs(fig, fig_returned)
        self.assertIs(ax_provided, ax_returned)

    def test_color_column_invalid(self):
        with self.assertRaises(KeyError) as cm:
            tsne_plot(self.adata, color_column='invalid_column')
        self.assertEqual(
            str(cm.exception),
            "\"'invalid_column' not found in adata.obs or adata.var.\""
        )


if __name__ == '__main__':
    unittest.main()
