import unittest
import anndata
import numpy as np
from spac.visualization import tsne_plot


class TestTsnePlot(unittest.TestCase):

    def setUp(self):
        self.adata = anndata.AnnData(X=np.random.rand(10, 10))
        self.adata.obsm['X_tsne'] = np.random.rand(10, 2)

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


if __name__ == '__main__':
    unittest.main()
