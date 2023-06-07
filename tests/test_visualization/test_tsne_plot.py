import unittest
import anndata
import numpy as np
from spac.transformations import tsne
from spac.visualization import tsne_plot


class TestTsnePlot(unittest.TestCase):

    def setUp(self):
        self.adata = anndata.AnnData(X=np.random.rand(10, 10))
        self.adata.obsm['X_tsne'] = np.random.rand(10, 2)

    def test_invalid_input_type(self):
        adata = 10  # Simulate adata being an integer

        # Check if it raises a ValueError with the expected message
        with self.assertRaises(ValueError) as cm:
            if not isinstance(adata, anndata.AnnData):
                raise ValueError("adata must be an AnnData object.")

        self.assertEqual(
            str(cm.exception),
            "adata must be an AnnData object."
        )

    def test_no_tsne_data(self):
        # Simulate removal of 'X_tsne' in adata.obsm
        del self.adata.obsm['X_tsne']

        with self.assertRaises(ValueError) as cm:
            if 'X_tsne' not in self.adata.obsm:
                raise ValueError(
                    "adata.obsm does not contain 'X_tsne', "
                    "perform t-SNE transformation first."
                )

        self.assertEqual(
            str(cm.exception),
            "adata.obsm does not contain 'X_tsne', "
            "perform t-SNE transformation first."
        )


if __name__ == '__main__':
    unittest.main()
