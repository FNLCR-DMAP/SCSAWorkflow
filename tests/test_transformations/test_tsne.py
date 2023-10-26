import unittest
import anndata
import numpy as np
from spac.transformations import tsne


class TestTsne(unittest.TestCase):

    def setUp(self):
        self.adata = anndata.AnnData(X=np.random.rand(10, 10))
        self.layer = 'layer1'
        self.adata.layers[self.layer] = np.random.rand(10, 10)

    def test_typical_case(self):
        try:
            tsne(self.adata, self.layer)
        except Exception as e:
            self.fail(f"tsne raised exception: {e}")

        # Check that X_tsne was added to adata.obsm
        tsne_key = self.layer + "_tsne"
        self.assertIn(tsne_key, self.adata.obsm)

    def test_no_layer(self):
        try:
            tsne(self.adata)
        except Exception as e:
            self.fail(f"tsne raised exception: {e}")

        # Check that X_tsne was added to adata.obsm
        self.assertIn("X_tsne", self.adata.obsm)

    def test_invalid_layer(self):
        with self.assertRaises(ValueError):
            tsne(self.adata, 'invalid_layer')

    # Add a test for the shape of the output
    def test_output_shape(self):
        tsne(self.adata, self.layer)

        # Check that the shape of X_tsne matches the number of annotations
        tsne_shape = self.adata.obsm[self.layer + "_tsne"].shape[0]
        self.assertEqual(tsne_shape, self.adata.n_obs)


if __name__ == '__main__':
    unittest.main()
