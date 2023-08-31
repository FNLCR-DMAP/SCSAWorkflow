import unittest
import anndata
import numpy as np
from spac.transformations import UMAP


class TestUMAP(unittest.TestCase):

    def setUp(self):
        self.adata = anndata.AnnData(
            X=np.random.rand(10, 10).astype(np.float32)
        )
        self.layer = 'layer1'
        self.adata.layers[self.layer] = (
            np.random.rand(10, 10).astype(np.float32)
        )

    def test_typical_case(self):
        try:
            UMAP(self.adata, 5, 2, 0.1, 1.0, 2, 42, self.layer)
        except Exception as e:
            self.fail(f"UMAP raised exception: {e}")

        # Check that the UMAP coordinates were added to adata.obsm
        umap_key = self.layer + "_umap"
        self.assertIn(umap_key, self.adata.obsm)

    def test_no_layer(self):
        try:
            UMAP(self.adata, 5, 2, 0.1, 1.0, 2, 42)
        except Exception as e:
            self.fail(f"UMAP raised exception: {e}")

        # Check that the UMAP coordinates were added to adata.obsm
        self.assertIn("X_umap", self.adata.obsm)

    def test_output_shape(self):
        UMAP(self.adata, 5, 2, 0.1, 1.0, 2, 42, self.layer)

        # Check that the shape of the UMAP coordinates
        # matches the number of observations
        umap_shape = self.adata.obsm[self.layer + "_umap"].shape[0]
        self.assertEqual(umap_shape, self.adata.n_obs)


if __name__ == '__main__':
    unittest.main()
