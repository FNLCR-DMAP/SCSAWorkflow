import unittest
import anndata
import numpy as np
from spac.transformations import run_umap


class TestRunUMAP(unittest.TestCase):

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
            run_umap(self.adata, n_neighbors=5, min_dist=0.1, n_components=2,
                     random_state=42, layer=self.layer)
        except Exception as e:
            self.fail(f"run_umap raised exception: {e}")

        # Check that the UMAP coordinates were added to adata.obsm
        self.assertIn("X_umap", self.adata.obsm)

    def test_no_layer(self):
        try:
            run_umap(self.adata, n_neighbors=5, min_dist=0.1, n_components=2,
                     random_state=42)
        except Exception as e:
            self.fail(f"run_umap raised exception: {e}")

        # Check that the UMAP coordinates were added to adata.obsm
        self.assertIn("X_umap", self.adata.obsm)

    def test_output_shape(self):
        run_umap(self.adata, n_neighbors=5, min_dist=0.1, n_components=2,
                 random_state=42, layer=self.layer)

        # Check that the shape of the UMAP coordinates
        # matches the number of observations and dimensions
        umap_shape = self.adata.obsm["X_umap"].shape
        self.assertEqual(umap_shape, (self.adata.n_obs, 2))

    def test_type_consistency(self):
        run_umap(self.adata, n_neighbors=5, min_dist=0.1, n_components=2,
                 random_state=42, layer=self.layer)

        # Check data type of UMAP coordinates
        umap_dtype = self.adata.obsm["X_umap"].dtype
        is_floating = np.issubdtype(umap_dtype, np.floating)

        self.assertTrue(is_floating)

    def test_different_parameters(self):
        try:
            run_umap(self.adata, n_neighbors=3, min_dist=0.2, n_components=3,
                     random_state=0, layer=self.layer)
        except Exception as e:
            self.fail(
                f"run_umap raised exception with different parameters: {e}"
            )


if __name__ == '__main__':
    unittest.main()
