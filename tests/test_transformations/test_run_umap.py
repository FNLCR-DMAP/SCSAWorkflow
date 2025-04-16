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

        self.adata.obsm["derived_features"] = \
            np.random.rand(10, 3, 2).astype(float)

    def test_typical_case(self):
        run_umap(self.adata, n_neighbors=5, min_dist=0.1, n_components=2,
                 random_state=42, layer=self.layer)
        self.assertIn("X_umap", self.adata.obsm)

    def test_output_shape(self):
        run_umap(self.adata, n_neighbors=5, min_dist=0.1, n_components=2,
                 random_state=42, layer=self.layer)

        # Check that the shape of the UMAP coordinates
        # matches the number of observations and dimensions
        umap_shape = self.adata.obsm["X_umap"].shape
        self.assertEqual(umap_shape, (self.adata.n_obs, 2))

    def test_output_name(self):

        output_name = "my_umap"
        run_umap(
            self.adata,
            n_neighbors=5,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            layer=self.layer,
            output_derived_feature=output_name)

        # Check that the shape of the UMAP coordinates
        # matches the number of observations and dimensions
        umap_shape = self.adata.obsm[output_name].shape
        self.assertEqual(umap_shape, (self.adata.n_obs, 2))

    def test_associated_table(self):

        run_umap(
            self.adata,
            n_neighbors=5,
            min_dist=0.1,
            n_components=2,
            random_state=42,
            associated_table="derived_features"
        )

        umap_shape = self.adata.obsm['X_umap'].shape
        self.assertEqual(umap_shape, (self.adata.n_obs, 2))

    def test_type_consistency(self):
        run_umap(self.adata, n_neighbors=5, min_dist=0.1, n_components=2,
                 random_state=42, layer=self.layer)

        # Check data type of UMAP coordinates
        umap_dtype = self.adata.obsm["X_umap"].dtype
        is_floating = np.issubdtype(umap_dtype, np.floating)

        self.assertTrue(is_floating)

    def test_different_parameters(self):
        run_umap(self.adata, n_neighbors=3, min_dist=0.2, n_components=3,
                 random_state=0, layer=self.layer)

        # Check that the shape of the UMAP coordinates matches the expected
        umap_shape = self.adata.obsm["X_umap"].shape
        self.assertEqual(umap_shape, (self.adata.n_obs, 3))


if __name__ == '__main__':
    unittest.main()
