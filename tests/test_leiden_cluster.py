import unittest
import scanpy as sc
import anndata
import numpy as np
from spac.leiden_clustering import leiden_only_clustering, preprocess, plot  # make sure to import these

class TestClustering(unittest.TestCase):

    def setUp(self):
        # Create a mock `adata` object with the necessary attributes
        np.random.seed(0)
        X = np.random.rand(10, 5)  # 10 cells, 5 genes
        obs = {"cell_type": ["A", "B"] * 5}
        var = {"gene_ids": [f"gene{i}" for i in range(5)]}
        self.adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Check that neighbors and UMAP exist
    def test_preprocess_returns_anndata(self):
        ad_processed = preprocess(self.adata)
        self.assertIsInstance(ad_processed, anndata.AnnData, "Output should be an AnnData object")
        self.assertIn("X_pca", ad_processed.obsm, "PCA not computed")
        self.assertIn("neighbors", ad_processed.uns, "Neighbors not computed")
        self.assertIn("X_umap", ad_processed.obsm, "UMAP not computed")

    def test_leiden_only_clustering_creates_column(self):
        ad_clustered = leiden_only_clustering(self.adata)
        self.assertIn("leiden_clusters", ad_clustered.obs, "Leiden clustering column not added")
        self.assertEqual(ad_clustered.n_obs, self.adata.n_obs, "Number of cells should remain the same")

    def test_plot_runs_without_error(self):
        ad_clustered = leiden_only_clustering(self.adata)
        try:
            plot(ad_clustered)
        except Exception as e:
            self.fail(f"plot() raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
