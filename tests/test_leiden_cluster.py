# test_leiden.py
import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
from spac.leiden_clustering import preprocess, leiden_only_clustering, plot  

class TestLeidenClustering(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Small random dataset for basic tests
        data = np.random.rand(100, 5)
        self.adata = AnnData(X=data, var=pd.DataFrame(index=[f'gene{i}' for i in range(5)]))

        # Synthetic dataset for accuracy tests
        self.syn_data = self.create_synthetic_data()

    # ================== Synthetic Data Generator ==================
    def create_synthetic_data(self):
        """
        Creates a synthetic AnnData object with 2 well-separated clusters
        for testing clustering accuracy.
        """
        np.random.seed(42)
        cluster1 = np.random.normal(loc=0, scale=0.5, size=(50, 2))
        cluster2 = np.random.normal(loc=5, scale=0.5, size=(50, 2))
        data = np.vstack([cluster1, cluster2])
        ad = AnnData(X=data, var=pd.DataFrame(index=['gene1', 'gene2']))
        return ad

    # ================== Preprocess tests ==================
    def test_pca(self):
        ad = preprocess(self.adata)
        self.assertIn('X_pca', ad.obsm)

    def test_nearest_neighbors(self):
        ad = preprocess(self.adata)
        self.assertIn('neighbors', ad.uns)
        self.assertIn('distances', ad.obsp)
        self.assertIn('connectivities', ad.obsp)

    def test_umap_config(self):
        ad = preprocess(self.adata)
        self.assertIn('X_umap', ad.obsm)

    def test_return_preprocess_anndata(self):
        ad = preprocess(self.adata)
        self.assertIsInstance(ad, AnnData)

    # ================== leiden_only_clustering tests ==================
    def test_model_without_parameters(self):
        ad = leiden_only_clustering(self.adata)
        self.assertIn('leiden_clusters', ad.obs)
        self.assertEqual(ad.n_obs, self.adata.n_obs)

    def test_resolution(self):
        ad_low = leiden_only_clustering(self.adata, resolution=0.1)
        ad_high = leiden_only_clustering(self.adata, resolution=5.0)
        self.assertTrue(ad_low.obs['leiden_clusters'].nunique() < ad_high.obs['leiden_clusters'].nunique())

    def test_random_state(self):
        ad1 = leiden_only_clustering(self.adata, random_state=42)
        ad2 = leiden_only_clustering(self.adata, random_state=42)
        self.assertTrue((ad1.obs['leiden_clusters'] == ad2.obs['leiden_clusters']).all())

    def test_n_iterations(self):
        ad = leiden_only_clustering(self.adata, n_iterations=5)
        self.assertEqual(ad.n_obs, self.adata.n_obs)

    def test_key_added(self):
        custom_key = 'my_clusters'
        ad = leiden_only_clustering(self.adata, key_added=custom_key)
        self.assertIn(custom_key, ad.obs)

    # ================== Synthetic clustering accuracy tests ==================
    def test_clustering_accuracy(self):
        """
        Tests that leiden_only_clustering correctly separates the 2 known clusters.
        """
        ad = leiden_only_clustering(self.syn_data, resolution=0.5, random_state=42)
        self.assertIn('leiden_clusters', ad.obs)
        self.assertEqual(ad.obs['leiden_clusters'].nunique(), 2)

    def test_resolution_effect_on_synthetic_data(self):
        """
        Tests that increasing resolution increases the number of clusters
        """
        ad_low = leiden_only_clustering(self.syn_data, resolution=0.2, random_state=42)
        ad_high = leiden_only_clustering(self.syn_data, resolution=2.0, random_state=42)
        self.assertTrue(ad_low.obs['leiden_clusters'].nunique() < ad_high.obs['leiden_clusters'].nunique())

    def test_random_state_determinism_on_synthetic_data(self):
        ad1 = leiden_only_clustering(self.syn_data, random_state=123)
        ad2 = leiden_only_clustering(self.syn_data, random_state=123)
        self.assertTrue((ad1.obs['leiden_clusters'] == ad2.obs['leiden_clusters']).all())

    # ================== Plotting tests ==================
    def test_plot_without_parameters(self):
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad)
        except Exception as e:
            self.fail(f"Plotting with default parameters raised an exception: {e}")

    def test_title(self):
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, title="My UMAP Plot")
        except Exception as e:
            self.fail(f"Plotting with title raised an exception: {e}")

    def test_palettes(self):
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, palette='tab10')
        except Exception as e:
            self.fail(f"Plotting with palette raised an exception: {e}")

    def test_save_plot(self):
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, save='test_plot.pdf')
        except Exception as e:
            self.fail(f"Plotting with save parameter raised an exception: {e}")

    def test_point_size(self):
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, size=50)
        except Exception as e:
            self.fail(f"Plotting with point size raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()