import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

# testing the functionality of leiden_clustering.py
from spac.leiden_clustering import preprocess, leiden_only_clustering, plot

class TestLeidenClustering(unittest.TestCase):
    def setUp(self):
        """
        Initializes test fixtures before each test method.

        Creates two AnnData objects:
        - `self.adata`: A small 100-cell x 5-gene random dataset for general
          unit tests (preprocessing, parameter validation, etc.).
        - `self.syn_data`: A synthetic dataset with well-separated clusters
          for testing clustering accuracy and reproducibility.
        """
        np.random.seed(42)
        # Small random dataset for basic tests
        data = np.random.rand(100, 5)
        self.adata = AnnData(X=data, var=pd.DataFrame(index=[f'gene{i}' for i in range(5)]))

        # Synthetic dataset for accuracy tests
        self.syn_data = self.create_synthetic_data()

    # ================== Synthetic Data Generator ==================
    def create_synthetic_data(self):
        """
        Generates a synthetic AnnData object with 2 clearly separated clusters
        centered at 0 and 5 respectively, used to validate clustering accuracy.
        """
        np.random.seed(42)
        cluster1 = np.random.normal(loc=0, scale=0.5, size=(50, 2))
        cluster2 = np.random.normal(loc=5, scale=0.5, size=(50, 2))
        data = np.vstack([cluster1, cluster2])
        ad = AnnData(X=data, var=pd.DataFrame(index=['gene1', 'gene2']))
        return ad

    # ================== Preprocess Tests ==================
    def test_pca(self):
        """
        Verifies that PCA embedding is computed and stored in obsm['X_pca']
        after preprocessing.
        """
        ad = preprocess(self.adata)
        self.assertIn('X_pca', ad.obsm)

    def test_nearest_neighbors(self):
        """
        Verifies that the nearest neighbor graph is computed, checking that
        neighbor metadata, distance matrix, and connectivities matrix are all
        present after preprocessing.
        """
        ad = preprocess(self.adata)
        self.assertIn('neighbors', ad.uns)
        self.assertIn('distances', ad.obsp)
        self.assertIn('connectivities', ad.obsp)

    def test_umap_config(self):
        """
        Verifies that the UMAP embedding is computed and stored in
        obsm['X_umap'] after preprocessing.
        """
        ad = preprocess(self.adata)
        self.assertIn('X_umap', ad.obsm)

    def test_return_preprocess_anndata(self):
        """
        Verifies that preprocess() returns an AnnData object rather than
        modifying in place or returning an unexpected type.
        """
        ad = preprocess(self.adata)
        self.assertIsInstance(ad, AnnData)

    # ================== Leiden Clustering Tests ==================
    def test_model_without_parameters(self):
        """
        Verifies that leiden_only_clustering() runs with default parameters,
        produces the expected 'leiden_clusters' column, and preserves the
        original cell count.
        """
        ad = leiden_only_clustering(self.adata)
        self.assertIn('leiden_clusters', ad.obs)
        self.assertEqual(ad.n_obs, self.adata.n_obs)

    def test_resolution(self):
        """
        Verifies that resolution controls cluster granularity — a low resolution
        should produce fewer clusters than a high resolution on the same dataset.
        """
        ad_low = leiden_only_clustering(self.adata, resolution=0.1)
        ad_high = leiden_only_clustering(self.adata, resolution=5.0)
        self.assertTrue(ad_low.obs['leiden_clusters'].nunique() < ad_high.obs['leiden_clusters'].nunique())

    def test_random_state(self):
        """
        Verifies that clustering is deterministic when the same random_state is
        used — two runs with identical seeds should produce identical cluster labels.
        """
        ad1 = leiden_only_clustering(self.adata, random_state=42)
        ad2 = leiden_only_clustering(self.adata, random_state=42)
        self.assertTrue((ad1.obs['leiden_clusters'] == ad2.obs['leiden_clusters']).all())

    def test_n_iterations(self):
        """
        Verifies that setting a fixed number of iterations completes without error
        and does not drop or add any cells from the dataset.
        """
        ad = leiden_only_clustering(self.adata, n_iterations=5)
        self.assertEqual(ad.n_obs, self.adata.n_obs)

    def test_key_added(self):
        """
        Verifies that cluster labels are stored under a custom column name when
        key_added is specified, rather than the default 'leiden_clusters'.
        """
        custom_key = 'my_clusters'
        ad = leiden_only_clustering(self.adata, key_added=custom_key)
        self.assertIn(custom_key, ad.obs)

    # ================== Synthetic Clustering Accuracy Tests ==================
    def test_clustering_accuracy(self):
        """
        Verifies that Leiden correctly identifies exactly 2 clusters on a
        synthetic dataset with 2 well-separated Gaussian distributions.
        """
        ad = leiden_only_clustering(self.syn_data, resolution=0.1, random_state=42)
        self.assertIn('leiden_clusters', ad.obs)
        self.assertEqual(ad.obs['leiden_clusters'].nunique(), 2)

    def test_resolution_effect_on_synthetic_data(self):
        """
        Verifies that increasing resolution produces more clusters on synthetic
        data, confirming resolution behaves as expected on a known structure.
        """
        ad_low = leiden_only_clustering(self.syn_data, resolution=0.2, random_state=42)
        ad_high = leiden_only_clustering(self.syn_data, resolution=2.0, random_state=42)
        self.assertTrue(ad_low.obs['leiden_clusters'].nunique() < ad_high.obs['leiden_clusters'].nunique())

    def test_random_state_determinism_on_synthetic_data(self):
        """
        Verifies that clustering is reproducible on synthetic data — two runs
        with the same random_state should yield identical cluster assignments.
        Complements test_random_state by confirming determinism on structured data.
        """
        ad1 = leiden_only_clustering(self.syn_data, random_state=123)
        ad2 = leiden_only_clustering(self.syn_data, random_state=123)
        self.assertTrue((ad1.obs['leiden_clusters'] == ad2.obs['leiden_clusters']).all())

    # ================== Auto-Preprocess & Non-Mutation Tests ==================
    def test_auto_preprocess_when_no_neighbors(self):
        """
        Verifies that leiden_only_clustering automatically runs preprocessing
        when the input AnnData has no precomputed neighbor graph, allowing
        raw input to be clustered without a manual preprocess() call first.
        """
        raw = self.adata.copy()
        self.assertNotIn('neighbors', raw.uns)
        ad = leiden_only_clustering(raw)
        self.assertIn('leiden_clusters', ad.obs)
        self.assertIn('neighbors', ad.uns)
        self.assertIn('X_pca', ad.obsm)

    def test_skip_preprocess_when_neighbors_exist(self):
        """
        Verifies that leiden_only_clustering does NOT re-run preprocessing
        when the input AnnData already has a neighbor graph computed,
        confirming the conditional preprocess branch behaves correctly.
        """
        pre = preprocess(self.adata)
        pre.uns['neighbors']['_test_marker'] = 'preserved'
        ad = leiden_only_clustering(pre)
        self.assertEqual(ad.uns['neighbors'].get('_test_marker'), 'preserved')

    def test_preprocess_does_not_modify_input(self):
        """
        Verifies that preprocess() operates on a copy and leaves the original
        AnnData untouched (no PCA, neighbors, or UMAP added to the input).
        """
        original = self.adata.copy()
        _ = preprocess(self.adata)
        self.assertNotIn('X_pca', self.adata.obsm)
        self.assertNotIn('neighbors', self.adata.uns)
        self.assertNotIn('X_umap', self.adata.obsm)
        np.testing.assert_array_equal(self.adata.X, original.X)

    def test_clustering_does_not_modify_input(self):
        """
        Verifies that leiden_only_clustering() operates on a copy and does
        not write cluster labels back into the original AnnData's obs.
        """
        _ = leiden_only_clustering(self.adata)
        self.assertNotIn('leiden_clusters', self.adata.obs)

    # ================== Plotting Tests ==================
    def test_plot_without_parameters(self):
        """
        Verifies that plot() runs without error using only default parameters,
        confirming basic compatibility with a clustered AnnData object.
        """
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad)
        except Exception as e:
            self.fail(f"Plotting with default parameters raised an exception: {e}")

    def test_title(self):
        """
        Verifies that passing a custom title string does not cause an error,
        confirming the title parameter is correctly forwarded to sc.pl.umap.
        """
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, title="My UMAP Plot")
        except Exception as e:
            self.fail(f"Plotting with title raised an exception: {e}")

    def test_palettes(self):
        """
        Verifies that a named matplotlib palette can be passed without error,
        confirming the palette parameter is correctly forwarded to sc.pl.umap.
        """
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, palette='tab10')
        except Exception as e:
            self.fail(f"Plotting with palette raised an exception: {e}")

    def test_save_plot(self):
        """
        Verifies that passing a filename to save does not raise an error,
        confirming the save parameter is correctly forwarded to sc.pl.umap.
        """
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, save='test_plot.pdf')
        except Exception as e:
            self.fail(f"Plotting with save parameter raised an exception: {e}")

    def test_point_size(self):
        """
        Verifies that a custom point size can be passed without error,
        confirming the size parameter is correctly forwarded to sc.pl.umap.
        """
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, size=50)
        except Exception as e:
            self.fail(f"Plotting with point size raised an exception: {e}")

    def test_plot_with_custom_color(self):
        """
        Verifies that the plot() color parameter accepts a custom obs column
        (other than the default 'leiden_clusters'), confirming the parameter
        is correctly forwarded to sc.pl.umap.
        """
        ad = leiden_only_clustering(self.adata)
        ad.obs['custom_label'] = ad.obs['leiden_clusters'].astype(str)
        try:
            plot(ad, color='custom_label')
        except Exception as e:
            self.fail(f"Plotting with custom color raised an exception: {e}")

    def test_plot_with_legend_loc(self):
        """
        Verifies that a custom legend location string can be passed without
        error, confirming the legend_loc parameter is correctly forwarded
        to sc.pl.umap.
        """
        ad = leiden_only_clustering(self.adata)
        try:
            plot(ad, legend_loc='on data')
        except Exception as e:
            self.fail(f"Plotting with legend_loc raised an exception: {e}")

    def test_plot_with_ax(self):
        """
        Verifies that an existing matplotlib Axes object can be passed via
        the ax parameter, confirming the plot can be embedded into a
        user-managed figure rather than always creating a new one.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ad = leiden_only_clustering(self.adata)
        fig, ax = plt.subplots()
        try:
            plot(ad, ax=ax)
        except Exception as e:
            self.fail(f"Plotting with custom ax raised an exception: {e}")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()