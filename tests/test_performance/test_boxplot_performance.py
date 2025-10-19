import unittest
import time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from spac.visualization import boxplot, boxplot_interactive

matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestBoxplotPerformance(unittest.TestCase):
    """Performance comparison tests for boxplot vs boxplot_interactive."""

    @classmethod
    def setUpClass(cls):
        """Generate large datasets once for all tests."""
        print("\n" + "=" * 70)
        print("Setting up large datasets for boxplot performance tests...")
        print("=" * 70)
        
        # Generate 1M cell dataset
        print("\nGenerating 1M cell dataset...")
        start = time.time()
        cls.adata_1m = cls._generate_dataset(n_obs=1_000_000, random_state=42)
        print(f"  Completed in {time.time() - start:.2f} seconds")
        
        # Generate 5M cell dataset
        print("\nGenerating 5M cell dataset...")
        start = time.time()
        cls.adata_5m = cls._generate_dataset(n_obs=5_000_000, random_state=42)
        print(f"  Completed in {time.time() - start:.2f} seconds")
        
        # Generate 10M cell dataset
        print("\nGenerating 10M cell dataset...")
        start = time.time()
        cls.adata_10m = cls._generate_dataset(n_obs=10_000_000, random_state=42)
        print(f"  Completed in {time.time() - start:.2f} seconds")
        print("=" * 70 + "\n")

    @staticmethod
    def _generate_dataset(n_obs: int, random_state: int = 42) -> ad.AnnData:
        """
        Generate a synthetic AnnData object with realistic clustering.
        
        Creates dataset with:
        - 5 features (marker_1 to marker_5)
        - 5 annotations (cell_type, phenotype, region, batch, treatment)
        - 3 layers (normalized, log_transformed, scaled)
        """
        np.random.seed(random_state)
        
        # Generate base data with natural clustering
        n_features = 5
        n_centers = 5
        
        X, cluster_labels = make_blobs(
            n_samples=n_obs,
            n_features=n_features,
            centers=n_centers,
            cluster_std=1.5,
            random_state=random_state
        )
        
        # Make values positive and add variation
        X = np.abs(X) + np.random.exponential(scale=2.0, size=X.shape)
        
        # Create feature names
        feature_names = [f"marker_{i+1}" for i in range(n_features)]
        
        # Create annotations based on clusters
        cell_types = [f"Type_{chr(65+i)}" for i in range(5)]
        cell_type = np.array([cell_types[i % 5] for i in cluster_labels])
        
        phenotypes = [f"Pheno_{i+1}" for i in range(4)]
        phenotype = np.array([phenotypes[i % 4] for i in cluster_labels])
        random_mask = np.random.random(n_obs) < 0.2
        phenotype[random_mask] = np.random.choice(phenotypes, size=random_mask.sum())
        
        regions = ["Region_X", "Region_Y", "Region_Z"]
        region = np.random.choice(regions, size=n_obs)
        
        batches = ["Batch_1", "Batch_2", "Batch_3"]
        batch = np.random.choice(batches, size=n_obs)
        
        treatments = ["Control", "Treated"]
        treatment = np.random.choice(treatments, size=n_obs, p=[0.5, 0.5])
        
        # Create observations DataFrame
        obs = pd.DataFrame({
            'cell_type': pd.Categorical(cell_type),
            'phenotype': pd.Categorical(phenotype),
            'region': pd.Categorical(region),
            'batch': pd.Categorical(batch),
            'treatment': pd.Categorical(treatment)
        })
        
        # Create AnnData object
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = feature_names
        
        # Create layers with different transformations
        X_normalized = np.zeros_like(X)
        for i in range(n_features):
            feature_min = X[:, i].min()
            feature_max = X[:, i].max()
            X_normalized[:, i] = (X[:, i] - feature_min) / (feature_max - feature_min)
        adata.layers['normalized'] = X_normalized
        
        adata.layers['log_transformed'] = np.log1p(X)
        
        scaler = StandardScaler()
        adata.layers['scaled'] = scaler.fit_transform(X)
        
        return adata

    def tearDown(self):
        """Clean up matplotlib figures after each test."""
        plt.close('all')

    def _run_comparison(self, adata, test_name):
        """Run comparison between boxplot and boxplot_interactive."""
        n_obs = adata.n_obs
        features = ['marker_1', 'marker_2', 'marker_3', 'marker_4', 'marker_5']
        annotation = 'cell_type'
        layer = 'normalized'
        
        print(f"\n{'=' * 70}")
        print(f"{test_name}: {n_obs:,} cells")
        print(f"  Features: {', '.join(features)}")
        print(f"  Annotation: {annotation}")
        print(f"  Layer: {layer}")
        print(f"{'=' * 70}")
        
        # Test boxplot
        print("\n  Running boxplot...")
        start = time.time()
        fig, ax, df = boxplot(
            adata,
            features=features,
            annotation=annotation,
            layer=layer
        )
        boxplot_time = time.time() - start
        print(f"    Time: {boxplot_time:.2f} seconds")
        plt.close('all')
        
        # Test boxplot_interactive with downsampling
        print("\n  Running boxplot_interactive (with downsampling)...")
        start = time.time()
        result = boxplot_interactive(
            adata,
            features=features,
            annotation=annotation,
            layer=layer,
            showfliers='downsample'
        )
        interactive_time = time.time() - start
        print(f"    Time: {interactive_time:.2f} seconds")
        
        # Calculate speedup
        speedup = boxplot_time / interactive_time if interactive_time > 0 else 0
        
        print(f"\n  Results:")
        print(f"    boxplot:             {boxplot_time:.2f}s")
        print(f"    boxplot_interactive: {interactive_time:.2f}s")
        print(f"    Speedup factor:      {speedup:.2f}x")
        
        if speedup > 1:
            print(f"    → boxplot is {speedup:.2f}x faster")
        elif speedup < 1:
            print(f"    → boxplot_interactive is {1/speedup:.2f}x faster")
        else:
            print(f"    → Both functions have similar performance")
        
        print(f"{'=' * 70}\n")
        
        # Store results for potential further analysis
        return {
            'n_obs': n_obs,
            'boxplot_time': boxplot_time,
            'boxplot_interactive_time': interactive_time,
            'speedup_factor': speedup
        }

    def test_comparison_1m(self):
        """Compare boxplot vs boxplot_interactive with 1M cells."""
        self._run_comparison(self.adata_1m, "Boxplot Performance Comparison [1M cells]")

    def test_comparison_5m(self):
        """Compare boxplot vs boxplot_interactive with 5M cells."""
        self._run_comparison(self.adata_5m, "Boxplot Performance Comparison [5M cells]")

    def test_comparison_10m(self):
        """Compare boxplot vs boxplot_interactive with 10M cells."""
        self._run_comparison(self.adata_10m, "Boxplot Performance Comparison [10M cells]")


if __name__ == '__main__':
    unittest.main(verbosity=2)
