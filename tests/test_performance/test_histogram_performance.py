import unittest
import time
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from spac.visualization import histogram
from spac.utils import check_annotation, check_feature, check_table

matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window




skip_perf = unittest.skipUnless(
    os.getenv("SPAC_RUN_PERF") == "1",
    "Perf tests disabled by default"
)

@skip_perf
class TestHistogramPerformance(unittest.TestCase):
    """Performance comparison tests for histogram vs histogram_old."""

    @classmethod
    def setUpClass(cls):
        """Generate large datasets once for all tests."""
        print("\n" + "=" * 70)
        print("Setting up large datasets for histogram performance tests...")
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

    @staticmethod
    def histogram_old(adata, feature=None, annotation=None, layer=None,
                      group_by=None, together=False, ax=None,
                      x_log_scale=False, y_log_scale=False, **kwargs):
        """
        Old histogram implementation for performance comparison.

        Copied from commit 1cfad52f00aa6c1b8384f727b60e3bf07f57bee6 in
        visualization.py, before the refactor to histogram
        """
        # If no feature or annotation is specified, apply default behavior
        if feature is None and annotation is None:
            feature = adata.var_names[0]
            warnings.warn(
                "No feature or annotation specified. "
                "Defaulting to the first feature: "
                f"'{feature}'.",
                UserWarning
            )

        # Use utility functions for input validation
        if layer:
            check_table(adata, tables=layer)
        if annotation:
            check_annotation(adata, annotations=annotation)
        if feature:
            check_feature(adata, features=feature)
        if group_by:
            check_annotation(adata, annotations=group_by)

        # If layer is specified, get the data from that layer
        if layer:
            df = pd.DataFrame(
                adata.layers[layer], index=adata.obs.index, columns=adata.var_names
            )
        else:
            df = pd.DataFrame(
                 adata.X, index=adata.obs.index, columns=adata.var_names
            )
            layer = 'Original'

        df = pd.concat([df, adata.obs], axis=1)

        if feature and annotation:
            raise ValueError("Cannot pass both feature and annotation,"
                             " choose one.")

        data_column = feature if feature else annotation

        # Check for negative values and apply log1p transformation if x_log_scale is True
        if x_log_scale:
            if (df[data_column] < 0).any():
                print(
                    "There are negative values in the data, disabling x_log_scale."
                )
                x_log_scale = False
            else:
                df[data_column] = np.log1p(df[data_column])

        if ax is not None:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots()

        axs = []

        # Prepare the data for plotting
        plot_data = df.dropna(subset=[data_column])

        # Bin calculation section
        def cal_bin_num(num_rows):
            bins = max(int(2*(num_rows ** (1/3))), 1)
            print(f'Automatically calculated number of bins is: {bins}')
            return(bins)

        num_rows = plot_data.shape[0]

        # Check if bins is being passed
        if 'bins' not in kwargs:
            kwargs['bins'] = cal_bin_num(num_rows)

        # Plotting with or without grouping
        if group_by:
            groups = df[group_by].dropna().unique().tolist()
            n_groups = len(groups)
            if n_groups == 0:
                raise ValueError("There must be at least one group to create a"
                                 " histogram.")

            if together:
                kwargs.setdefault("multiple", "stack")
                kwargs.setdefault("element", "bars")

                sns.histplot(data=df.dropna(), x=data_column, hue=group_by,
                             ax=ax, **kwargs)
                if feature:
                    ax.set_title(f'Layer: {layer}')
                axs.append(ax)
            else:
                fig, ax_array = plt.subplots(
                    n_groups, 1, figsize=(5, 5 * n_groups)
                )

                if n_groups == 1:
                    ax_array = [ax_array]
                else:
                    ax_array = ax_array.flatten()

                for i, ax_i in enumerate(ax_array):
                    group_data = plot_data[plot_data[group_by] == groups[i]]

                    sns.histplot(data=group_data, x=data_column, ax=ax_i, **kwargs)
                    if feature:
                        ax_i.set_title(f'{groups[i]} with Layer: {layer}')
                    else:
                        ax_i.set_title(f'{groups[i]}')

                    if y_log_scale:
                        ax_i.set_yscale('log')

                    if x_log_scale:
                        xlabel = f'log({data_column})'
                    else:
                        xlabel = data_column
                    ax_i.set_xlabel(xlabel)

                    stat = kwargs.get('stat', 'count')
                    ylabel_map = {
                        'count': 'Count',
                        'frequency': 'Frequency',
                        'density': 'Density',
                        'probability': 'Probability'
                    }
                    ylabel = ylabel_map.get(stat, 'Count')
                    if y_log_scale:
                        ylabel = f'log({ylabel})'
                    ax_i.set_ylabel(ylabel)

                    axs.append(ax_i)
        else:
            sns.histplot(data=plot_data, x=data_column, ax=ax, **kwargs)
            if feature:
                ax.set_title(f'Layer: {layer}')
            axs.append(ax)

        if y_log_scale:
            ax.set_yscale('log')

        if x_log_scale:
            xlabel = f'log({data_column})'
        else:
            xlabel = data_column
        ax.set_xlabel(xlabel)

        stat = kwargs.get('stat', 'count')
        ylabel_map = {
            'count': 'Count',
            'frequency': 'Frequency',
            'density': 'Density',
            'probability': 'Probability'
        }
        ylabel = ylabel_map.get(stat, 'Count')
        if y_log_scale:
            ylabel = f'log({ylabel})'
        ax.set_ylabel(ylabel)

        if len(axs) == 1:
            return fig, axs[0]
        else:
            return fig, axs

    def _run_comparison(self, adata, test_name):
        """Run comparison between histogram_old and histogram."""
        n_obs = adata.n_obs
        feature = 'marker_1'
        annotation = None
        layer = 'normalized'

        print(f"\n{'=' * 70}")
        print(f"{test_name}: {n_obs:,} cells")
        print(f"  Feature: {feature}")
        print(f"  Annotation: {annotation}")
        print(f"  Layer: {layer}")
        print(f"{'=' * 70}")

        # Test histogram_old
        print("\n  Running histogram_old...")
        start = time.time()
        fig_old, ax_old = self.histogram_old(
            adata,
            feature=feature,
            annotation=annotation,
            layer=layer
        )
        old_time = time.time() - start
        print(f"    Time: {old_time:.2f} seconds")
        plt.close('all')

        # Test histogram from SPAC
        print("\n  Running histogram (SPAC)...")
        start = time.time()
        result = histogram(
            adata,
            feature=feature,
            annotation=annotation,
            layer=layer
        )
        new_time = time.time() - start
        print(f"    Time: {new_time:.2f} seconds")
        plt.close('all')

        # Calculate speedup
        speedup = old_time / new_time if new_time > 0 else 0

        print(f"\n  Results:")
        print(f"    histogram_old:  {old_time:.2f}s")
        print(f"    histogram:      {new_time:.2f}s")
        print(f"    Speedup factor: {speedup:.2f}x")

        if speedup > 1:
            print(f"    → histogram (SPAC) is {speedup:.2f}x faster")
        elif speedup < 1:
            print(f"    → histogram_old is {1/speedup:.2f}x faster")
        else:
            print(f"    → Both functions have similar performance")

        print(f"{'=' * 70}\n")

        # Store results for potential further analysis
        return {
            'n_obs': n_obs,
            'histogram_old_time': old_time,
            'histogram_time': new_time,
            'speedup_factor': speedup
        }

    def test_comparison_1m(self):
        """Compare histogram_old vs histogram with 1M cells."""
        self._run_comparison(self.adata_1m, "Histogram Performance Comparison [1M cells]")

    def test_comparison_5m(self):
        """Compare histogram_old vs histogram with 5M cells."""
        self._run_comparison(self.adata_5m, "Histogram Performance Comparison [5M cells]")

    def test_comparison_10m(self):
        """Compare histogram_old vs histogram with 10M cells."""
        self._run_comparison(self.adata_10m, "Histogram Performance Comparison [10M cells]")


if __name__ == '__main__':
    unittest.main(verbosity=2)
