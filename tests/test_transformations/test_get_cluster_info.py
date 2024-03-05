import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import get_cluster_info


class TestGetClusterInfo(unittest.TestCase):

    def setUp(self):
        # Create a mock AnnData object
        X_data = np.random.rand(100, 5)  # 100 cells, 5 features
        cluster_choices = ["Cluster1", "Cluster2", "Cluster3"]
        obs_data = {
            "phenograph": np.random.choice(cluster_choices, 100),
            "another_annotation": np.random.choice(["A", "B"], 100),
        }
        var_data = pd.DataFrame(index=[f"Gene{i}" for i in range(1, 6)])
        obs_df = pd.DataFrame(obs_data)
        self.adata = AnnData(X=X_data, obs=obs_df, var=var_data)
        # Add a layer with some modified data
        self.adata.layers["scaled"] = X_data * 2

    def test_default_features_with_percentage(self):
        result = get_cluster_info(self.adata)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Cluster", result.columns)
        self.assertIn("Number of Cells", result.columns)
        self.assertIn("Percentage", result.columns)
        self.assertIn("mean_Gene1", result.columns)
        # Check that the sum of all percentages is approximately 100
        self.assertAlmostEqual(
            result["Percentage"].sum(), 100, places=1
        )

    def test_specific_features_with_percentage(self):
        result = get_cluster_info(self.adata, features=["Gene1", "Gene2"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Cluster", result.columns)
        self.assertIn("Number of Cells", result.columns)
        self.assertIn("Percentage", result.columns)
        self.assertIn("mean_Gene1", result.columns)
        self.assertIn("mean_Gene2", result.columns)
        self.assertNotIn("mean_Gene3", result.columns)

    def test_layer_specific_features_with_percentage(self):
        # Test functionality with a specific layer
        result = get_cluster_info(self.adata, layer="scaled")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Cluster", result.columns)
        self.assertIn("Number of Cells", result.columns)
        self.assertIn("Percentage", result.columns)
        # Ensure features are processed from the specified layer
        self.assertIn("mean_Gene1", result.columns)
        # Extract the mean of the first feature in the 'scaled' layer
        scaled_mean = self.adata.layers["scaled"][:, 0]
        expected_mean = np.mean(scaled_mean)
        # Calculate the mean of the reported means for Gene1 in Cluster1
        # Select rows where the cluster is 'Cluster1'
        cluster1_filter = result["Cluster"] == "Cluster1"
        # Calculate the mean of 'mean_Gene1' for those rows
        cluster1_means = result.loc[cluster1_filter, "mean_Gene1"]
        actual_mean = cluster1_means.mean()
        self.assertTrue(np.isclose(expected_mean, actual_mean, atol=0.1))

    def test_default_features_with_known_values_and_percentage(self):
        # Create a mock AnnData object with specific known values
        X_data = np.array([
            [1, 2],
            [2, 1],
            [1, 2],
            [3, 4],
            [4, 3]
        ])  # 5 cells, 2 features
        cluster_choices = [
            "Cluster1", "Cluster1", "Cluster2", "Cluster2", "Cluster2"
        ]
        obs_data = {
            "phenograph": cluster_choices,
        }
        var_data = pd.DataFrame(index=[f"Gene{i}" for i in range(1, 3)])
        obs_df = pd.DataFrame(obs_data)
        adata = AnnData(X=X_data, obs=obs_df, var=var_data)

        # Call the function
        result = get_cluster_info(
            adata,
            annotation="phenograph",
            features=["Gene1", "Gene2"]
        )

        # Manually calculate expected output
        # For Cluster1, percentage should be (2/5)*100 = 40.0
        cluster_filter = result['Cluster'] == 'Cluster1'
        percentage_value = result.loc[cluster_filter, 'Percentage']\
                                 .iloc[0]
        self.assertAlmostEqual(percentage_value, 40.0, places=1)

        # For Cluster1, mean of Gene1 should be (1+2)/2 = 1.5
        mean_gene1_value = result.loc[cluster_filter, 'mean_Gene1'].iloc[0]
        self.assertAlmostEqual(mean_gene1_value, 1.5, places=4)


if __name__ == "__main__":
    unittest.main()
