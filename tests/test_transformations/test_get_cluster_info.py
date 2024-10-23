import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import get_cluster_info


class TestGetClusterInfo(unittest.TestCase):

    def setUp(self):
        # Create a mock AnnData object with specific known values
        self.base_data = np.array([
            [1, 2],
            [2, 1],
            [1, 2],
            [3, 4],
            [4, 3]
        ])  # 5 cells, 2 features
        cluster_choices = [
            "Cluster1", "Cluster1", "Cluster2", "Cluster2", "Cluster2"
        ]
        obs_data = {"phenograph": cluster_choices}
        var_data = pd.DataFrame(index=[f"Gene{i}" for i in range(1, 3)])
        obs_df = pd.DataFrame(obs_data)
        self.adata = AnnData(X=self.base_data, obs=obs_df, var=var_data)

    def test_default_features_with_percentage(self):
        result = get_cluster_info(self.adata, annotation="phenograph")
        # Check that the sum of all percentages is approximately 100
        self.assertAlmostEqual(result["Percentage"].sum(), 100, places=1)

    def test_specific_features_with_default_layer(self):
        result = get_cluster_info(
            self.adata,
            annotation="phenograph",
            features=["Gene1", "Gene2"]
        )

        # Manually calculate expected output
        # For Cluster1, percentage should be (2/5)*100 = 40.0
        cluster_filter = result['phenograph'] == 'Cluster1'
        percentage_value = result.loc[cluster_filter, 'Percentage']\
                                 .iloc[0]
        self.assertAlmostEqual(percentage_value, 40.0, places=1)

        # For Cluster1, mean of Gene1 should be (1+2)/2 = 1.5
        mean_gene1_value = result.loc[cluster_filter, 'mean_Gene1'].iloc[0]
        self.assertAlmostEqual(mean_gene1_value, 1.5, places=4)

    def test_specific_feature_with_custom_layer(self):
        # Add a custom layer to the AnnData object for this specific test
        self.adata.layers["custom_layer"] = self.base_data * 1.5

        # Call the function with the "custom_layer" specified
        result = get_cluster_info(
            self.adata,
            annotation="phenograph",
            layer="custom_layer",
            features=["Gene1", "Gene2"]
        )

        # For Cluster1, calculate expected mean values for custom_layer
        cluster_filter = result['phenograph'] == 'Cluster1'
        expected_mean_gene1_cluster1 = ((1.5*1 + 1.5*2) / 2)
        mean_gene1_value = result.loc[cluster_filter, 'mean_Gene1'].iloc[0]
        self.assertAlmostEqual(
            mean_gene1_value,
            expected_mean_gene1_cluster1,
            places=4
        )


if __name__ == "__main__":
    unittest.main()
