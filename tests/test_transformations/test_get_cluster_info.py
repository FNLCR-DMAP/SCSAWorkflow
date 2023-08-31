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

    def test_default_features(self):
        result = get_cluster_info(self.adata)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Cluster", result.columns)
        self.assertIn("Number of Cells", result.columns)
        self.assertIn("mean_Gene1", result.columns)

    def test_specific_features(self):
        result = get_cluster_info(self.adata, features=["Gene1", "Gene2"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Cluster", result.columns)
        self.assertIn("Number of Cells", result.columns)
        self.assertIn("mean_Gene1", result.columns)
        self.assertIn("mean_Gene2", result.columns)
        self.assertNotIn("mean_Gene3", result.columns)


if __name__ == "__main__":
    unittest.main()
