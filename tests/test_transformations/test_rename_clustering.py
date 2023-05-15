import anndata
import unittest
import numpy as np
import pandas as pd
from spac.transformations import rename_clustering


class TestRenameClustering(unittest.TestCase):

    def create_test_anndata(self, n_cells=100, n_genes=10):
        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(index=np.arange(n_cells))
        obs['phenograph'] = np.random.choice([0, 1, 2], size=n_cells)
        var = pd.DataFrame(index=np.arange(n_genes))
        return anndata.AnnData(X=X, obs=obs, var=var)

    def test_rename_clustering(self):
        test_adata = self.create_test_anndata()

        # Define the new_phenotypes dictionary
        new_phenotypes = {
            "0": "group_1",
            "1": "group_2",
            "2": "group_3",
        }

        # Apply the rename_clustering function
        adata = rename_clustering(
            test_adata,
            "phenograph",
            new_phenotypes,
            "renamed_clusters")

        # Check if the new_column_name exists in adata.obs
        self.assertIn("renamed_clusters", adata.obs.columns)

        # Check if the new cluster names have been assigned correctly
        expected_cluster_names = ["group_1", "group_2", "group_3"]
        self.assertTrue(
            set(
                adata.obs["renamed_clusters"]
                ).issubset(set(expected_cluster_names))
            )

        # Test with non-existent keys in new_phenotypes
        new_phenotypes = {
            "0": "group_1",
            "1": "group_2",
            "3": "group_3",
        }

        with self.assertRaises(ValueError):
            adata = rename_clustering(
                test_adata,
                "phenograph",
                new_phenotypes,
                "renamed_clusters"
                )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
