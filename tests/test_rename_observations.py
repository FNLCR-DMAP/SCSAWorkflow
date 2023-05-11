import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")


import unittest
import anndata
import pandas as pd
import numpy as np
from spac.transformations import rename_observations


class TestRenameObservations(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object
        obs = pd.DataFrame(
            {"phenograph": ["0", "1", "0", "2", "1", "2"]},
            index=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
        )
        self.adata = anndata.AnnData(X=np.random.randn(6, 3), obs=obs)

    def test_typical_case(self):
        mappings = {
            "0": "group_8",
            "1": "group_2",
            "2": "group_6"
        }
        dest_observation = "renamed_observations"
        result = rename_observations(self.adata, "phenograph", dest_observation, mappings)
        expected = pd.Series(
            ["group_8", "group_2", "group_8", "group_6", "group_2", "group_6"],
            index=self.adata.obs.index,
            name=dest_observation,
            dtype="category"
        )
        pd.testing.assert_series_equal(result.obs[dest_observation], expected)

    def test_missing_src_observation(self):
        with self.assertRaises(ValueError):
            rename_observations(self.adata, "missing_src", "new_dest", {"0": "group_8"})

    def test_existing_dest_observation(self):
        with self.assertRaises(ValueError):
            rename_observations(self.adata, "phenograph", "phenograph", {"0": "group_8"})

    def test_invalid_mappings(self):
        with self.assertRaises(ValueError):
            rename_observations(self.adata, "phenograph", "new_dest", {"5": "group_8"})

    def test_partial_mappings(self):
        mappings = {
            "0": "group_8",
            "1": "group_2"
        }
        dest_observation = "renamed_observations"
        result = rename_observations(self.adata, "phenograph", dest_observation, mappings)
        expected = pd.Series(
            ["group_8", "group_2", "group_8", "2", "group_2", "2"],
            index=self.adata.obs.index,
            name=dest_observation,
            dtype="category"
        )
        pd.testing.assert_series_equal(result.obs[dest_observation], expected)


    def test_rename_observations_basic(self):
        # Create an AnnData object with 3 cells and dummy gene expression data
        X = np.random.rand(3, 4)
        obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        obs["original_phenotype"] = ["A", "B", "A"]

        adata = anndata.AnnData(X=X, obs=obs)

        # Map two of these cells to a group called "group_0" and one cell to a group called "group_1"
        new_phenotypes = {
            "A": "group_0",
            "B": "group_1",
        }

        # Use the rename_observations function
        adata = rename_observations(adata, "original_phenotype", "renamed_clusters", new_phenotypes)

        # Verify that the mapping has taken place correctly
        self.assertTrue(all(adata.obs["renamed_clusters"] == ["group_0", "group_1", "group_0"]))

    def create_test_anndata(self, n_cells=100, n_genes=10):
        X = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(index=np.arange(n_cells))
        obs['phenograph'] = np.random.choice([0, 1, 2], size=n_cells)
        var = pd.DataFrame(index=np.arange(n_genes))
        return anndata.AnnData(X=X, obs=obs, var=var)

    def test_rename_observations(self):
        test_adata = self.create_test_anndata()

        # Define the new_phenotypes dictionary
        new_phenotypes = {
            "0": "group_1",
            "1": "group_2",
            "2": "group_3",
        }

        # Apply the rename_observations function
        adata = rename_observations(test_adata, "phenograph", "renamed_clusters", new_phenotypes)

        # Check if the new_column_name exists in adata.obs
        self.assertIn("renamed_clusters", adata.obs.columns)

        # Check if the new cluster names have been assigned correctly
        expected_cluster_names = ["group_1", "group_2", "group_3"]
        self.assertTrue(set(adata.obs["renamed_clusters"]).issubset(set(expected_cluster_names)))

        # Test with non-existent keys in new_phenotypes
        new_phenotypes = {
            "0": "group_1",
            "1": "group_2",
            "3": "group_3",
        }

        with self.assertRaises(ValueError):
            adata = rename_observations(test_adata, "phenograph", "renamed_clusters", new_phenotypes)


if __name__ == "__main__":
    unittest.main()
