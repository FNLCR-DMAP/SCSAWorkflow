import os
import sys
import unittest

import anndata
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
from spac.transformations import rename_observations


class TestRenameObservations(unittest.TestCase):
    def setUp(self):
        """Set up a sample AnnData object for testing."""
        obs = pd.DataFrame(
            {"phenograph": ["0", "1", "0", "2", "1", "2"]},
            index=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
        )
        self.adata = anndata.AnnData(X=np.random.randn(6, 3), obs=obs)

    def test_typical_case(self):
        """Test rename_observations with typical parameters."""
        mappings = {"0": "group_8", "1": "group_2", "2": "group_6"}
        dest_observation = "renamed_observations"
        result = rename_observations(
            self.adata, "phenograph", dest_observation, mappings
        )
        expected = pd.Series(
            ["group_8", "group_2", "group_8", "group_6", "group_2", "group_6"],
            index=self.adata.obs.index,
            name=dest_observation,
            dtype="category"
        )
        pd.testing.assert_series_equal(result.obs[dest_observation], expected)

    def test_missing_src_observation(self):
        """Test rename_observations with a missing source observation."""
        with self.assertRaises(ValueError):
            rename_observations(
                self.adata,
                "missing_src",
                "new_dest",
                {"0": "group_8"}
            )

    def test_existing_dest_observation(self):
        """
        Test rename_observations with an existing destination observation.
        """
        with self.assertRaises(ValueError):
            rename_observations(
                self.adata,
                "phenograph",
                "phenograph",
                {"0": "group_8"}
            )

    def test_invalid_mappings(self):
        """Test rename_observations with invalid mappings."""
        with self.assertRaises(ValueError):
            rename_observations(
                self.adata,
                "phenograph",
                "new_dest",
                {"5": "group_8"}
            )

    def test_partial_mappings(self):
        """Test rename_observations with partial mappings."""
        mappings = {"0": "group_8", "1": "group_2"}
        dest_observation = "renamed_observations"
        result = rename_observations(
            self.adata, "phenograph", dest_observation, mappings
        )
        expected = pd.Series(
            ["group_8", "group_2", "group_8", "2", "group_2", "2"],
            index=self.adata.obs.index,
            name=dest_observation,
            dtype="category"
        )
        pd.testing.assert_series_equal(result.obs[dest_observation], expected)

    def test_rename_observations_basic(self):
        """Test basic functionality of rename_observations."""
        data_matrix = np.random.rand(3, 4)
        obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        obs["original_phenotype"] = ["A", "B", "A"]

        adata = anndata.AnnData(X=data_matrix, obs=obs)

        new_phenotypes = {"A": "group_0", "B": "group_1"}

        adata = rename_observations(
            adata, "original_phenotype", "renamed_clusters", new_phenotypes
        )
        self.assertTrue(
            all(
                adata.obs["renamed_clusters"] ==
                ["group_0", "group_1", "group_0"]
            )
        )

    def create_test_anndata(self, n_cells=100, n_genes=10):
        """Create a test AnnData object with random gene expressions."""
        data_matrix = np.random.rand(n_cells, n_genes)
        obs = pd.DataFrame(index=np.arange(n_cells))
        obs['phenograph'] = np.random.choice([0, 1, 2], size=n_cells)
        var = pd.DataFrame(index=np.arange(n_genes))
        return anndata.AnnData(X=data_matrix, obs=obs, var=var)

    def test_rename_observations(self):
        """Test rename_observations with generated AnnData object."""
        test_adata = self.create_test_anndata()

        new_phenotypes = {
            "0": "group_1",
            "1": "group_2",
            "2": "group_3",
        }

        adata = rename_observations(
            test_adata, "phenograph", "renamed_clusters", new_phenotypes
        )

        self.assertIn("renamed_clusters", adata.obs.columns)

        expected_cluster_names = ["group_1", "group_2", "group_3"]
        renamed_clusters_set = set(adata.obs["renamed_clusters"])
        expected_clusters_set = set(expected_cluster_names)
        self.assertTrue(
            renamed_clusters_set.issubset(expected_clusters_set)
        )

        new_phenotypes = {
            "0": "group_1",
            "1": "group_2",
            "3": "group_3",
        }

        with self.assertRaises(ValueError):
            adata = rename_observations(
                test_adata, "phenograph", "renamed_clusters", new_phenotypes
            )

    def test_multiple_observations_to_one_group(self):
        """Test case where two observations are mapped to one group."""
        data_matrix = np.random.rand(3, 4)
        obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        obs["original_phenotype"] = ["A", "B", "A"]

        adata = anndata.AnnData(X=data_matrix, obs=obs)

        new_phenotypes = {"A": "group_0", "B": "group_0"}

        adata = rename_observations(
            adata, "original_phenotype", "renamed_clusters", new_phenotypes
        )

        renamed_clusters = adata.obs["renamed_clusters"]
        self.assertTrue(
            all(renamed_clusters == ["group_0", "group_0", "group_0"])
        )


if __name__ == "__main__":
    unittest.main()
