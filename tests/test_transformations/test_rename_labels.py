import unittest
import anndata
import pandas as pd
import numpy as np
from spac.transformations import rename_labels


class TestRenameLabels(unittest.TestCase):
    def setUp(self):
        """Set up a sample AnnData object for testing."""
        annotation = pd.DataFrame(
            {"phenograph": ["0", "1", "0", "2", "1", "2"]},
            index=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
        )
        self.adata = anndata.AnnData(X=np.random.randn(6, 3), obs=annotation)

    def test_typical_case(self):
        """Test rename_labels with typical parameters."""
        mappings = {"0": "group_8", "1": "group_2", "2": "group_6"}
        dest_annotation = "renamed_annotations"
        result = rename_labels(
            self.adata, "phenograph", dest_annotation, mappings
        )
        expected = pd.Series(
            ["group_8", "group_2", "group_8", "group_6", "group_2", "group_6"],
            index=self.adata.obs.index,
            name=dest_annotation,
            dtype="category"
        )
        pd.testing.assert_series_equal(result.obs[dest_annotation], expected)

    def test_missing_src_annotation(self):
        """Test rename_labels with a missing source annotation."""
        with self.assertRaises(ValueError):
            rename_labels(
                self.adata,
                "missing_src",
                "new_dest",
                {"0": "group_8"}
            )

    def test_existing_dest_annotation(self):
        """
        Test rename_labels with an existing destination annotation.
        """
        with self.assertRaises(ValueError):
            rename_labels(
                self.adata,
                "phenograph",
                "phenograph",
                {"0": "group_8"}
            )

    def test_invalid_mappings(self):
        """Test rename_labels with invalid mappings."""
        with self.assertRaises(ValueError):
            rename_labels(
                self.adata,
                "phenograph",
                "new_dest",
                {"5": "group_8"}
            )

    def test_rename_labels_basic(self):
        """Test basic functionality of rename_labels."""
        data_matrix = np.random.rand(3, 4)
        annotation = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        annotation["original_phenotype"] = ["A", "B", "A"]

        adata = anndata.AnnData(X=data_matrix, obs=annotation)

        new_phenotypes = {"A": "group_0", "B": "group_1"}

        adata = rename_labels(
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
        annotation = pd.DataFrame(index=np.arange(n_cells))
        annotation['phenograph'] = np.random.choice([0, 1, 2], size=n_cells)
        var = pd.DataFrame(index=np.arange(n_genes))
        return anndata.AnnData(X=data_matrix, obs=annotation, var=var)

    def test_rename_labels(self):
        """Test rename_labels with generated AnnData object."""
        test_adata = self.create_test_anndata()

        new_phenotypes = {
            "0": "group_1",
            "1": "group_2",
            "2": "group_3",
        }

        adata = rename_labels(
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
            adata = rename_labels(
                test_adata, "phenograph", "renamed_clusters", new_phenotypes
            )

    def test_multiple_annotations_to_one_group(self):
        """Test case where two annotations are mapped to one group."""
        data_matrix = np.random.rand(3, 4)
        annotation = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        annotation["original_phenotype"] = ["A", "B", "A"]

        adata = anndata.AnnData(X=data_matrix, obs=annotation)

        new_phenotypes = {"A": "group_0", "B": "group_0"}

        adata = rename_labels(
            adata, "original_phenotype", "renamed_clusters", new_phenotypes
        )

        renamed_clusters = adata.obs["renamed_clusters"]
        self.assertTrue(
            all(renamed_clusters == ["group_0", "group_0", "group_0"])
        )

    def test_not_all_categories_covered(self):
        """
        Test rename_labels with mappings that do not cover
        all unique values in the source annotation.
        """
        mappings = {"0": "group_8", "1": "group_2"}
        with self.assertRaises(ValueError) as cm:
            rename_labels(
                self.adata,
                "phenograph",
                "incomplete_dest",
                mappings
            )
        self.assertEqual(
            str(cm.exception),
            "Not all unique values in the source annotation are "
            "covered by the mappings. "
            "Please ensure that the mappings cover all unique values."
        )


if __name__ == "__main__":
    unittest.main()
