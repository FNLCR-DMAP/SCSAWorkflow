import unittest
import anndata
import pandas as pd
import numpy as np
from spac.transformations import rename_annotations
import warnings


class TestRenameAnnotations(unittest.TestCase):
    def setUp(self):
        """Set up a sample AnnData object for testing."""
        annotation = pd.DataFrame(
            {"phenograph": ["0", "1", "0", "2", "1", "2"]},
            index=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
        )
        self.adata = anndata.AnnData(X=np.random.randn(6, 3), obs=annotation)

    def test_handle_missing_mappings(self):
        """
        Test how the function handles labels without corresponding mappings.
        """
        mappings = {"0": "group_8", "1": "group_2"}
        dest_annotation = "renamed_annotations"

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            rename_annotations(
                self.adata, "phenograph", dest_annotation, mappings
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            expected_msg = (
                "Missing mappings for the following labels: ['2']. "
                "They will be set to NaN in the 'renamed_annotations' column."
            )
            self.assertIn(expected_msg, str(w[-1].message))

        # Check that the missing mappings are set to NaN
        self.assertTrue(self.adata.obs[dest_annotation].isna().any())

    def test_multiple_annotations_to_one_group(self):
        """Test case where two annotations are mapped to one group."""
        data_matrix = np.random.rand(3, 4)
        annotation = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        annotation["original_phenotype"] = ["A", "B", "A"]

        adata = anndata.AnnData(X=data_matrix, obs=annotation)
        new_phenotypes = {"A": "group_0", "B": "group_0"}

        rename_annotations(
            adata, "original_phenotype", "renamed_clusters", new_phenotypes
        )

        renamed_clusters = adata.obs["renamed_clusters"]
        self.assertTrue(
            all(renamed_clusters == ["group_0", "group_0", "group_0"])
        )

    def test_typical_case(self):
        """Test rename_labels with typical parameters."""
        mappings = {"0": "group_8", "1": "group_2", "2": "group_6"}
        dest_annotation = "renamed_annotations"
        rename_annotations(self.adata, "phenograph", dest_annotation, mappings)
        expected = pd.Series(
            ["group_8", "group_2", "group_8", "group_6", "group_2", "group_6"],
            index=self.adata.obs.index,
            name=dest_annotation,
            dtype="category"
        )
        pd.testing.assert_series_equal(
            self.adata.obs[dest_annotation], expected
        )

    def create_test_anndata(self, n_cells=100, n_genes=10):
        data_matrix = np.random.rand(n_cells, n_genes)
        annotation = pd.DataFrame(index=np.arange(n_cells))
        annotation['phenograph'] = np.random.choice([0, 1, 2], size=n_cells)
        var = pd.DataFrame(index=np.arange(n_genes))
        return anndata.AnnData(X=data_matrix, obs=annotation, var=var)

    def test_rename_labels(self):
        test_adata = self.create_test_anndata()
        new_phenotypes = {"0": "group_1", "1": "group_2", "2": "group_3"}
        rename_annotations(
            test_adata, "phenograph", "renamed_clusters", new_phenotypes
        )

        self.assertIn("renamed_clusters", test_adata.obs.columns)
        expected_cluster_names = ["group_1", "group_2", "group_3"]
        renamed_clusters_set = set(test_adata.obs["renamed_clusters"])
        expected_clusters_set = set(expected_cluster_names)
        self.assertTrue(renamed_clusters_set.issubset(expected_clusters_set))

    def test_rename_labels_with_missing_key(self):
        test_adata = self.create_test_anndata()
        new_phenotypes = {"0": "group_1", "1": "group_2", "3": "group_3"}

        rename_annotations(
            test_adata, "phenograph", "renamed_clusters", new_phenotypes
        )

        # Check that the missing mappings are set to NaN
        unmapped_cells = test_adata.obs["phenograph"] == '2'
        self.assertTrue(
            test_adata.obs.loc[unmapped_cells, "renamed_clusters"].isna().all()
        )

    if __name__ == "__main__":
        unittest.main()
