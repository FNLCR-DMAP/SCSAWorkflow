import unittest
import pandas as pd
import anndata as ad
from spac.utils import annotation_category_relations


class TestAnnotationCategoryRelations(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object for testing
        data = {
            "feature": pd.Series([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                ], dtype='int64')
        }
        self.adata = ad.AnnData(
            X=pd.DataFrame(data, dtype=data["feature"].dtype),
            obs={"annotation1": [
                "A", "A", "B", "B", "B", "C", "C", "D", "D", "D", "D"
                ],
                 "annotation2": [
                "b", "b", "a", "a", "a", "d", "c", "d", "c", "d", "c"
                ]}
        )

    def test_output_type(self):
        result = annotation_category_relations(
            self.adata,
            'annotation1',
            'annotation2'
        )

        # Assert the result is a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)

    def test_annotation_category_relations(self):
        result = annotation_category_relations(
            self.adata,
            'annotation1',
            'annotation2'
        )

        # Create a ground truth DataFrame for checking the result
        ground_truth = pd.DataFrame({
            'Source': ['A', 'B', 'C', 'C', 'D', 'D'],
            'Target': ['b', 'a', 'c', 'd', 'c', 'd'],
            'Count': [2, 3, 1, 1, 2, 2]
        })

        # Assert the result is equal to the ground truth
        pd.testing.assert_frame_equal(result, ground_truth)


if __name__ == "__main__":
    unittest.main()
