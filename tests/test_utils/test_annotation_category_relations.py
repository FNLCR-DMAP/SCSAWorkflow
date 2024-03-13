import unittest
import pandas as pd
import anndata as ad
from spac.utils import annotation_category_relations


class TestAnnotationCategoryRelations(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object for testing
        data = {
            "feature": pd.Series([
                0, 1, 2, 3
                ], dtype='int64')
        }
        self.adata = ad.AnnData(
            X=pd.DataFrame(data, dtype=data["feature"].dtype),
            obs={"annotation1": [
                "a", "a", "b", "b"
                ],
                 "annotation2": [
                "x", "y", "y", "y"
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
            'source': ['a', 'a', 'b'],
            'target': ['x', 'y', 'y'],
            'count': [1, 1, 2],
            'percentage_source': [50.0, 50.0, 100.0],
            'percentage_target': [100.0, 33.3, 66.7]
        })

        # Assert the result is equal to the ground truth
        pd.testing.assert_frame_equal(result, ground_truth)


if __name__ == "__main__":
    unittest.main()
