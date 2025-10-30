import unittest
import numpy as np
import pandas as pd
import anndata as ad
from spac.utils import check_annotation


class TestCheckAnnotation(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object for testing
        data = {
            "feature1": [1, 3],
            "feature2": [2, 4]
        }
        self.adata = ad.AnnData(
            X=pd.DataFrame(data),
            layers={"table1": np.array([[5, 6], [7, 8]]),
                    "table2": np.array([[9, 10], [11, 12]])},
            obs={"annotation1": [1, 2],
                 "annotation2": [3, 4]}
        )
        print("minor")

    def test_wrong_adata_type(self):
        with self.assertRaises(TypeError) as context:
            check_annotation("not an AnnData object")
        self.assertEqual(
            str(context.exception),
            "Input dataset should be an instance of anndata.AnnData, "
            "please check the input dataset source."
        )

    def test_set_parameter_name(self):
        with self.assertRaises(ValueError) as context:
            check_annotation(self.adata,
                             annotations=1,
                             parameter_name="test_parameter_name")
        self.assertEqual(
            str(context.exception),
            "The 'test_parameter_name' parameter "
            "should be a string or a list of strings."
        )


    def test_valid_annotation(self):
        # Test with valid annotation
        self.assertIsNone(check_annotation(self.adata,
                                           annotations="annotation1"))

    def test_valid_list_annotation(self):
        # Test with list input for annotation
        self.assertIsNone(check_annotation(self.adata,
                                           annotations=["annotation1",
                                                        "annotation2"]))

    def test_invalid_annotation(self):
        # Test with invalid annotation
        with self.assertRaises(ValueError):
            check_annotation(self.adata, annotations="invalid_annotation")
        with self.assertRaises(ValueError):
            check_annotation(self.adata,
                             annotations=["annotation1", "invalid_annotation"])

    def test_missing_annotations(self):
        with self.assertRaises(ValueError) as context:
            check_annotation(
                self.adata,
                annotations=["annotation1", "missing_annotation2"])
        self.assertEqual(
            "The annotation 'missing_annotation2' does not "
            "exist in the provided dataset.\n"
            "Existing annotations are:\nannotation1\nannotation2",
            str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            check_annotation(
                self.adata,
                annotations=["missing_annotation1", "annotation2"])
        self.assertEqual(
            "The annotation 'missing_annotation1' does not "
            "exist in the provided dataset.\n"
            "Existing annotations are:\nannotation1\nannotation2",
            str(context.exception)
        )

    def test_valid_new_annotation(self):
        # Test with valid new annotations
        self.assertIsNone(check_annotation(
            self.adata,
            annotations=["new_annotation1", "new_annotation2"],
            should_exist=False))

    def test_invalid_new_annotation(self):
        with self.assertRaises(ValueError):
            check_annotation(
                self.adata,
                annotations=["new_annotation1", "annotation2"],
                should_exist=False)


if __name__ == "__main__":
    unittest.main()
