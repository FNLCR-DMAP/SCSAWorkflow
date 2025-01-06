import unittest
import numpy as np
import pandas as pd
import anndata as ad
from spac.utils import check_label


class TestCheckLabel(unittest.TestCase):

    def setUp(self):
        # Create a minimal AnnData object
        # with a single annotation column and a few labels
        data = {"feature1": [1, 3], "feature2": [2, 4]}
        self.adata = ad.AnnData(
            X=pd.DataFrame(data),
            obs={"cell_type": ["B_cell", "T_cell"]}
        )

    def test_labels_not_list_or_str(self):
        # Test invalid labels input type
        with self.assertRaises(ValueError) as context:
            check_label(self.adata, annotation="cell_type", labels=123)
        expected_msg = (
            "The 'labels' parameter should be a string or a list of strings."
        )
        self.assertEqual(str(context.exception), expected_msg)

    def test_valid_label(self):
        # Test with a valid single label that should exist
        self.assertIsNone(check_label(self.adata, annotation="cell_type",
                                      labels="B_cell"))

    def test_valid_list_labels(self):
        # Test with a valid list of labels that should exist
        self.assertIsNone(check_label(self.adata, annotation="cell_type",
                                      labels=["B_cell", "T_cell"]))

    def test_invalid_label(self):
        # Test with a single invalid label
        with self.assertRaises(ValueError) as context:
            check_label(self.adata, annotation="cell_type",
                        labels="unknown_label")
        expected_msg = (
            "The label(s) in the annotation 'cell_type' 'unknown_label' "
            "does not exist in the provided dataset.\n"
            "Existing label(s) in the annotation 'cell_type's are:\n"
            "B_cell\nT_cell"
        )
        self.assertEqual(str(context.exception), expected_msg)

    def test_mixed_labels(self):
        # Test with a list containing both valid and invalid labels
        with self.assertRaises(ValueError) as context:
            check_label(self.adata, annotation="cell_type",
                        labels=["B_cell", "Not_exist"])
        expected_msg = (
            "The label(s) in the annotation 'cell_type' 'Not_exist' "
            "does not exist in the provided dataset.\n"
            "Existing label(s) in the annotation 'cell_type's are:\n"
            "B_cell\nT_cell"
        )
        self.assertEqual(str(context.exception), expected_msg)

    def test_new_labels_should_not_exist(self):
        # Test with labels that should not exist
        # These labels do not exist in the annotation, so should pass
        self.assertIsNone(
            check_label(self.adata, annotation="cell_type",
                        labels=["new_label1", "new_label2"],
                        should_exist=False)
        )

    def test_existing_label_should_not_exist(self):
        # Test a scenario where the label does exist but should not
        with self.assertRaises(ValueError) as context:
            check_label(self.adata, annotation="cell_type",
                        labels=["new_label1", "T_cell"],
                        should_exist=False)
        expected_msg = (
            "The label(s) in the annotation 'cell_type' 'T_cell' exist in the "
            "provided dataset.\n"
            "Existing label(s) in the annotation 'cell_type's are:\n"
            "B_cell\nT_cell"
        )
        self.assertEqual(str(context.exception), expected_msg)


if __name__ == "__main__":
    unittest.main()
