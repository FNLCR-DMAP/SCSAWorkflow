import unittest
import pandas as pd
import numpy as np
from spac.data_utils import bin2cat


class TestBin2Cat(unittest.TestCase):
    def test_bin2cat(self):
        # Sample input data
        data = pd.DataFrame({
            'A_0': [1, 0, 0, 0, 0],
            'A_1': [0, 1, 0, 0, 0],
            'B_0': [0, 0, 1, 0, 0],
            'B_1': [0, 0, 0, 1, 0]
        })
        one_hot_annotations = ['A.*', 'B.*']
        new_annotation = 'new_category'

        # Expected output data
        expected_data = pd.DataFrame({
            'A_0': [1, 0, 0, 0, 0],
            'A_1': [0, 1, 0, 0, 0],
            'B_0': [0, 0, 1, 0, 0],
            'B_1': [0, 0, 0, 1, 0],
            'new_category': ['A_0', 'A_1', 'B_0', 'B_1', np.nan],
        })

        # Call the function
        result = bin2cat(data, one_hot_annotations, new_annotation)

        # Check if the result matches the expected output
        pd.testing.assert_frame_equal(result, expected_data)

    def test_bin2cat_duplicate_new_annotation(self):
        # Sample input data
        data = pd.DataFrame({
            'A_0': [1, 0, 0],
            'A_1': [0, 1, 0],
        })
        one_hot_annotations = ['A.*']
        new_annotation = 'A_0'  # Duplicate name

        # Call the function and check for ValueError
        with self.assertRaises(ValueError):
            bin2cat(data, one_hot_annotations, new_annotation)

    def test_bin2cat_multiple_instances(self):
        # Sample input data
        data = pd.DataFrame({
            'A_0': [1, 0, 0],
            'A_1': [0, 1, 0],
            'B_0': [0, 0, 1],
            'B_1': [1, 0, 1],
        })
        one_hot_annotations = ['A.*', 'B.*']
        new_annotation = 'new_category'

        # Call the function and check for ValueError
        with self.assertRaises(ValueError) as cm:
            bin2cat(data, one_hot_annotations, new_annotation)
            expect_str = "Multiple instances found: " + \
                "Index(['A_0', 'B_1'], dtype='object')"

            self.assertEqual(
                str(cm.exception),
                expect_str
            )

    def test_bin2cat_no_columns_found(self):
        # Sample input data
        data = pd.DataFrame({
            'A_0': [1, 0, 0],
            'A_1': [0, 1, 0],
        })
        one_hot_annotations = ['B.*']  # Non-existent column
        new_annotation = 'new_category'

        # Call the function and check for ValueError
        with self.assertRaises(ValueError) as cm:
            bin2cat(data, one_hot_annotations, new_annotation)
            error_string = "No column was found in the dataframe " + \
                "with current regex pattern(s)."
            self.assertEqual(str(cm.exception), error_string)


if __name__ == '__main__':
    unittest.main()
