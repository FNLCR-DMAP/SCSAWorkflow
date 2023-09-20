import pandas as pd
import unittest
from spac.data_utils import append_annotation


class TestAppendAnnotation(unittest.TestCase):

    def test_valid_annotation(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = [{'C': 'Alice'}, {'D': 30}]
        result = append_annotation(data.copy(), annotation)
        self.assertTrue(all(col in result.columns for col in ['C', 'D']))

    def test_invalid_annotation_type(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = 'Invalid'  # Should be a list
        with self.assertRaises(ValueError) as context:
            append_annotation(data.copy(), annotation)
        self.assertEqual(
            str(context.exception),
            "Annotation must be provided as a list."
        )

    def test_invalid_annotation_entry(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = [{'C': 'Alice'}, 'Invalid']  # Invalid entry in the list
        with self.assertRaises(ValueError) as context:
            append_annotation(data.copy(), annotation)
        self.assertEqual(
            str(context.exception),
            "The entry Invalid is not a dictionary, please check."
        )

    def test_invalid_annotation_value_type(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = [{'C': 'Alice'}, {'D': [1, 2, 3]}]  # Invalid value type
        with self.assertRaises(ValueError) as context:
            append_annotation(data.copy(), annotation)

        expecated_str = "The value [1, 2, 3] in D is not a single" + \
            " string or numeric value, please check."

        self.assertEqual(
            str(context.exception),
            expecated_str
        )

    def test_existing_column(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = [{'A': 'Alice'}]  # A already exists in the DataFrame
        with self.assertRaises(ValueError) as context:
            append_annotation(data.copy(), annotation)
        self.assertEqual(
            str(context.exception), "'A' already exists in the DataFrame.")


if __name__ == '__main__':
    unittest.main()