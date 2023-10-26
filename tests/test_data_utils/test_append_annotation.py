import pandas as pd
import unittest
from spac.data_utils import append_annotation


class TestAppendAnnotation(unittest.TestCase):

    def test_valid_annotation(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = {'C': 'Alice', 'D': 30}
        result = append_annotation(data.copy(), annotation)
        self.assertTrue(all(col in result.columns for col in ['C', 'D']))
        # Check if 'C' column always contains 'Alice'
        self.assertTrue(all(result['C'] == 'Alice'))
        # Check if 'D' column always contains 30
        self.assertTrue(all(result['D'] == 30))

    def test_invalid_annotation_type(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = 'Invalid'  # Should be a dictionary
        with self.assertRaises(ValueError) as context:
            append_annotation(data.copy(), annotation)
        self.assertEqual(
            str(context.exception),
            "Annotation must be provided as a dictionary."
        )

    def test_invalid_annotation_key_type(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = {'C': 'Alice', 1: 'Alice'}  # Invalid key in the dict
        with self.assertRaises(ValueError) as context:
            append_annotation(data.copy(), annotation)

        expected_str = "The key 1 is not " + \
            "a single string, please check."

        self.assertEqual(
            str(context.exception),
            expected_str
        )

    def test_invalid_annotation_value_type(self):
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        annotation = {'C': 'Alice', 'D': [1, 2, 3]}  # Invalid value type
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
        annotation = {'A': 'Alice'}  # A already exists in the DataFrame
        with self.assertRaises(ValueError) as context:
            append_annotation(data.copy(), annotation)
        self.assertEqual(
            str(context.exception), "'A' already exists in the DataFrame.")


if __name__ == '__main__':
    unittest.main()