import unittest
import numpy as np
from spac.transformations import apply_per_batch


class TestApplyPerBatch(unittest.TestCase):

    def setUp(self):
        # Setup data and batch annotations
        self.data = np.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0],
                              [10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0],
                              [16.0, 17.0, 18.0]])

    def test_arcsinh_transformation(self):
        # Test with different orders of batch labels
        annotations = [
            np.array(['batch1', 'batch1', 'batch2', 'batch2',
                      'batch3', 'batch3']),
            np.array(['batch2', 'batch2', 'batch1', 'batch1',
                      'batch3', 'batch3']),
            np.array(['batch3', 'batch3', 'batch2', 'batch2',
                      'batch1', 'batch1'])
        ]
        co_factor = 5.0
        expected_data = np.arcsinh(self.data / co_factor)

        for annotation in annotations:
            transformed_data = apply_per_batch(self.data, annotation,
                                               'arcsinh_transformation',
                                               co_factor=co_factor)
            np.testing.assert_array_almost_equal(transformed_data,
                                                 expected_data, decimal=6)

    def test_arcsinh_transformation_percentile(self):
        # Test with different orders of batch labels
        annotations = [
            np.array(['batch1', 'batch1', 'batch2', 'batch2',
                      'batch3', 'batch3']),
            np.array(['batch2', 'batch2', 'batch1', 'batch1',
                      'batch3', 'batch3']),
            np.array(['batch3', 'batch3', 'batch2', 'batch2',
                      'batch1', 'batch1'])
        ]
        percentile = 20

        perc_values = [
            np.percentile(self.data[:2], percentile, axis=0),
            np.percentile(self.data[2:4], percentile, axis=0),
            np.percentile(self.data[4:], percentile, axis=0)
        ]

        expected_data = np.vstack([
            np.arcsinh(self.data[:2] / perc_values[0]),
            np.arcsinh(self.data[2:4] / perc_values[1]),
            np.arcsinh(self.data[4:] / perc_values[2])
        ])

        for annotation in annotations:
            transformed_data = apply_per_batch(self.data, annotation,
                                               'arcsinh_transformation',
                                               percentile=percentile)
            np.testing.assert_array_almost_equal(transformed_data,
                                                 expected_data, decimal=6)

    def test_arcsinh_transformation_both_cofactor_percentile(self):
        # Test apply_per_batch raises ValueError when both co_factor and
        # percentile are provided
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, np.array(['batch1', 'batch1',
                                                 'batch2', 'batch2',
                                                 'batch3', 'batch3']),
                            'arcsinh_transformation', co_factor=5.0,
                            percentile=20)
        self.assertEqual(
            str(context.exception),
            "Please specify either co_factor or percentile, not both."
        )

    def test_arcsinh_transformation_none_cofactor_percentile(self):
        # Test apply_per_batch raises ValueError when neither co_factor nor
        # percentile is provided
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, np.array(['batch1', 'batch1',
                                                 'batch2', 'batch2',
                                                 'batch3', 'batch3']),
                            'arcsinh_transformation')
        self.assertEqual(
            str(context.exception),
            "Either co_factor or percentile must be provided."
        )

    def test_normalize_features(self):
        # Test apply_per_batch with normalize_features_core
        annotations = [
            np.array(['batch1', 'batch1', 'batch2', 'batch2',
                      'batch3', 'batch3']),
            np.array(['batch2', 'batch2', 'batch1', 'batch1',
                      'batch3', 'batch3']),
            np.array(['batch3', 'batch3', 'batch2', 'batch2',
                      'batch1', 'batch1'])
        ]
        low_quantile = 0.1
        high_quantile = 0.9

        # Hard-code the low and high quantile values for each batch
        # low_values = np.quantile(self.data, low_quantile, axis=0)
        # high_values = np.quantile(self.data, high_quantile, axis=0)
        low_values = [
            np.array([1.3, 2.3, 3.3]),  # Low quantile for batch 1
            np.array([7.3, 8.3, 9.3]),  # Low quantile for batch 2
            np.array([13.3, 14.3, 15.3])  # Low quantile for batch 3
        ]
        high_values = [
            np.array([3.7, 4.7, 5.7]),  # High quantile for batch 1
            np.array([9.7, 10.7, 11.7]),  # High quantile for batch 2
            np.array([15.7, 16.7, 17.7])  # High quantile for batch 3
        ]

        # Manually compute the expected normalized data
        expected_data = np.vstack([
            (np.clip(self.data[:2], low_values[0], high_values[0]) -
             low_values[0]) / (high_values[0] - low_values[0]),
            (np.clip(self.data[2:4], low_values[1], high_values[1]) -
             low_values[1]) / (high_values[1] - low_values[1]),
            (np.clip(self.data[4:], low_values[2], high_values[2]) -
             low_values[2]) / (high_values[2] - low_values[2])
        ])

        for annotation in annotations:
            transformed_data = apply_per_batch(self.data, annotation,
                                               'normalize_features',
                                               low_quantile=low_quantile,
                                               high_quantile=high_quantile)
            np.testing.assert_array_almost_equal(transformed_data,
                                                 expected_data, decimal=6)

    def test_normalize_features_raises_valueerror(self):
        # Test apply_per_batch with normalize_features_core raising ValueError
        low_quantile = 1.1
        high_quantile = 0.9
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, np.array(['batch1', 'batch1',
                                                 'batch2', 'batch2',
                                                 'batch3', 'batch3']),
                            'normalize_features', low_quantile=low_quantile,
                            high_quantile=high_quantile)
        self.assertEqual(
            str(context.exception),
            "The low quantile should be smaller than the high quantile, "
            "current values are:\n"
            f"low quantile: {low_quantile}\n"
            f"high quantile: {high_quantile}"
        )

    def test_invalid_method(self):
        # Test apply_per_batch with an invalid method
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, np.array(['batch1', 'batch1',
                                                 'batch2', 'batch2',
                                                 'batch3', 'batch3']),
                            'invalid_method')
        self.assertEqual(str(context.exception),
                         "method must be 'arcsinh_transformation' or "
                         "'normalize_features'")

    def test_data_integrity_check(self):
        # Test apply_per_batch with invalid data types
        invalid_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with self.assertRaises(ValueError) as context:
            apply_per_batch(invalid_data, np.array(['batch1', 'batch1',
                                                    'batch2', 'batch2',
                                                    'batch3', 'batch3']),
                            'arcsinh_transformation')
        self.assertEqual(str(context.exception),
                         "data and annotation must be numpy arrays")

        invalid_annotation = ['batch1', 'batch1', 'batch2']
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, invalid_annotation,
                            'arcsinh_transformation')
        self.assertEqual(str(context.exception),
                         "data and annotation must be numpy arrays")

        invalid_data_shape = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError) as context:
            apply_per_batch(invalid_data_shape, np.array(['batch1', 'batch1',
                                                          'batch2', 'batch2',
                                                          'batch3', 'batch3']),
                            'arcsinh_transformation')
        self.assertEqual(
            str(context.exception),
            "data and annotation must have the same number of rows"
        )


if __name__ == '__main__':
    unittest.main()
