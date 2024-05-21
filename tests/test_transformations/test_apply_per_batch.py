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

        # Each row in the data belongs to a batch
        self.annotation = np.array(['batch1', 'batch1', 'batch2', 'batch2',
                                    'batch3', 'batch3'])

    def test_arcsinh_transformation(self):
        # Test apply_per_batch with arcsinh_transformation_core
        co_factor = 5.0
        # Manually calculated expected_data using arcsinh transformation
        # with co_factor
        # expected_data[i, j] = np.arcsinh(self.data[i, j] / co_factor)
        expected_data = np.array([[0.198690, 0.390035, 0.568824],
                                  [0.732668, 0.881374, 1.015973],
                                  [1.137982, 1.248983, 1.350441],
                                  [1.443635, 1.529661, 1.609438],
                                  [1.683743, 1.753229, 1.818446],
                                  [1.879864, 1.937879, 1.992836]])
        transformed_data = apply_per_batch(self.data, self.annotation,
                                           'arcsinh_transformation',
                                           co_factor=co_factor)
        # print("Transformed data for arcsinh_transformation:",
        #       transformed_data)
        # print("Expected data for arcsinh_transformation:", expected_data)
        np.testing.assert_array_almost_equal(transformed_data, expected_data,
                                             decimal=6)

    def test_arcsinh_transformation_percentile(self):
        # Test apply_per_batch with arcsinh_transformation_core using
        # percentile
        percentile = 20
        # Calculate the 20th percentile value for each batch
        # batch1_perc_value = np.percentile(self.data[:2], percentile, axis=0)
        # batch2_perc_value = np.percentile(self.data[2:4], percentile, axis=0)
        # batch3_perc_value = np.percentile(self.data[4:], percentile, axis=0)

        # batch1_perc_value = [1.6, 2.6, 3.6] (20th percentile of batch1)
        # batch2_perc_value = [7.6, 8.6, 9.6] (20th percentile of batch2)
        # batch3_perc_value = [13.6, 14.6, 15.6] (20th percentile of batch3)

        # Manually calculate expected_data using arcsinh transformation
        # with calculated co_factor based on percentile per batch
        # expected_data[i, j] = np.arcsinh(self.data[i, j] / perc_value[j])
        expected_data = np.array([[0.590144, 0.708461, 0.758486],
                                  [1.647231, 1.408696, 1.283796],
                                  [0.824434, 0.831170, 0.836482],
                                  [1.088041, 1.065625, 1.047593],
                                  [0.849831, 0.852014, 0.853914],
                                  [1.000822, 0.992971, 0.986088]])
        transformed_data = apply_per_batch(self.data, self.annotation,
                                           'arcsinh_transformation',
                                           percentile=percentile)
        # print("Transformed data for arcsinh_transformation_percentile:",
        #       transformed_data)
        # print("Expected data for arcsinh_transformation_percentile:",
        #       expected_data)
        np.testing.assert_array_almost_equal(transformed_data, expected_data,
                                             decimal=6)

    def test_arcsinh_transformation_both_cofactor_percentile(self):
        # Test apply_per_batch raises ValueError when both co_factor and
        # percentile are provided
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, self.annotation,
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
            apply_per_batch(self.data, self.annotation,
                            'arcsinh_transformation')
        self.assertEqual(
            str(context.exception),
            "Either co_factor or percentile must be provided."
        )

    def test_normalize_features(self):
        # Test apply_per_batch with normalize_features_core
        low_quantile = 0.1
        high_quantile = 0.9
        # Manually calculate the low and high quantile values for each batch
        batch1_low_values = np.percentile(self.data[:2], low_quantile * 100,
                                          axis=0)
        batch1_high_values = np.percentile(self.data[:2], high_quantile * 100,
                                           axis=0)
        batch2_low_values = np.percentile(self.data[2:4], low_quantile * 100,
                                          axis=0)
        batch2_high_values = np.percentile(self.data[2:4], high_quantile * 100,
                                           axis=0)
        batch3_low_values = np.percentile(self.data[4:], low_quantile * 100,
                                          axis=0)
        batch3_high_values = np.percentile(self.data[4:], high_quantile * 100,
                                           axis=0)

        # batch1_low_values = [1.3, 2.3, 3.3] (10th percentile of batch1)
        # batch1_high_values = [3.7, 4.7, 5.7] (90th percentile of batch1)
        # batch2_low_values = [7.3, 8.3, 9.3] (10th percentile of batch2)
        # batch2_high_values = [9.7, 10.7, 11.7] (90th percentile of batch2)
        # batch3_low_values = [13.3, 14.3, 15.3] (10th percentile of batch3)
        # batch3_high_values = [15.7, 16.7, 17.7] (90th percentile of batch3)

        # Manually calculate expected_data using normalization per batch
        batch1_expected_data = (
            (np.clip(self.data[:2], batch1_low_values, batch1_high_values) -
             batch1_low_values) / (batch1_high_values - batch1_low_values)
        )
        batch2_expected_data = (
            (np.clip(self.data[2:4], batch2_low_values, batch2_high_values) -
             batch2_low_values) / (batch2_high_values - batch2_low_values)
        )
        batch3_expected_data = (
            (np.clip(self.data[4:], batch3_low_values, batch3_high_values) -
             batch3_low_values) / (batch3_high_values - batch3_low_values)
        )

        expected_data = np.vstack([
            batch1_expected_data, batch2_expected_data, batch3_expected_data
        ])

        # expected_data = np.array([[0.0, 0.0, 0.0],
        #                          [1.0, 1.0, 1.0],
        #                          [0.0, 0.0, 0.0],
        #                          [1.0, 1.0, 1.0],
        #                          [0.0, 0.0, 0.0],
        #                          [1.0, 1.0, 1.0]])

        transformed_data = apply_per_batch(self.data, self.annotation,
                                           'normalize_features',
                                           low_quantile=low_quantile,
                                           high_quantile=high_quantile)
        # print("Transformed data for normalize_features:", transformed_data)
        # print("Expected data for normalize_features:", expected_data)
        np.testing.assert_array_almost_equal(transformed_data, expected_data,
                                             decimal=6)

    def test_normalize_features_raises_valueerror(self):
        # Test apply_per_batch with normalize_features_core raising ValueError
        low_quantile = 1.1
        high_quantile = 0.9
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, self.annotation, 'normalize_features',
                            low_quantile=low_quantile,
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
            apply_per_batch(self.data, self.annotation, 'invalid_method')
        self.assertEqual(str(context.exception), "method must be "
                                                 "'arcsinh_transformation' or "
                                                 "'normalize_features'")

    def test_data_integrity_check(self):
        # Test apply_per_batch with invalid data types
        invalid_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with self.assertRaises(ValueError) as context:
            apply_per_batch(invalid_data, self.annotation,
                            'arcsinh_transformation')
        self.assertEqual(str(context.exception), "data and annotation must be "
                                                 "numpy arrays")

        invalid_annotation = ['batch1', 'batch1', 'batch2']
        with self.assertRaises(ValueError) as context:
            apply_per_batch(self.data, invalid_annotation,
                            'arcsinh_transformation')
        self.assertEqual(str(context.exception), "data and annotation must be "
                                                 "numpy arrays")

        invalid_data_shape = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError) as context:
            apply_per_batch(invalid_data_shape, self.annotation,
                            'arcsinh_transformation')
        self.assertEqual(
            str(context.exception),
            "data and annotation must have the same number of rows"
        )


if __name__ == '__main__':
    unittest.main()
