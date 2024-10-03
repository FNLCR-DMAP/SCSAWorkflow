import unittest
import numpy as np
from spac.transformations import normalize_features_core


class TestNormalizeFeaturesCore(unittest.TestCase):

    def setUp(self):
        # Create simple 2D numpy array datasets
        self.data_normalization = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]
        ])
        # Create a random dataset with 100 samples and 5 features
        np.random.seed(42)
        self.random_data = np.random.rand(100, 5) * 100

    def test_invalid_high_quantile(self):
        # Test with an invalid high_quantile (not numeric)
        with self.assertRaises(TypeError) as context:
            normalize_features_core(self.data_normalization, 0.2, 'invalid')
        self.assertEqual(
            str(context.exception),
            "The high quantile should be a numeric value, currently got "
            "<class 'str'>")

    def test_invalid_low_quantile(self):
        # Test with an invalid low_quantile (not numeric)
        with self.assertRaises(TypeError) as context:
            normalize_features_core(self.data_normalization, 'invalid', 0.8)
        self.assertEqual(
            str(context.exception),
            "The low quantile should be a numeric value, currently got "
            "<class 'str'>")

    def test_high_quantile_out_of_range(self):
        # Test with high_quantile out of range (should be within (0, 1])
        with self.assertRaises(ValueError) as context:
            normalize_features_core(self.data_normalization, 0.2, 1.5)
        self.assertEqual(
            str(context.exception),
            "The high quantile value should be within (0, 1], current value: "
            "1.5")

        with self.assertRaises(ValueError) as context:
            normalize_features_core(self.data_normalization, 0.2, 1.01)
        self.assertEqual(
            str(context.exception),
            "The high quantile value should be within (0, 1], current value: "
            "1.01")

    def test_low_quantile_out_of_range(self):
        # Test with low_quantile out of range (should be within [0, 1))
        with self.assertRaises(ValueError) as context:
            normalize_features_core(self.data_normalization, -0.5, 0.8)
        self.assertEqual(
            str(context.exception),
            "The low quantile value should be within [0, 1), current value: "
            "-0.5")

        with self.assertRaises(ValueError) as context:
            normalize_features_core(self.data_normalization, -0.01, 0.8)
        self.assertEqual(
            str(context.exception),
            "The low quantile value should be within [0, 1), current value: "
            "-0.01")

    def test_low_quantile_greater_than_high_quantile(self):
        # Test with low_quantile greater than high_quantile
        with self.assertRaises(ValueError) as context:
            normalize_features_core(self.data_normalization, 0.8, 0.7)
        self.assertEqual(
            str(context.exception),
            "The low quantile should be smaller than the high quantile, "
            "current values are:\nlow quantile: 0.8\nhigh quantile: 0.7")

    def test_low_quantile_equal_to_high_quantile(self):
        # Test with low_quantile equal to high_quantile
        with self.assertRaises(ValueError) as context:
            normalize_features_core(self.data_normalization, 0.5, 0.5)
        self.assertEqual(
            str(context.exception),
            "The low quantile should be smaller than the high quantile, "
            "current values are:\nlow quantile: 0.5\nhigh quantile: 0.5")

    def test_invalid_interpolation(self):
        # Test with invalid interpolation method
        with self.assertRaises(ValueError) as context:
            normalize_features_core(self.data_normalization, 0.2, 0.8,
                                    interpolation='invalid')
        self.assertEqual(
            str(context.exception),
            "Interpolation must be either 'nearest' or 'linear', passed value "
            "is: invalid"
        )

    def test_correct_normalization_linear(self):
        # Test scaling the features between 0-1 with no clipping
        low_quantile = 0.2
        high_quantile = 0.8
        interpolation = 'linear'

        # Manually calculated qmin and qmax for linear method
        # qmin = [1.8, 2.4, 3, 2.4, 1.8]
        # qmax = [4.2, 3.6, 3, 3.6, 4.2]

        # clipped_data = np.clip(self.data_normalization, qmin, qmax)
        # Clipped data
        # [[1.8, 2.4, 3, 3.6, 4.2],
        #  [4.2, 3.6, 3, 2.4, 1.8]]

        # Example calculation for normalization:
        # For value 1 in the first column (clipped to 1.8):
        # (1.8 - qmin[0]) / (qmax[0] - qmin[0])
        # = (1.8 - 1.8) / (4.2 - 1.8) = 0
        # For value 3 in the third column (clipped to 3):
        # (3 - qmin[1]) / (qmax[1] - qmin[1])
        # = (3 - 3) / (3 - 3) = NaN (handled as 0)

        # Normalize data based on calculated qmin and qmax
        expected_result = np.array([
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0]
        ])
        normalized_data = normalize_features_core(
            self.data_normalization, low_quantile, high_quantile, interpolation
        )
        np.testing.assert_almost_equal(normalized_data, expected_result,
                                       decimal=6)

    def test_correct_normalization_nearest(self):
        # Test scaling the features between 0-1 with 'nearest' interpolationn
        low_quantile = 0.2
        high_quantile = 0.8
        interpolation = 'nearest'

        # Manually calculated qmin and qmax for nearest method
        # qmin = [1, 2, 3, 2, 1]
        # qmax = [5, 4, 3, 4, 5]

        # clipped_data = np.clip(self.data_normalization, qmin, qmax)
        # Clipped data
        # [[1, 2, 3, 4, 5],
        #  [5, 4, 3, 2, 1]]

        # Example calculation for normalization:
        # For value 1 in the first column (clipped to 1):
        # (1 - qmin[0]) / (qmax[0] - qmin[0])
        # = (1 - 1) / (5 - 1) = 0
        # For value 2 in the second column (clipped to 2):
        # (2 - qmin[1]) / (qmax[1] - qmin[1])
        # = (2 - 2) / (4 - 2) = 0
        # Normalize data based on calculated qmin and qmax
        expected_result = np.array([
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0]
        ])

        normalized_data = normalize_features_core(
            self.data_normalization, low_quantile, high_quantile, interpolation
        )
        np.testing.assert_almost_equal(normalized_data, expected_result,
                                       decimal=6)

    def test_random_data_return_value(self):
        # Test the function does not raise any exceptions with valid parameters
        low_quantile = 0.02
        high_quantile = 0.98
        interpolation = 'linear'

        # Normalize the dataset using the function
        normalized_data = normalize_features_core(
            self.random_data, low_quantile, high_quantile, interpolation)

        # Check that normalized_data is not None
        self.assertIsNotNone(normalized_data,
                             "Normalized data should not be None")

        # Check that normalized_data has the same shape as the original data
        self.assertEqual(self.random_data.shape, normalized_data.shape,
                         "Normalized data should have the same shape as the "
                         "original data")

        # Check for NaN values in the normalized data
        self.assertFalse(np.isnan(normalized_data).any(),
                         "Normalized data contains NaN values")

        # Check that values in normalized_data are within the range [0, 1]
        self.assertTrue(np.all(normalized_data >= 0),
                        "All values in normalized data should be >= 0")
        self.assertTrue(np.all(normalized_data <= 1),
                        "All values in normalized data should be <= 1")

    def test_qmin_equals_qmax(self):
        # Create a dataset where all features have constant values
        data = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ])
        low_quantile = 0.0
        high_quantile = 1.0
        interpolation = 'linear'

        # Since qmin and qmax for each column will be the same value
        # Normalize data for each column should result in zeros
        expected_result = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        normalized_data = normalize_features_core(
            data, low_quantile, high_quantile, interpolation
        )
        np.testing.assert_almost_equal(
            normalized_data, expected_result, decimal=6
        )


if __name__ == '__main__':
    unittest.main()
