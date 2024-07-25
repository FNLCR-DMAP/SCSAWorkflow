import unittest
import numpy as np
import pandas as pd
import anndata
from spac.transformations import normalize_features


class TestNormalizeFeatures(unittest.TestCase):

    def setUp(self):
        data_df = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [5, 10, 15, 20, 25]
        })
        var_df = pd.DataFrame(index=data_df.columns)
        self.adata = anndata.AnnData(
            X=data_df.values,
            var=var_df,
            dtype=np.float32
        )

        layer1_data = data_df.values + np.array(
            [[1, -1], [-1, 2], [0, 0], [1, -1], [-1, 2]]
        ).astype(np.float32)
        self.adata.layers["layer1"] = layer1_data

        self.adata.obs['batch'] = [1, 1, 1, 2, 2]

    def test_normalize_features_main_matrix(self):
        low_quantile = 0.2
        high_quantile = 0.8
        interpolation = 'linear'

        normalized_adata = normalize_features(
            self.adata, low_quantile=low_quantile,
            high_quantile=high_quantile,
            interpolation=interpolation
        )

        # Expected results with detailed calculations:
        # Feature1: qmin = 1.8, qmax = 4.2
        # Values: [1, 2, 3, 4, 5] -> Clipped: [1.8, 2, 3, 4, 4.2]
        # -> Normalized: [0, 0.0833, 0.5, 0.9167, 1]
        # Feature2: qmin = 9, qmax = 21
        # Values: [5, 10, 15, 20, 25] -> Clipped: [9, 10, 15, 20, 21]
        # -> Normalized: [0, 0.0833, 0.5, 0.9167, 1]

        expected_result = np.array([
            [0.0, 0.0],
            [0.0833, 0.0833],
            [0.5, 0.5],
            [0.9167, 0.9167],
            [1.0, 1.0]
        ])

        normalized_data = normalized_adata.layers['normalized_feature']
        # print("Normalized Data (Main Matrix):")
        # print(normalized_data)
        np.testing.assert_almost_equal(
            normalized_data, expected_result, decimal=4
        )

    def test_normalize_features_custom_output_layer(self):
        output_name = "custom_normalized_layer"
        low_quantile = 0.2
        high_quantile = 0.8
        interpolation = 'linear'

        normalized_adata = normalize_features(
            self.adata, low_quantile=low_quantile,
            high_quantile=high_quantile,
            interpolation=interpolation,
            output_layer=output_name
        )

        # Expected results with detailed calculations:
        # Feature1: qmin = 1.8, qmax = 4.2
        # Values: [1, 2, 3, 4, 5] -> Clipped: [1.8, 2, 3, 4, 4.2]
        # -> Normalized: [0, 0.0833, 0.5, 0.9167, 1]
        # Feature2: qmin = 9, qmax = 21
        # Values: [5, 10, 15, 20, 25] -> Clipped: [9, 10, 15, 20, 21]
        # -> Normalized: [0, 0.0833, 0.5, 0.9167, 1]

        expected_result = np.array([
            [0.0, 0.0],
            [0.0833, 0.0833],
            [0.5, 0.5],
            [0.9167, 0.9167],
            [1.0, 1.0]
        ])

        self.assertIn(output_name, normalized_adata.layers)
        normalized_data = normalized_adata.layers[output_name]
        # print("Normalized Data (Custom Output Layer):")
        # print(normalized_data)
        np.testing.assert_almost_equal(
            normalized_data, expected_result, decimal=4
        )

    def test_normalize_features_input_layer(self):
        low_quantile = 0.2
        high_quantile = 0.8
        interpolation = 'linear'

        # Expected results with detailed calculations:
        # Feature1: qmin = 1.8, qmax = 4.2
        # Values: [2, 1, 3, 5, 4] -> Clipped: [2, 1.8, 3, 4.2, 4]
        # -> Normalized: [0.0833, 0, 0.5, 1, 0.9167]
        # Feature2: qmin = 10.2, qmax = 22.8
        # Values: [4, 12, 15, 19, 27] -> Clipped: [10.2, 12, 15, 19, 22.8]
        # -> Normalized: [0, 0.1569, 0.4510, 0.8431, 1]

        expected_result = np.array([
            [0.0833, 0.0],
            [0.0, 0.1569],
            [0.5, 0.4510],
            [1.0, 0.8431],
            [0.9167, 1.0]
        ])

        normalized_adata = normalize_features(
            self.adata, low_quantile=low_quantile,
            high_quantile=high_quantile,
            interpolation=interpolation,
            input_layer='layer1'
        )

        normalized_data = normalized_adata.layers['normalized_feature']
        # print("Normalized Data (Input Layer):")
        # print(normalized_data)
        np.testing.assert_almost_equal(
            normalized_data, expected_result, decimal=4
        )

    def test_normalize_features_invalid_high_quantile(self):
        with self.assertRaises(TypeError) as context:
            normalize_features(
                self.adata, low_quantile=0.2,
                high_quantile='invalid'
            )
        self.assertEqual(
            str(context.exception),
            "The high quantile should be a numeric value, currently got "
            "<class 'str'>"
        )

    def test_normalize_features_invalid_low_quantile(self):
        with self.assertRaises(TypeError) as context:
            normalize_features(
                self.adata, low_quantile='invalid',
                high_quantile=0.8
            )
        self.assertEqual(
            str(context.exception),
            "The low quantile should be a numeric value, currently got "
            "<class 'str'>"
        )

    def test_normalize_features_high_quantile_out_of_range(self):
        with self.assertRaises(ValueError) as context:
            normalize_features(
                self.adata, low_quantile=0.2,
                high_quantile=1.5
            )
        self.assertEqual(
            str(context.exception),
            "The high quantile value should be within (0, 1], current value: "
            "1.5"
        )

    def test_normalize_features_low_quantile_out_of_range(self):
        with self.assertRaises(ValueError) as context:
            normalize_features(
                self.adata, low_quantile=-0.5,
                high_quantile=0.8
            )
        self.assertEqual(
            str(context.exception),
            "The low quantile value should be within [0, 1), current value: "
            "-0.5"
        )

    def test_normalize_features_low_greater_than_high(self):
        with self.assertRaises(ValueError) as context:
            normalize_features(
                self.adata, low_quantile=0.8,
                high_quantile=0.7
            )
        self.assertEqual(
            str(context.exception),
            "The low quantile should be smaller than the high quantile, "
            "current values are: low quantile: 0.8, high_quantile: 0.7"
        )

    def test_normalize_features_invalid_method(self):
        with self.assertRaises(ValueError) as context:
            normalize_features(
                self.adata, low_quantile=0.2,
                high_quantile=0.8,
                interpolation='invalid'
            )
        self.assertEqual(
            str(context.exception),
            "interpolation must be either 'nearest' or 'linear', passed value "
            "is: invalid"
        )

    def test_normalize_features_per_batch(self):
        # Define the quantile and interpolation parameters
        low_quantile = 0.2
        high_quantile = 0.8
        interpolation = 'linear'

        # Perform the normalization per batch
        normalized_adata = normalize_features(
            self.adata, low_quantile=low_quantile,
            high_quantile=high_quantile,
            interpolation=interpolation,
            per_batch=True, annotation='batch'
        )

        # Expected results with detailed calculations:
        # Batch 1:
        # Feature1: qmin = 1.4, qmax = 2.6
        # Values: [1, 2, 3] -> Clipped: [1.4, 2, 2.6]
        # -> Normalized: [0, 0.5, 1]
        # Feature2: qmin = 6, qmax = 12
        # Values: [5, 10, 15] -> Clipped: [7, 10, 13]
        # -> Normalized: [0, 0.5, 1]
        # Batch 2:
        # Feature1: qmin = 4.2, qmax = 4.8
        # Values: [4, 5] -> Clipped: [4.2, 4.8] -> Normalized: [0, 1]
        # Feature2: qmin = 21, qmax = 24
        # Values: [20, 25] -> Clipped: [21, 24] -> Normalized: [0, 1]
        expected_result = np.array([
            [0, 0],
            [0.5, 0.5],
            [1, 1],
            [0, 0],
            [1, 1]
        ])

        # Print the normalized data for verification
        normalized_data = normalized_adata.layers['normalized_feature']
        # print("Normalized Data (Per Batch):")
        # print(normalized_data)

        # Assert that the normalized data matches the expected results
        np.testing.assert_almost_equal(
            normalized_data, expected_result, decimal=4
        )


if __name__ == '__main__':
    unittest.main()
