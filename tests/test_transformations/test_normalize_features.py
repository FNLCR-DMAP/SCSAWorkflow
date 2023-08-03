import unittest
import pandas as pd
from anndata import AnnData
from spac.transformations import normalize_features


class TestNormalizeFeatures(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object with a test dataframe
        self.data = {
            'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature2': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
        self.adata = AnnData(pd.DataFrame(self.data))

    def test_invalid_high_quantile(self):
        # Test with an invalid high_quantile (not numeric)
        with self.assertRaises(TypeError):
            normalize_features(self.adata, high_quantile='2')

    def test_invalid_low_quantile(self):
        # Test with an invalid low_quantile (not numeric)
        with self.assertRaises(TypeError):
            normalize_features(self.adata, low_quantile='2')

    def test_high_quantile_out_of_range(self):
        # Test with high_quantile out of range (should be within (0, 1])
        with self.assertRaises(ValueError):
            normalize_features(self.adata, high_quantile=1.5)

    def test_low_quantile_out_of_range(self):
        # Test with low_quantile out of range (should be within [0, 1))
        with self.assertRaises(ValueError):
            normalize_features(self.adata, low_quantile=-0.5)

    def test_low_quantile_greater_than_high_quantile(self):
        # Test with low_quantile greater than high_quantile
        with self.assertRaises(ValueError):
            normalize_features(
                self.adata,
                low_quantile=0.8,
                high_quantile=0.7
            )

    def test_correct_output(self):
        # Test with valid inputs and check the output values
        low_quantile = 0.2
        high_quantile = 0.8
        target_layer = None
        new_layer_name = "normalized_feature"
        overwrite = True

        normalize_features(
            self.adata,
            low_quantile,
            high_quantile,
            target_layer,
            new_layer_name,
            overwrite)

        normalized_df = self.adata.layers[new_layer_name]
        normalized_df = pd.DataFrame(
            normalized_df,
            columns=['feature1', 'feature2']
        )

        # Define the expected normalized values
        expected_result = {
            'feature1': [
                0.0,
                0.08333333385073478,
                0.500000034148494,
                0.9166666723580822,
                1.0
            ],
            'feature2': [
                0.0,
                0.08333330849806302,
                0.5000000993410804,
                0.9166666418313965,
                1.0
            ]
        }

        # Convert the DataFrame to a dictionary of lists
        normalized_dict = {
            col: list(normalized_df[col]) for col in normalized_df.columns
        }

        # Check if the normalized values match the expected values
        self.assertDictEqual(normalized_dict, expected_result)


if __name__ == '__main__':
    unittest.main()
