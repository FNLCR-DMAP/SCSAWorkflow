import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import normalize_features


class TestNormalizeFeatures(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object with a test dataframe
        self.data = {
            'feature1': [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                100
            ],
            'feature2': [
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
                160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
                170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
                180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
                190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                200
            ]
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

        with self.assertRaises(ValueError):
            normalize_features(self.adata, high_quantile=1.01)

    def test_low_quantile_out_of_range(self):
        # Test with low_quantile out of range (should be within [0, 1))
        with self.assertRaises(ValueError):
            normalize_features(self.adata, low_quantile=-0.5)

        with self.assertRaises(ValueError):
            normalize_features(self.adata, low_quantile=-0.01)

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
        low_quantile = 0.02
        high_quantile = 0.98
        target_layer = None
        new_layer_name = "normalized_feature"
        overwrite = True
        self.maxDiff = None
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
                    0.000000, 0.000000, 0.000000, 0.010417, 0.020833,
                    0.031250, 0.041667, 0.052083, 0.062500, 0.072917,
                    0.083333, 0.093750, 0.104167, 0.114583, 0.125000,
                    0.135417, 0.145833, 0.156250, 0.166667, 0.177083,
                    0.187500, 0.197917, 0.208333, 0.218750, 0.229167,
                    0.239583, 0.250000, 0.260417, 0.270833, 0.281250,
                    0.291667, 0.302083, 0.312500, 0.322917, 0.333333,
                    0.343750, 0.354167, 0.364583, 0.375000, 0.385417,
                    0.395833, 0.406250, 0.416667, 0.427083, 0.437500,
                    0.447917, 0.458333, 0.468750, 0.479167, 0.489583,
                    0.500000, 0.510417, 0.520833, 0.531250, 0.541667,
                    0.552083, 0.562500, 0.572917, 0.583333, 0.593750,
                    0.604167, 0.614583, 0.625000, 0.635417, 0.645833,
                    0.656250, 0.666667, 0.677083, 0.687500, 0.697917,
                    0.708333, 0.718750, 0.729167, 0.739583, 0.750000,
                    0.760417, 0.770833, 0.781250, 0.791667, 0.802083,
                    0.812500, 0.822917, 0.833333, 0.843750, 0.854167,
                    0.864583, 0.875000, 0.885417, 0.895833, 0.906250,
                    0.916667, 0.927083, 0.937500, 0.947917, 0.958333,
                    0.968750, 0.979167, 0.989583, 1.000000, 1.000000,
                    1.000000
                ],
            'feature2': [
                    0.000000, 0.000000, 0.000000, 0.010417, 0.020833,
                    0.031250, 0.041667, 0.052083, 0.062500, 0.072917,
                    0.083333, 0.093750, 0.104167, 0.114583, 0.125000,
                    0.135417, 0.145833, 0.156250, 0.166667, 0.177083,
                    0.187500, 0.197917, 0.208333, 0.218750, 0.229167,
                    0.239583, 0.250000, 0.260417, 0.270833, 0.281250,
                    0.291667, 0.302083, 0.312500, 0.322917, 0.333333,
                    0.343750, 0.354167, 0.364583, 0.375000, 0.385417,
                    0.395833, 0.406250, 0.416667, 0.427083, 0.437500,
                    0.447917, 0.458333, 0.468750, 0.479167, 0.489583,
                    0.500000, 0.510417, 0.520833, 0.531250, 0.541667,
                    0.552083, 0.562500, 0.572917, 0.583333, 0.593750,
                    0.604167, 0.614583, 0.625000, 0.635417, 0.645833,
                    0.656250, 0.666667, 0.677083, 0.687500, 0.697917,
                    0.708333, 0.718750, 0.729167, 0.739583, 0.750000,
                    0.760417, 0.770833, 0.781250, 0.791667, 0.802083,
                    0.812500, 0.822917, 0.833333, 0.843750, 0.854167,
                    0.864583, 0.875000, 0.885417, 0.895833, 0.906250,
                    0.916667, 0.927083, 0.937500, 0.947917, 0.958333,
                    0.968750, 0.979167, 0.989583, 1.000000, 1.000000,
                    1.000000
            ]
        }

        # Convert the DataFrame to a dictionary of lists
        normalized_dict = {
            col: [
                round(val, 6)
                for val in normalized_df[col]
            ]
            for col in normalized_df.columns
        }

        # Check if the normalized values match the expected values
        self.assertDictEqual(normalized_dict, expected_result)

        # Check if normalization information is stored in `uns`
        layer_name_uns = new_layer_name + "_info"
        self.assertIn(layer_name_uns, self.adata.uns)
        normalization_info = self.adata.uns[layer_name_uns]

        # Define the expected normalization information
        expected_info = pd.DataFrame(
            {
                'Pre-Norm: count': ['101.0', '101.0'],
                'Pre-Norm: mean': ['50.0', '150.0'],
                'Pre-Norm: std': ['29.3', '29.3'],
                'Pre-Norm: min': ['0.0', '100.0'],
                'Pre-Norm: 25%': ['25.0', '125.0'],
                'Pre-Norm: 50%': ['50.0', '150.0'],
                'Pre-Norm: 75%': ['75.0', '175.0'],
                'Pre-Norm: max': ['100.0', '200.0'],
                'Pre-Norm: quantile_low': ['2.0', '102.0'],
                'Pre-Norm: quantile_high': ['98.0', '198.0'],
                'Post-Norm: count': ['101.0', '101.0'],
                'Post-Norm: mean': ['0.5', '0.5'],
                'Post-Norm: std': ['0.304', '0.304'],
                'Post-Norm: min': ['0.0', '0.0'],
                'Post-Norm: 25%': ['0.24', '0.24'],
                'Post-Norm: 50%': ['0.5', '0.5'],
                'Post-Norm: 75%': ['0.76', '0.76'],
                'Post-Norm: max': ['1.0', '1.0']
            },
            index=['feature1', 'feature2']
        )

        # Convert the DataFrame to a dictionary of dictionaries
        expected_info_dict = expected_info.to_dict()
        normalization_info = normalization_info.apply(
            lambda x: str(np.round(x, 1))
            if np.issubdtype(x.dtype, np.floating)
            else x
        )
        normalization_info_dict = normalization_info.to_dict()

        # Check if the stored normalization
        # information matches the expected values
        self.assertDictEqual(
            normalization_info_dict,
            expected_info_dict
        )


if __name__ == '__main__':
    unittest.main()
