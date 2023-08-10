import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import normalize_features


class TestNormalizeFeatures(unittest.TestCase):

    def create_dataset(self, data=None):
        # Create a sample AnnData object with a test dataframe
        if data is None: 
            data = {
                'feature1': [i for i in range(0,11)],
                'feature2': [i for i in range(0, 101, 10)]
            }

        df_data  = pd.DataFrame(data) 

        adata = AnnData(pd.DataFrame(df_data))
        adata.var_names = df_data.columns

        return adata


    def setUp(self):
        #setup an anndata object 
        self.adata = self.create_dataset()


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

    def test_low_quantile_equal_to_high_quantile(self):
        # Test with low_quantile equal to high_quantile
        with self.assertRaises(ValueError):
            normalize_features(
                self.adata,
                low_quantile=0.5,
                high_quantile=0.5
            )

    def test_invalid_interpolation(self):
        # Test with invalid interpolation
        with self.assertRaises(ValueError):
            normalize_features(
                self.adata,
                low_quantile=0.5,
                high_quantile=0.8,
                interpolation='invalid'
            )


    def validate_correct_values(self, 
                                adata, 
                                low_quantile, 
                                high_quantile, 
                                expected_result):

        target_layer = None
        new_layer_name = "normalized_feature"
        overwrite = True
        normalize_features(
            adata=adata,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            new_layer_name=new_layer_name,
            overwrite=overwrite)

        normalized_df = adata.layers[new_layer_name]
        normalized_df = pd.DataFrame(
            normalized_df,
            columns=adata.var_names
        )

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


    def test_correct_scale(self):
        # Test scaling the features betwen 0-1 wih no clipping 
        data = {
            'feature1': [i for i in range(0,11)],
            'feature2': [i for i in range(0, 101, 10)]
        }
        adata = self.create_dataset(data)

        low_quantile = 0
        high_quantile = 1 

        # Define the expected normalized values
        expected_result = {
            'feature1': [i/10 for i in range(0, 11)],
            'feature2': [i/10 for i in range(0, 11)]
        }

        self.validate_correct_values(adata, 
                                     low_quantile, 
                                     high_quantile, 
                                     expected_result) 

    def test_correct_normalization(self):
        # Test scaling the features betwen 0-1 wih no clipping 
        data = {
            'feature1': [-1, 0, 2, 4, 6, 6, 8, 10, 50, 100]
        }
        adata = self.create_dataset(data)

        low_quantile = 0.15
        high_quantile = 0.8 

        # Define the expected normalized values
        expected_result = {
            'feature1': [0, 0, 0.2, 0.4, 0.6, 0.6, 0.8, 1, 1, 1]
        }

        self.validate_correct_values(adata, 
                                     low_quantile, 
                                     high_quantile, 
                                     expected_result) 

        
    def test_quantile_return_value(self):

        # Test scaling the features betwen 0-1 wih no clipping 
        data = {
            'feature1': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        adata = self.create_dataset(data)

        low_quantile = 0.2
        high_quantile = 0.8 

        expected_result = {
            'feature1': [0, 0, 0.2, 0.4, 0.6, 0.6, 0.8, 1, 1, 1]
        }

        # Call the function
        quantiles = normalize_features(adata, low_quantile, high_quantile)
        print(quantiles)

        # Check the return value type
        self.assertIsInstance(quantiles, pd.DataFrame, msg="Quantiles not returned as a pandas DataFrame")

        # Check if the quantiles DataFrame has the expected structure
        expected_columns = ['feature1']
        self.assertListEqual(list(quantiles.columns), expected_columns, msg="Quantiles DataFrame columns mismatch")

        # Check if the quantiles DataFrame has the expected indices
        expected_indices = [low_quantile, high_quantile]
        self.assertListEqual(list(quantiles.index), expected_indices, msg="Quantiles DataFrame indices mismatch")


if __name__ == '__main__':
    unittest.main()
