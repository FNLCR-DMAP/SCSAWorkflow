import unittest
import anndata
import numpy as np
import pandas as pd
from spac.transformations import _select_input_features


class TestSelectFeatureInput(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object for testing
        self.adata = anndata.AnnData(
            X=np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]),
            layers={'layer1': np.array([
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]])},
            obsm={'input_derived_feature': np.array([
                [[19, 20]],
                [[21, 22]],
                [[23, 24]]])},
            var=pd.DataFrame(index=["a", "b", "c"])
        )

    def test_no_input(self):
        # Test when no layer or input_derived_feature is specified
        selected_array = _select_input_features(self.adata)
        self.assertTrue(np.array_equal(selected_array, self.adata.X))

    def test_with_layer(self):
        # Test when layer is specified
        selected_array = _select_input_features(self.adata, layer='layer1')
        self.assertTrue(
            np.array_equal(selected_array, self.adata.layers['layer1']))

    def test_input_derived_feature(self):
        # Test when input_derived_feature is specified
        selected_array = _select_input_features(
            self.adata,
            input_derived_feature='input_derived_feature')
        expected_array = self.adata.obsm['input_derived_feature'].reshape(3, 2)
        self.assertTrue(np.array_equal(selected_array, expected_array))

    def test_features_names_list(self):
        selected_array = _select_input_features(
            self.adata,
            features=['a', 'c'])
        expected_array = self.adata.to_df()[['a', 'c']].values
        self.assertTrue(np.array_equal(selected_array, expected_array))

    def test_features_names_str(self):
        selected_array = _select_input_features(
            self.adata,
            features='a')
        # Convert the 1D array to 2D
        expected_array = np.reshape(self.adata.X[:, 0], (-1,1))
        self.assertTrue(np.array_equal(selected_array, expected_array))


if __name__ == '__main__':
    unittest.main()
