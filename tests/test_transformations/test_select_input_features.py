import unittest
import anndata
import numpy as np
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
                [[23, 24]]])}
        )

    def test__select_input_features_no_input(self):
        # Test when no layer or input_derived_feature is specified
        selected_array = _select_input_features(self.adata)
        self.assertTrue(np.array_equal(selected_array, self.adata.X))

    def test__select_input_features_with_layer(self):
        # Test when layer is specified
        selected_array = _select_input_features(self.adata, layer='layer1')
        self.assertTrue(
            np.array_equal(selected_array, self.adata.layers['layer1']))

    def test__select_input_features_with_input_derived_feature(self):
        # Test when input_derived_feature is specified
        selected_array = _select_input_features(
            self.adata,
            input_derived_feature='input_derived_feature')
        expected_array = self.adata.obsm['input_derived_feature'].reshape(3, 2)
            
        self.assertTrue(np.array_equal(selected_array, expected_array))

    def test__select_input_features_invalid_args(self):
        # Test when both layer and input_derived_feature are specified
        with self.assertRaises(ValueError) as context:
            _select_input_features(
                self.adata, layer='layer1',
                input_derived_feature='input_derived_feature')

        expected_message = ("Cannot specify both 'associated table':"
                            "'input_derived_feature'"
                            " and 'table':'layer1'. Please choose one.")
        self.assertEqual(expected_message, str(context.exception))

        with self.assertRaises(ValueError):
            _select_input_features(
                self.adata,
                input_derived_feature='unknown_key')


if __name__ == '__main__':
    unittest.main()
