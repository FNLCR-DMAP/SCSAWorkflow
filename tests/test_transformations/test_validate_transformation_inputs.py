import unittest
import anndata 
from spac.transformations import _validate_transformation_inputs


class TestValidateTransformationInputs(unittest.TestCase):

    def test_input_derived_feature_and_layer(self):
        adata = anndata.AnnData()
        with self.assertRaises(ValueError) as context:
            _validate_transformation_inputs(
                adata,
                layer="layer_name",
                input_derived_feature="input_derived_feature")

        expected_message = ("Cannot specify both 'associated table':"
                            "'input_derived_feature'"
                            " and 'table':'layer_name'. Please choose one.")
        self.assertEqual(expected_message, str(context.exception))



    def test_input_derived_feature_existence(self):
        adata = anndata.AnnData()
        with self.assertRaises(ValueError):
            _validate_transformation_inputs(
                adata,
                input_derived_feature="feature_name")

    def test_layer_existence(self):
        adata = anndata.AnnData()
        with self.assertRaises(ValueError):
            _validate_transformation_inputs(adata, layer="layer_name")

    def test_features_existence(self):
        adata = anndata.AnnData()
        with self.assertRaises(ValueError):
            _validate_transformation_inputs(
                adata,
                features=["feature1", "feature2"])
