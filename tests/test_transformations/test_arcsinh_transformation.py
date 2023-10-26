import unittest
import numpy as np
import pandas as pd
import anndata
from unittest.mock import patch
from spac.transformations import arcsinh_transformation


class TestArcsinhTransformation(unittest.TestCase):

    def setUp(self):
        data_df = pd.DataFrame({
            'Feature1': [2, 4, 6],
            'Feature2': [5, 10, 15]
        })
        var_df = pd.DataFrame(index=data_df.columns)
        self.adata = anndata.AnnData(X=data_df.values, var=var_df)

        # Modify data for layer1 to make it different from main matrix
        layer1_data = data_df.values + np.array([[1, -1], [-1, 2], [0, 0]])
        self.adata.layers["layer1"] = layer1_data

    def test_arcsinh_transformation_main_matrix(self):
        transformed_adata = arcsinh_transformation(self.adata)
        # Hardcoded expected transformation using 20th percentile as co-factor
        expected_data = np.array([
            [0.6643306, 0.6643306],
            [1.15447739, 1.15447739],
            [1.50575679, 1.50575679]
        ])
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data
        )

    def test_arcsinh_transformation_layer(self):
        transformed_adata = arcsinh_transformation(
            self.adata, input_layer="layer1")
        # Hardcoded expected transformation using 20th percentile
        # as co-factor on layer1
        expected_data = np.array([
            [0.88137359, 0.530343],
            [0.88137359, 1.283796],
            [1.44363548, 1.480294]
        ])
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data
        )

    def test_arcsinh_transformation_fixed_co_factor(self):
        transformed_adata = arcsinh_transformation(self.adata, co_factor=5)
        # Hardcoded expected transformation using fixed co-factor of 5
        expected_data = np.array([
            [0.39003532, 0.88137359],
            [0.73266826, 1.44363548],
            [1.01597313, 1.81844646]
        ])
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data
        )

    def test_arcsinh_transformation_custom_output_layer(self):
        output_name = "custom_layer"
        transformed_adata = arcsinh_transformation(
            self.adata, output_layer=output_name)
        # Hardcoded expected transformation using 20th percentile as co-factor
        expected_data = np.array([
            [0.6643306, 0.6643306],
            [1.15447739, 1.15447739],
            [1.50575679, 1.50575679]
        ])
        self.assertIn(output_name, transformed_adata.layers)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers[output_name], expected_data
        )

    def test_invalid_percentile_handling(self):
        with self.assertRaises(ValueError) as context:
            arcsinh_transformation(self.adata, percentile=-0.1)
        self.assertEqual(
            str(context.exception), "Percentile should be between 0 and 100."
        )

        with self.assertRaises(ValueError) as context:
            arcsinh_transformation(self.adata, percentile=100.1)
        self.assertEqual(
            str(context.exception), "Percentile should be between 0 and 100."
        )

    def test_warning_on_overwriting_layer(self):
        # Since the default output_layer is "arcsinh", the second
        # transformation should overwrite it
        arcsinh_transformation(self.adata)

        with patch("spac.transformations.logging.warning") as mock_warning:
            arcsinh_transformation(self.adata)
            mock_warning.assert_called_once_with(
                "Layer 'arcsinh' already exists. It will be overwritten with "
                "the new transformed data."
            )

    def test_cofactor_takes_precedence_over_percentile(self):
        # Set both co_factor and percentile
        transformed_adata = arcsinh_transformation(
            self.adata, co_factor=5, percentile=20
        )

        # Hardcoded expected transformation using fixed co-factor of 5
        expected_data = np.array([
            [0.39003532, 0.88137359],
            [0.73266826, 1.44363548],
            [1.01597313, 1.81844646]
        ])

        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data
        )


if __name__ == "__main__":
    unittest.main()
