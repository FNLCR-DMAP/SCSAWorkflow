import unittest
import numpy as np
import pandas as pd
import anndata
from spac.transformations import arcsinh_transformation
import warnings


class TestArcsinhTransformation(unittest.TestCase):

    def setUp(self):
        data_df = pd.DataFrame({
            'Feature1': [2, 4, 6, 8, 10, 12],
            'Feature2': [5, 10, 15, 20, 25, 30]
        })
        var_df = pd.DataFrame(index=data_df.columns)
        self.adata = anndata.AnnData(
            X=data_df.values,
            var=var_df,
            dtype=np.float32
        )

        # Modify data for layer1 to make it different from main matrix
        layer1_data = data_df.values + np.array(
            [[1, -1], [-1, 2], [0, 0], [1, -1], [-1, 2], [0, 0]]
        ).astype(np.float32)
        self.adata.layers["layer1"] = layer1_data

        # Batch annotations for testing per-batch normalization
        self.adata.obs['batch'] = [1, 1, 1, 2, 2, 2]

    def test_arcsinh_transformation_main_matrix(self):
        data = self.adata.X
        percentile = 20
        co_factor = np.percentile(data, percentile, axis=0)
        expected_data = np.arcsinh(data / co_factor)
        # print("Computed Co-factors (main matrix):", co_factor)
        # print("Expected Data (main matrix):", expected_data)

        transformed_adata = arcsinh_transformation(
            self.adata, percentile=percentile
        )
        # print("Actual (main matrix):", transformed_adata.layers['arcsinh'])
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'],
            expected_data,
            decimal=5
        )

    def test_arcsinh_transformation_main_matrix_hardcoded(self):
        transformed_adata = arcsinh_transformation(
            self.adata, percentile=20
        )
        # Hardcoded expected transformation using 20th percentile
        # Expected results with detailed calculations:
        # Percentile calculation:
        # Feature1: 20th percentile of [2, 4, 6, 8, 10, 12] is 3.2
        # Feature2: 20th percentile of [5, 10, 15, 20, 25, 30] is 8
        # Transformation:
        # Feature1: arcsinh([2, 4, 6, 8, 10, 12] / 3.2)
        #           = [0.4812, 0.8814, 1.1948, 1.4436, 1.6472, 1.8184]
        # Feature2: arcsinh([5, 10, 15, 20, 25, 30] / 8)
        #           = [0.4812, 0.8814, 1.1948, 1.4436, 1.6472, 1.8184]
        expected_data = np.array([
            [0.4812, 0.4812],
            [0.8814, 0.8814],
            [1.1948, 1.1948],
            [1.4436, 1.4436],
            [1.6472, 1.6472],
            [1.8184, 1.8184]
        ])
        # print("Actual:", transformed_adata.layers['arcsinh'])
        # print("Expected:", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=4
        )

    def test_arcsinh_transformation_layer(self):
        transformed_adata = arcsinh_transformation(
            self.adata, input_layer="layer1", percentile=20
        )
        # Expected transformation using 20th percentile on layer1
        # Expected results with detailed calculations:
        # Percentile calculation:
        # Feature1: 20th percentile of [3, 3, 6, 9, 9, 12] is 3.6
        # Feature2: 20th percentile of [4, 12, 15, 19, 27, 30] is 7.4
        # Transformation:
        # Feature1: arcsinh([3, 3, 6, 9, 9, 12] / 3.6)
        #           = [0.8814, 0.8814, 1.4436, 1.8184, 1.8184, 2.0947]
        # Feature2: arcsinh([4, 12, 15, 19, 27, 30] / 7.4)
        #           = [0.3275, 0.8814, 1.0476, 1.2401, 1.5502, 1.6472]
        expected_data = np.array([
            [0.8814, 0.3275],
            [0.8814, 0.8814],
            [1.4436, 1.0476],
            [1.8184, 1.2401],
            [1.8184, 1.5502],
            [2.0947, 1.6472]
        ])
        # print("Actual:", transformed_adata.layers['arcsinh'])
        # print("Expected:", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=4
        )

    def test_arcsinh_transformation_fixed_co_factor(self):
        transformed_adata = arcsinh_transformation(self.adata, co_factor=5)
        # Hardcoded expected transformation using fixed co-factor of 5
        # Co-factor: 5
        # Transformation:
        # Feature1: arcsinh([2, 4, 6, 8, 10, 12] / 5)
        #           = [0.3900, 0.7327, 1.0160, 1.2490, 1.4436, 1.6094]
        # Feature2: arcsinh([5, 10, 15, 20, 25, 30] / 5)
        #           = [0.8814, 1.4436, 1.8184, 2.0947, 2.3124, 2.4918]
        expected_data = np.array([
            [0.3900, 0.8814],
            [0.7327, 1.4436],
            [1.0160, 1.8184],
            [1.2490, 2.0947],
            [1.4436, 2.3124],
            [1.6094, 2.4918]
        ])
        # print(
        #    "Actual (fixed co_factor=5):",
        #    transformed_adata.layers['arcsinh']
        # )
        # print("Expected (fixed co_factor=5):", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=4
        )

    def test_arcsinh_transformation_custom_output_layer(self):
        output_name = "custom_layer"
        transformed_adata = arcsinh_transformation(
            self.adata, output_layer=output_name, percentile=20
        )
        # Expected transformation using 20th percentile
        # Percentile calculation:
        # Feature1: 20th percentile of [2, 4, 6, 8, 10, 12] is 3.2
        # Feature2: 20th percentile of [5, 10, 15, 20, 25, 30] is 8
        # Transformation:
        # Feature1: arcsinh([2, 4, 6, 8, 10, 12] / 3.2)
        #           = [0.4812, 0.8814, 1.1948, 1.4436, 1.6472, 1.8184]
        # Feature2: arcsinh([5, 10, 15, 20, 25, 30] / 8)
        #           = [0.4812, 0.8814, 1.1948, 1.4436, 1.6472, 1.8184]
        expected_data = np.array([
            [0.4812, 0.4812],
            [0.8814, 0.8814],
            [1.1948, 1.1948],
            [1.4436, 1.4436],
            [1.6472, 1.6472],
            [1.8184, 1.8184]
        ])
        # print("Actual:", transformed_adata.layers[output_name])
        # print("Expected:", expected_data)
        self.assertIn(output_name, transformed_adata.layers)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers[output_name], expected_data, decimal=4
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
        # First transformation to create the output layer
        arcsinh_transformation(self.adata, percentile=20)

        expected_message = (
            "Layer 'arcsinh' already exists. "
            "It will be overwritten with the new transformed data."
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Trigger all warnings

            # Cause a warning by calling the function with an existing
            # output_layer
            arcsinh_transformation(self.adata, percentile=20)

            # Check if the warning message is as expected
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertEqual(expected_message, str(w[-1].message))

    def test_invalid_both_cofactor_and_percentile(self):
        with self.assertRaises(ValueError) as context:
            arcsinh_transformation(self.adata, co_factor=5, percentile=20)
        self.assertEqual(
            str(context.exception), "Please specify either co_factor or "
            "percentile, not both."
        )

    def test_invalid_neither_cofactor_nor_percentile(self):
        with self.assertRaises(ValueError) as context:
            arcsinh_transformation(self.adata)
        self.assertEqual(
            str(context.exception), "Either co_factor or percentile "
            "must be provided."
        )

    def test_arcsinh_transformation_per_batch(self):
        transformed_adata = arcsinh_transformation(
            self.adata, per_batch=True, annotation='batch', percentile=20
        )
        # Expected transformation using 20th percentile per batch
        # Batch 1:
        # Feature1: 20th percentile of [2, 4, 6] is 2.8
        # Feature2: 20th percentile of [5, 10, 15] is 7
        # Transformation:
        # Feature1: arcsinh([2, 4, 6] / 2.8)
        #           = arcsinh([0.714, 1.429, 2.143])
        #           = [0.66433, 1.15448, 1.50576]
        # Feature2: arcsinh([5, 10, 15] / 7)
        #           = arcsinh([0.714, 1.429, 2.143])
        #           = [0.66433, 1.15448, 1.50576]
        # Batch 2:
        # Feature1: 20th percentile of [8, 10, 12] is 8.8
        # Feature2: 20th percentile of [20, 25, 30] is 22
        # Transformation:
        # Feature1: arcsinh([8, 10, 12] / 8.8)
        #           = arcsinh([0.909, 1.136, 1.364])
        #           = [0.81561, 0.97459, 1.11666]
        # Feature2: arcsinh([20, 25, 30] / 22)
        #           = arcsinh([0.909, 1.136, 1.364])
        #           = [0.81561, 0.97459, 1.11666]
        expected_data = np.array([
            [0.66433, 0.66433],
            [1.15448, 1.15448],
            [1.50576, 1.50576],
            [0.81561, 0.81561],
            [0.97459, 0.97459],
            [1.11666, 1.11666]
        ])
        # print("Actual (per batch):", transformed_adata.layers['arcsinh'])
        # print("Expected (per batch):", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
