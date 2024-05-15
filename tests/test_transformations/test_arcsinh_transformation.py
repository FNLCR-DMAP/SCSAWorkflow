import unittest
import numpy as np
import pandas as pd
import anndata
from spac.transformations import arcsinh_transformation
import warnings
import logging


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
        self.batches = pd.Series(
            [1, 1, 1, 2, 2, 2], index=self.adata.obs_names
        )

    def test_arcsinh_transformation_main_matrix(self):
        data = self.adata.X
        percentile = 20
        co_factor = np.percentile(data, percentile, axis=0)
        expected_data = np.arcsinh(data / co_factor)
        print("Computed Co-factors (main matrix):", co_factor)
        print("Expected Data (main matrix):", expected_data)

        transformed_adata = arcsinh_transformation(self.adata)
        print("Actual (main matrix):", transformed_adata.layers['arcsinh'])
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'],
            expected_data,
            decimal=5
        )

    def test_arcsinh_transformation_main_matrix_hardcoded(self):
        transformed_adata = arcsinh_transformation(self.adata)
        # Hardcoded expected transformation using 20th percentile as co-factor
        expected_data = np.array([
            [0.48121183, 0.48121183],
            [0.88137359, 0.88137359],
            [1.19476322, 1.19476322],
            [1.44363548, 1.44363548],
            [1.64723115, 1.64723115],
            [1.81844646, 1.81844646]
        ])
        print("Actual:", transformed_adata.layers['arcsinh'])
        print("Expected:", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=5
        )

    def test_arcsinh_transformation_layer(self):
        transformed_adata = arcsinh_transformation(
            self.adata, input_layer="layer1")
        # Hardcoded expected transformation using 20th percentile
        # as co-factor on layer1
        expected_data = np.array([
            [0.88137359, 0.32745015],
            [0.88137359, 0.88137359],
            [1.44363548, 1.04759301],
            [1.81844646, 1.24011680],
            [1.81844646, 1.55015796],
            [2.09471255, 1.64723115]
        ])
        print("Actual:", transformed_adata.layers['arcsinh'])
        print("Expected:", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=5
        )

    def test_arcsinh_transformation_fixed_co_factor(self):
        transformed_adata = arcsinh_transformation(self.adata, co_factor=5)
        # Hardcoded expected transformation using fixed co-factor of 5
        expected_data = np.array([
            [0.39003533, 0.88137360],
            [0.73266830, 1.44363550],
            [1.0159732, 1.8184465],
            [1.2489834, 2.0947125],
            [1.4436355, 2.3124382],
            [1.609438, 2.4917798]
        ])
        print(
            "Actual (fixed co_factor=5):",
            transformed_adata.layers['arcsinh']
        )
        print("Expected (fixed co_factor=5):", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=5
        )

    def test_arcsinh_transformation_custom_output_layer(self):
        output_name = "custom_layer"
        transformed_adata = arcsinh_transformation(
            self.adata, output_layer=output_name)
        # Hardcoded expected transformation using 20th percentile as co-factor
        expected_data = np.array([
            [0.48121183, 0.48121183],
            [0.88137359, 0.88137359],
            [1.19476322, 1.19476322],
            [1.44363548, 1.44363548],
            [1.64723115, 1.64723115],
            [1.81844646, 1.81844646]
        ])
        print("Actual:", transformed_adata.layers[output_name])
        print("Expected:", expected_data)
        self.assertIn(output_name, transformed_adata.layers)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers[output_name], expected_data, decimal=5
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

        logging.basicConfig(
            level=logging.INFO,
            format='SPAC:%(asctime)s - %(levelname)s - %(message)s')

        expected_message = ("Layer 'arcsinh' already exists."
                            " It will be overwritten with "
                            "the new transformed data.")

        with warnings.catch_warnings(record=True) as w:
            # Cause a warning by calling the function with an
            # existing output_layer
            arcsinh_transformation(self.adata)

            # Check if the warning message is as expected
            self.assertEqual(len(w), 1)
            self.assertEqual(expected_message, str(w[0].message))

    def test_cofactor_takes_precedence_over_percentile(self):
        # Set both co_factor and percentile
        transformed_adata = arcsinh_transformation(
            self.adata, co_factor=5, percentile=20
        )

        # Hardcoded expected transformation using fixed co-factor of 5
        expected_data = np.array([
            [0.39003533, 0.8813736],
            [0.7326683, 1.4436355],
            [1.0159732, 1.8184465],
            [1.2489834, 2.0947125],
            [1.4436355, 2.3124382],
            [1.609438, 2.4917798]
        ])
        print(
            "Actual (cofactor precedence):",
            transformed_adata.layers['arcsinh']
        )
        print("Expected (cofactor precedence):", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=5
        )

    def test_arcsinh_transformation_per_batch(self):
        transformed_adata = arcsinh_transformation(
            self.adata, per_batch=True, batches=self.batches, percentile=20
        )
        # Hardcoded expected transformation using 20th percentile as co-factor
        # per batch
        expected_data = np.array([
            [0.66433060, 0.66433060],
            [1.15447739, 1.15447739],
            [1.50575679, 1.50575679],
            [0.81560890, 0.81560890],
            [0.97458797, 0.97458797],
            [1.11666279, 1.11666279]
        ])
        print("Actual (per batch):", transformed_adata.layers['arcsinh'])
        print("Expected (per batch):", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['arcsinh'], expected_data, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
