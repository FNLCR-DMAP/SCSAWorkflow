import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import apply_per_batch


# A simple transformation function for testing
def dummy_transformation(adata, input_layer=None, output_layer="transformed"):
    data = adata.X if input_layer is None else adata.layers[input_layer]
    transformed_data = np.log1p(data)  # simple log transformation for testing
    adata.layers[output_layer] = transformed_data
    return adata


class TestApplyPerBatch(unittest.TestCase):

    def create_dataset(self, data=None):
        # Create a sample AnnData object with a test dataframe
        if data is None:
            data = {
                'feature1': [i for i in range(0, 11)],
                'feature2': [i for i in range(0, 101, 10)]
            }

        df_data = pd.DataFrame(data)

        adata = AnnData(pd.DataFrame(df_data))
        adata.var_names = df_data.columns

        return adata

    def setUp(self):
        # Setup an AnnData object
        self.adata = self.create_dataset()

        # Batch annotations for testing per-batch normalization
        self.batches = pd.Series(
            [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], index=self.adata.obs_names
        )

    def test_apply_per_batch_main_matrix(self):
        # Test apply_per_batch on the main matrix
        transformed_adata = apply_per_batch(
            self.adata, self.batches, dummy_transformation,
            output_layer="transformed"
        )
        # Expected transformation using log1p
        expected_data = np.log1p(self.adata.X)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['transformed'], expected_data
        )

    def test_apply_per_batch_layer(self):
        # Test apply_per_batch on a specific layer
        self.adata.layers["layer1"] = self.adata.X + 1  # create a new layer
        transformed_adata = apply_per_batch(
            self.adata, self.batches, dummy_transformation,
            output_layer="transformed", input_layer="layer1"
        )
        # Expected transformation using log1p
        expected_data = np.log1p(self.adata.layers["layer1"])
        np.testing.assert_array_almost_equal(
            transformed_adata.layers['transformed'], expected_data
        )

    def test_apply_per_batch_custom_output_layer(self):
        # Test apply_per_batch with a custom output layer
        output_layer = "custom_transformed"
        transformed_adata = apply_per_batch(
            self.adata, self.batches, dummy_transformation,
            output_layer=output_layer
        )
        # Expected transformation using log1p
        expected_data = np.log1p(self.adata.X)
        self.assertIn(output_layer, transformed_adata.layers)
        np.testing.assert_array_almost_equal(
            transformed_adata.layers[output_layer], expected_data
        )

    def test_apply_per_batch_correct_batches(self):
        # Test apply_per_batch to ensure it applies transformation per batch
        def batch_specific_transformation(
                adata, input_layer=None, output_layer="transformed"
        ):
            data = (adata.X if input_layer is None
                    else adata.layers[input_layer])
            mean_per_batch = data.mean(axis=0)
            adata.layers[output_layer] = data - mean_per_batch
            return adata

        transformed_adata = apply_per_batch(
            self.adata, self.batches, batch_specific_transformation,
            output_layer="transformed"
        )

        # Manually calculate the expected result
        expected_data = self.adata.X.copy()
        for batch in self.batches.unique():
            batch_indices = self.batches[self.batches == batch].index
            batch_data = self.adata.X[
                self.adata.obs_names.get_indexer_for(batch_indices), :
            ]
            mean_per_batch = batch_data.mean(axis=0)
            expected_data[
                self.adata.obs_names.get_indexer_for(batch_indices), :
            ] -= mean_per_batch

        np.testing.assert_array_almost_equal(
            transformed_adata.layers['transformed'], expected_data
        )

    def test_apply_per_batch_invalid_batches(self):
        # Test apply_per_batch with invalid batch annotations
        invalid_batches = pd.Series(
            [1, 1, 2, 2, 3, 3], index=self.adata.obs_names[:6]
        )
        with self.assertRaises(ValueError):
            apply_per_batch(
                self.adata, invalid_batches, dummy_transformation,
                output_layer="transformed"
            )


if __name__ == '__main__':
    unittest.main()
