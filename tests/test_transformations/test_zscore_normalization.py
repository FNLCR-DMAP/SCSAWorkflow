import unittest
import numpy as np
import pandas as pd
import anndata
from spac.transformations import z_score_normalization


class TestZScoreNormalization(unittest.TestCase):

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

    def test_z_score_normalization_main_matrix(self):
        z_score_normalization(self.adata, output_layer='z_scores')
        # Expected z-scores calculated manually
        expected_z_scores = np.array([
            [-1.22474487, -1.22474487],
            [0., 0.],
            [1.22474487, 1.22474487]
        ])
        np.testing.assert_array_almost_equal(
            self.adata.layers['z_scores'], expected_z_scores
        )

    def test_z_score_normalization_layer(self):
        z_score_normalization(
            self.adata,
            output_layer='z_scores',
            input_layer="layer1"
        )
        expected_z_scores = np.array([
            [-0.7071, -1.3641],
            [-0.7071, 0.3590],
            [1.4142, 1.0051]
        ])
        np.testing.assert_array_almost_equal(
            self.adata.layers['z_scores'], expected_z_scores, decimal=3
        )


if __name__ == "__main__":
    unittest.main()
