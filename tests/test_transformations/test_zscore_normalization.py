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
        self.adata.layers["layer1"] = data_df.values * 2

    def test_z_score_normalization_main_matrix(self):
        normalized_adata = z_score_normalization(self.adata)
        expected_z_scores = np.array([
            [-1.22474487, -1.22474487],
            [0, 0],
            [1.22474487, 1.22474487]
        ])
        np.testing.assert_array_almost_equal(
            normalized_adata.layers['z_scores'], expected_z_scores
        )

    def test_z_score_normalization_layer(self):
        normalized_adata = z_score_normalization(self.adata, layer="layer1")
        expected_z_scores = np.array([
            [-1.22474487, -1.22474487],
            [0, 0],
            [1.22474487, 1.22474487]
        ])
        np.testing.assert_array_almost_equal(
            normalized_adata.layers['z_scores'], expected_z_scores
        )


if __name__ == "__main__":
    unittest.main()
