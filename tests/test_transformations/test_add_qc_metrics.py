import unittest
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from spac.transformations import add_qc_metrics

class TestAddQCMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    def create_test_adata(self, sparse=False):
        X = np.array([
            [1, 0, 3, 0],
            [0, 2, 0, 4],
            [5, 0, 0, 6]
        ])
        var_names = ["MT-CO1", "MT-CO2", "GeneA", "GeneB"]
        obs_names = ["cell1", "cell2", "cell3"]
        adata = sc.AnnData(X=csr_matrix(X) if sparse else X)
        adata.var_names = var_names
        adata.obs_names = obs_names
        return adata

    def test_qc_metrics_dense(self):
        adata = self.create_test_adata(sparse=False)
        add_qc_metrics(adata, organism="hs")
        self.assertIn("nFeatue", adata.obs)
        self.assertIn("nCount", adata.obs)
        self.assertIn("nCount_mt", adata.obs)
        self.assertIn("percent.mt", adata.obs)
        np.testing.assert_array_equal(adata.obs["nFeatue"].values, [2, 2, 2])
        np.testing.assert_array_equal(adata.obs["nCount"].values, [4, 6, 11])
        np.testing.assert_array_equal(adata.obs["nCount_mt"].values, [1, 2, 5])
        np.testing.assert_allclose(adata.obs["percent.mt"].values, 
                                   [25.0, 33.333333, 45.454545], rtol=1e-4)

    def test_qc_metrics_sparse(self):
        adata = self.create_test_adata(sparse=True)
        add_qc_metrics(adata, organism="hs")
        self.assertIn("nFeatue", adata.obs)
        self.assertIn("nCount", adata.obs)
        self.assertIn("nCount_mt", adata.obs)
        self.assertIn("percent.mt", adata.obs)
        np.testing.assert_array_equal(adata.obs["nFeatue"].values, [2, 2, 2])
        np.testing.assert_array_equal(adata.obs["nCount"].values, [4, 6, 11])
        np.testing.assert_array_equal(adata.obs["nCount_mt"].values, [1, 2, 5])
        np.testing.assert_allclose(adata.obs["percent.mt"].values, 
                                   [25.0, 33.333333, 45.454545], rtol=1e-4)

    def test_custom_mt_pattern(self):
        adata = self.create_test_adata()
        add_qc_metrics(adata, mt_match_pattern="Gene")
        np.testing.assert_array_equal(adata.obs["nCount_mt"].values, [3, 4, 6])

    def test_invalid_layer(self):
        adata = self.create_test_adata()
        with self.assertRaises(ValueError):
            add_qc_metrics(adata, layer="not_a_layer")

if __name__ == "__main__":
    unittest.main()