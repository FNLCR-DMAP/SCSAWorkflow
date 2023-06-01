import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import phenograph_clustering


class TestPhenographClustering(unittest.TestCase):
    def setUp(self):
        # This method is run before each test.
        # It sets up a test case with an AnnData object, a list of features,
        # and a layer name.
        self.adata = AnnData(np.random.rand(100, 3),
                             var=pd.DataFrame(index=['gene1',
                                                     'gene2',
                                                     'gene3']))
        self.adata.layers['counts'] = np.random.rand(100, 3)
        self.features = ['gene1', 'gene2']
        self.layer = 'counts'

        self.syn_data = AnnData(
            [
                np.concatenate(
                        (
                            np.random.normal(100, 1, 500),
                            np.random.normal(10, 1, 500)
                        )
                    ),
                np.concatenate(
                        (
                            np.random.normal(10, 1, 500),
                            np.random.normal(100, 1, 500)
                        )
                    ),
            ],
            var=pd.DataFrame(index=['gene1',
                                    'gene2'])
                                    )

    @patch('scanpy.external.tl.phenograph',
           return_value=(np.random.randint(0, 3, 100), {}))
    def test_typical_case(self, mock_phenograph):
        # This test checks if the function correctly adds 'phenograph' to the
        # AnnData object's obs attribute and if it correctly sets
        # 'phenograph_features' in the AnnData object's uns attribute.
        phenograph_clustering(self.adata, self.features, self.layer)
        self.assertIn('phenograph', self.adata.obs)
        self.assertEqual(self.adata.uns['phenograph_features'],
                         self.features)

    def test_invalid_adata(self):
        # This test checks if the function raises a TypeError when the
        # adata argument is not an AnnData object.
        with self.assertRaises(TypeError):
            phenograph_clustering('invalid', self.features, self.layer)

    def test_invalid_features(self):
        # This test checks if the function raises a TypeError when the
        # features argument is not a list of strings.
        with self.assertRaises(TypeError):
            phenograph_clustering(self.adata, 'invalid', self.layer)

    def test_invalid_layer(self):
        # This test checks if the function raises a ValueError when the
        # layer argument is not found in the AnnData object's layers.
        with self.assertRaises(ValueError):
            phenograph_clustering(self.adata, self.features, 'invalid')

    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            phenograph_clustering(self.adata, self.features, self.layer,
                                  'invalid')

    def test_features_not_in_var_names(self):
        # This test checks if the function raises a ValueError when one or
        # more of the features are not found in the AnnData object's var_names.
        with self.assertRaises(ValueError):
            phenograph_clustering(self.adata, ['invalid'], self.layer)

    @patch('scanpy.external.tl.phenograph',
           return_value=(np.random.randint(0, 3, 100), {}))
    def test_wide_range_inputs(self, mock_phenograph):
        # This test checks the function's behavior for a wide range of possible
        # inputs. It repeatedly calls the function with different random inputs
        # and checks if the expected outputs are correct.
        for _ in range(100):
            adata = AnnData(np.random.rand(100, 3),
                            var=pd.DataFrame(index=['gene1',
                                                    'gene2',
                                                    'gene3']))
            adata.layers['counts'] = np.random.rand(100, 3)
            features = np.random.choice(['gene1', 'gene2', 'gene3'],
                                        size=np.random.randint(1, 3),
                                        replace=False).tolist()
            layer = 'counts'
            phenograph_clustering(adata, features, layer)
            self.assertIn('phenograph', adata.obs)
            self.assertEqual(sorted(adata.uns['phenograph_features']),
                             sorted(features))


if __name__ == '__main__':
    unittest.main()
