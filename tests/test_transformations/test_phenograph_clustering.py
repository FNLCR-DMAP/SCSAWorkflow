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

        self.syn_dataset = np.array([
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
                ]).reshape(-1, 2)

        self.syn_data = AnnData(
                self.syn_dataset,
                var=pd.DataFrame(index=['gene1',
                                        'gene2'])
                )
        self.syn_data.layers['counts'] = self.syn_dataset

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

    @patch('scanpy.external.tl.phenograph',
           return_value=(np.random.randint(0, 3, 100), {}))
    def test_layer_none_case(self, mock_phenograph):
        # This test checks if the function works correctly when layer is None.
        phenograph_clustering(self.adata, self.features, None)
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

    def test_clustering_accuracy(self):
        phenograph_clustering(self.syn_data,
                              self.features,
                              'counts',
                              500)
        self.assertIn('phenograph', self.syn_data.obs)
        self.assertEqual(
            len(np.unique(self.syn_data.obs['phenograph'])),
            2)


if __name__ == '__main__':
    unittest.main()
