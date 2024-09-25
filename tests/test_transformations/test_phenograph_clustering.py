import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import phenograph_clustering


class TestPhenographClustering(unittest.TestCase):
    def setUp(self):
        # This method is run before each test.
        # It sets up a test case with an AnnData object, a list of features,
        # and a layer name.
        n_cells = 100 
        self.adata = AnnData(np.random.rand(n_cells, 3),
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

        self.syn_data.obsm["derived_features"] = \
            self.syn_dataset

    def test_same_cluster_assignments_with_same_seed(self):
        # Run phenograph_clustering with a specific seed
        # and store the cluster assignments
        phenograph_clustering(self.adata, self.features, self.layer, seed=42)
        first_run_clusters = self.adata.obs['phenograph'].copy()

        # Reset the phenograph annotation and run again with the same seed
        del self.adata.obs['phenograph']
        phenograph_clustering(self.adata, self.features, self.layer, seed=42)

        # Check if the cluster assignments are the same
        self.assertTrue(
            (first_run_clusters == self.adata.obs['phenograph']).all()
        )

    def test_typical_case(self):
        # This test checks if the function correctly adds 'phenograph' to the
        # AnnData object's obs attribute and if it correctly sets
        # 'phenograph_features' in the AnnData object's uns attribute.
        phenograph_clustering(self.adata, self.features, self.layer)
        self.assertIn('phenograph', self.adata.obs)
        self.assertEqual(self.adata.uns['phenograph_features'],
                         self.features)

    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_layer" 
        # to the # AnnData object's obs attribute 
        output_annotation_name = 'my_output_annotation'
        phenograph_clustering(self.adata,
                              self.features,
                              self.layer,
                              output_annotation=output_annotation_name)
        self.assertIn(output_annotation_name, self.adata.obs)

    def test_layer_none_case(self):
        # This test checks if the function works correctly when layer is None.
        phenograph_clustering(self.adata, self.features, None)
        self.assertIn('phenograph', self.adata.obs)
        self.assertEqual(self.adata.uns['phenograph_features'],
                         self.features)

    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            phenograph_clustering(self.adata, self.features, self.layer,
                                  'invalid')

    def test_clustering_accuracy(self):
        phenograph_clustering(self.syn_data,
                              self.features,
                              'counts',
                              k=50,
                              resolution_parameter=0.1)

        self.assertIn('phenograph', self.syn_data.obs)
        self.assertEqual(
            len(np.unique(self.syn_data.obs['phenograph'])),
            2)

    def test_associated_features(self):
        # Run phenograph using the derived feature and generate two clusters
        output_annotation = 'derived_phenograph'
        associated_table = 'derived_features'
        phenograph_clustering(
            adata=self.syn_data,
            features=None,
            layer=None,
            k=50,
            seed=None,
            output_annotation=output_annotation,
            associated_table=associated_table,
            resolution_parameter=0.1
        )

        self.assertEqual(
            len(np.unique(self.syn_data.obs[output_annotation])),
            2)



if __name__ == '__main__':
    unittest.main()
