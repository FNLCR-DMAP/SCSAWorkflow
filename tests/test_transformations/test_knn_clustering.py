import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.datasets import load_breast_cancer
from spac.transformations import kNN_clustering


class TestkNNClustering(unittest.TestCase):
    def setUp(self):
        # This method is run before each test.
        # It sets up a test case with an AnnData object, a list of features,
        # and a layer name.

        # Using sklearn UCI ML Breast Cancer Wisconsin (Diagnostic) dataset
        n_cells = 100
        cancer_df = load_breast_cancer(as_frame=True)

        # Set up AnnData object with 100 rows, all 30 features
        # and each row's class in obsm
        self.adata = AnnData(cancer_df.data.iloc[:n_cells, :],
                             var=pd.DataFrame(index=cancer_df.data.columns))
        self.adata.obsm["classes"]=cancer_df.target.iloc[:n_cells].to_numpy()

        self.adata.layers['counts'] = cancer_df.data.iloc[:n_cells, :]

        self.features = pd.DataFrame(index=cancer_df.data.columns)
        self.layer = 'counts'

        self.adata.obsm["derived_features"] = cancer_df.data.iloc[:n_cells, :]
        
    def test_same_cluster_assignments_with_same_seed(self):
        # Run kNN_clustering with a specific seed
        # and store the cluster assignments
        kNN_clustering(self.adata, self.features, self.layer, seed=42)
        first_run_clusters = self.adata.obs['kNN'].copy()

        # Reset the kNN annotation and run again with the same seed
        del self.adata.obs['kNN']
        kNN_clustering(self.adata, self.features, self.layer, seed=42)

        # Check if the cluster assignments are the same
        self.assertTrue(
            (first_run_clusters == self.adata.obs['kNN']).all()
        )

    def test_typical_case(self):
            # This test checks if the function correctly adds 'kNN' to the
            # AnnData object's obs attribute and if it correctly sets
            # 'kNN_features' in the AnnData object's uns attribute.
            kNN_clustering(self.adata, self.features, self.layer)
            self.assertIn('kNN', self.adata.obs)
            self.assertEqual(self.adata.uns['kNN_features'],
                            self.features)
        
    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_layer" 
        # to the # AnnData object's obs attribute 
        output_annotation_name = 'my_output_annotation'
        kNN_clustering(self.adata,
                              self.features,
                              self.layer,
                              output_annotation=output_annotation_name)
        self.assertIn(output_annotation_name, self.adata.obs)

    def test_layer_none_case(self):
        # This test checks if the function works correctly when layer is None.
        kNN_clustering(self.adata, self.features, None)
        self.assertIn('kNN', self.adata.obs)
        self.assertEqual(self.adata.uns['kNN_features'],
                         self.features)
        
    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            kNN_clustering(self.adata, self.features, self.layer,
                                  'invalid')

    def test_clustering_accuracy(self):
        kNN_clustering(self.adata,
                              self.features,
                              'counts',
                              k=50,
                              resolution_parameter=0.1)

        self.assertIn('kNN', self.adata.obs)
        self.assertEqual(
            len(np.unique(self.adata.obs['kNN'])),
            2)
        
    def test_associated_features(self):
        # Run kNN using the derived feature and generate two clusters
        output_annotation = 'derived_kNN'
        associated_table = 'derived_features'
        kNN_clustering(
            adata=self.adata,
            features=None,
            layer=None,
            k=50,
            seed=None,
            output_annotation=output_annotation,
            associated_table=associated_table,
            resolution_parameter=0.1
        )

        self.assertEqual(
            len(np.unique(self.adata.obs[output_annotation])),
            2)

if __name__ == '__main__':
    unittest.main()
