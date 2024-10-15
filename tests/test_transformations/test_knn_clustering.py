import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.datasets import load_iris
from spac.transformations import knn_clustering


class TestKnnClustering(unittest.TestCase):
    def setUp(self):
        """
        This method is run before each test.

        It sets up a test AnnData object `syn_data` with the following attributes:

        `syn_data.X`
            A layer with 1000 rows and 2 columns generated from a normal distribution [500 rows ~ N(10, 1), 500 rows ~ N(100, 1)]

        `syn_data.obs['classes']`
            Class labels for rows: label -1 = missing, label 0 = mean 10, label 1 = mean 100
            Approx 50% of the rows are missing label
        """

        n_rows = 1000

        # Generate 1000 rows, half with mean (10, 10) and half with mean (100, 100)
        mean_10 = np.random.normal(loc = 10, scale = 1, size=(n_rows // 2, 2))
        mean_100 = np.random.normal(loc = 100, scale = 1, size=(n_rows // 2, 2))
        data_rows = np.vstack((mean_10, mean_100))

        # Generate class labels, label 0 = mean 10, label 1 = mean 100
        class_labels = np.array([0] * (n_rows // 2) + [1] * (n_rows // 2))

        # Replace ~50% of class labels with "missing" values
        mask = np.random.rand(*class_labels.shape) < 0.5
        class_labels[mask] = "no_label"

        # Combine data columns with class labels
        self.syn_dataset = data_rows

        self.syn_data = AnnData(
                self.syn_dataset,
                var=pd.DataFrame(index=['gene1',
                                        'gene2'])
                )
        self.syn_data.layers['counts'] = self.syn_dataset

        self.syn_data.obsm["derived_features"] = \
            self.syn_dataset

        self.syn_data.obs['classes'] = class_labels

        # Using sklearn iris dataset
        n_iris = 100
        iris_df = load_iris(as_frame=True)

        # Set up AnnData object with 100 rows, all 30 features
        # and each row's class in obs
        self.adata = AnnData(iris_df.data.iloc[:n_iris, :],
                             var=pd.DataFrame(index=iris_df.data.columns))
        self.adata.obs["classes"]=iris_df.target.iloc[:n_iris].to_numpy()

        self.adata.layers['counts'] = iris_df.data.iloc[:n_iris, :]

        self.features = pd.DataFrame(index=iris_df.data.columns)
        self.layer = 'counts'

        self.adata.obsm["derived_features"] = iris_df.data.iloc[:n_iris, :]

    def test_typical_case(self):
            # This test checks if the function correctly adds 'knn' to the
            # AnnData object's obs attribute and if it correctly sets
            # 'knn_features' in the AnnData object's uns attribute.
            knn_clustering(self.adata, self.features, self.layer)
            self.assertIn('knn', self.adata.obs)
            self.assertEqual(self.adata.uns['knn_features'],
                            self.features)
        
    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_layer" 
        # to the # AnnData object's obs attribute 
        output_annotation_name = 'my_output_annotation'
        knn_clustering(self.adata,
                              self.features,
                              self.layer,
                              output_annotation=output_annotation_name)
        self.assertIn(output_annotation_name, self.adata.obs)

    def test_layer_none_case(self):
        # This test checks if the function works correctly when layer is None.
        knn_clustering(self.adata, self.features, None)
        self.assertIn('knn', self.adata.obs)
        self.assertEqual(self.adata.uns['knn_features'],
                         self.features)
        
    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            knn_clustering(self.adata, self.features, self.layer,
                                  'invalid')

    def test_clustering_accuracy(self):
        knn_clustering(self.adata,
                              self.features,
                              'counts',
                              k=50,
                              resolution_parameter=0.1)

        self.assertIn('knn', self.adata.obs)
        self.assertEqual(
            len(np.unique(self.adata.obs['knn'])),
            2)
        
    def test_associated_features(self):
        # Run knn using the derived feature and generate two clusters
        output_annotation = 'derived_knn'
        associated_table = 'derived_features'
        knn_clustering(
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
