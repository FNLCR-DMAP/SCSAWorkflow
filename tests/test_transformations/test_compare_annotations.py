import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import compare_annotations
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from plotly.graph_objs import Figure
 
class TestCompareAnnotations(unittest.TestCase):
 
    def setUp(self):
        self.obs = pd.DataFrame({
                'cluster1': [0, 1, 0, 1],
                'cluster2': [1, 0, 1, 0],
                'cluster3': [0, 0, 1, 2]
            })
        self.adata = AnnData(X=np.random.rand(4, 5))
        self.adata.obs = self.obs.copy()
 
    def test_annotation_length(self):
        # check if there are at least two annotations
        with self.assertRaises(ValueError):
            compare_annotations(self.adata, annotation_list = ['cluster1'])
 
    def test_annotation_exists(self):
        # check if the annotations exist in the given list
        with self.assertRaises(ValueError):
            compare_annotations(self.adata, annotation_list = ['cluster1', 'cluster4'])
 
    def test_adjusted_rand_score(self):
        # check the output of adjusted rand score
        expected = adjusted_rand_score(self.adata.obs['cluster1'], self.adata.obs['cluster2'])
        compare_annotations(self.adata, annotation_list = ['cluster1', 'cluster2'])
 
        # check adjusted rand score of comparing cluster1 and cluster2
        self.assertEqual(self.adata.uns["compare_annotations"][0,1], expected)
        self.assertEqual(self.adata.uns["compare_annotations"][1,0], expected)
 
        # check adjusted rand score on matrix diagonal where cluster is compared to itself
        self.assertEqual(self.adata.uns["compare_annotations"][0,0], 1.0)
        self.assertEqual(self.adata.uns["compare_annotations"][1,1], 1.0)
    
    def test_normalized_mutual_info_score(self):
        # check the output of normalized mutual info score
        expected = normalized_mutual_info_score(self.adata.obs['cluster1'], self.adata.obs['cluster3'])
        compare_annotations(self.adata, annotation_list=['cluster1', 'cluster3'], metric="normalized_mutual_info_score")
 
        # check normalized mutual info score of comparing cluster1 and cluster3
        self.assertEqual(self.adata.uns["compare_annotations"][0,1], expected)
        self.assertEqual(self.adata.uns["compare_annotations"][1,0], expected)
 
        # check normalized mutual info score on matrix diagonal where cluster is compared to itself
        self.assertEqual(self.adata.uns["compare_annotations"][0,0], 1.0)
        self.assertEqual(self.adata.uns["compare_annotations"][1,1], 1.0)
    
    def test_metric_error(self):
        # check if the given metric doesn't exist
        with self.assertRaises(ValueError):
            compare_annotations(self.adata, annotation_list = ['cluster1', 'cluster2'], metric = 'invalid')
    
    def test_heatmap_creation(self):
        # check if a proper figure is created
        fig = compare_annotations(self.adata, annotation_list = ['cluster1', 'cluster2', 'cluster3'])
 
        # Check function returns a Figure object
        self.assertIsInstance(fig, Figure)
 
        # Check that the trace is a heatmap
        self.assertEqual(fig.data[0].type, "heatmap")
 
        # Check the axis labels
        x_labels = list(fig.data[0].x)
        y_labels = list(fig.data[0].y)
        self.assertEqual(x_labels, self.adata.uns["compare_annotations_list"])
        self.assertEqual(y_labels, self.adata.uns["compare_annotations_list"])
    
 
 
if __name__ == "__main__":
    unittest.main()