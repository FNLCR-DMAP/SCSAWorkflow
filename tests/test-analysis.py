import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from scworkflow.analysis import subtract_min_per_region, ingest_cells, concatinate_regions, \
    normalize
import unittest
import pandas as pd 
import numpy as np

class TestAnalysisMethods(unittest.TestCase):

    def test_subtract_min_per_region(self):
        #Marker for first region
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [2, 4],
            'region': "reg1" 

        })

        df2 = pd.DataFrame({
            'marker1': [3, 6],
            'marker2': [4, 8],
            'region': "reg2"
        })

        adata1 = ingest_cells(df1, "marker*", region = "region")
        adata2 = ingest_cells(df2, "marker*", region = "region")

        all_adata = concatinate_regions([adata1, adata2])

        min_normalized_layer = "min_subtracted"
        subtract_min_per_region(all_adata, min_normalized_layer, min_quantile=0)

        ground_truth = np.array([[0, 1, 0, 3], [0, 2, 0, 4]]).transpose()


        min_subtracted_array = all_adata.to_df(layer=min_normalized_layer).to_numpy()
        #print(min_subtracted_array)
        #print(ground_truth)
        #print(np.array_equal(ground_truth, min_subtracted_array))
        self.assertEqual(np.array_equal(ground_truth, min_subtracted_array), True)

    def test_normalize_median_log(self):

        #Marker for first region
        #Medians of log2 (1+x)  = 2, 5
        df1 = pd.DataFrame({
            'marker1': [1, 3, 7],
            'marker2': [15, 31, 63],
            'region': "reg1"

        })

        #Median of log2 (1+x) = 5, 8
        df2 = pd.DataFrame({
            'marker1': [15, 31, 63],
            'marker2': [127, 255, 511],
            'region': "reg2"
        })

        #Global medians of log2 (1+x) = 3.5, 6.5

        adata1 = ingest_cells(df1, "marker*", region = "region")
        adata2 = ingest_cells(df2, "marker*", region = "region")

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "median_normalization"
        normalize(all_adata, median_normalized_layer, "median", log=True)

        ground_truth = np.array([[2.5, 3.5, 4.5, 2.5, 3.5, 4.5], \
            [5.5, 6.5, 7.5, 5.5, 6.5, 7.5]]).transpose()

        normalized_array = all_adata.to_df(layer=median_normalized_layer).to_numpy()
        #print(normalized_array)
        #print(ground_truth)
        self.assertEqual(np.array_equal(ground_truth, normalized_array), True)


    def test_normalize_median(self):

        #Marker for first region
        #Medians = 2, 5
        df1 = pd.DataFrame({
            'marker1': [1, 2, 3],
            'marker2': [4, 5, 6],
            'region': "reg1"

        })

        #Median = 5, 8
        df2 = pd.DataFrame({
            'marker1': [4, 5, 6],
            'marker2': [7, 8, 9],
            'region': "reg2"
        })

        #Global medians = 3.5, 6.5

        adata1 = ingest_cells(df1, "marker*", region = "region")
        adata2 = ingest_cells(df2, "marker*", region = "region")

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "median_normalization"
        normalize(all_adata, median_normalized_layer, "median")

        ground_truth = np.array([[2.5, 3.5, 4.5, 2.5, 3.5, 4.5], \
            [5.5, 6.5, 7.5, 5.5, 6.5, 7.5]]).transpose()

        normalized_array = all_adata.to_df(layer=median_normalized_layer).to_numpy()
        #print(normalized_array)
        #print(ground_truth)
        self.assertEqual(np.array_equal(ground_truth, normalized_array), True)

    def test_normalize_q50(self):

        #Marker for first region
        #Medians = 2, 5
        df1 = pd.DataFrame({
            'marker1': [1, 2, 3],
            'marker2': [4, 5, 6],
            'region': "reg1"

        })

        #Median = 5, 8
        df2 = pd.DataFrame({
            'marker1': [4, 5, 6],
            'marker2': [7, 8, 9],
            'region': "reg2"
        })

        #Global medians = 3.5, 6.5

        adata1 = ingest_cells(df1, "marker*", region = "region")
        adata2 = ingest_cells(df2, "marker*", region = "region")

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "q50_normalization"
        normalize(all_adata, median_normalized_layer, "Q50")

        ground_truth = np.array([[1.75, 3.5, 5.25, 2.8, 3.5, 4.2], \
            [5.2, 6.5, 7.8, 5.6875, 6.5, 7.3125]]).transpose()

        normalized_array = all_adata.to_df(layer=median_normalized_layer).to_numpy()
        #print(normalized_array)
        #print(ground_truth)
        self.assertEqual(np.array_equal(ground_truth, normalized_array), True)


    def test_normalize_q75(self):

        #Marker for first region
        #q75 = 1.75, 3.75 
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [3, 4],
            'region': "reg1"

        })

        #q75 = 3.75, 5.75
        df2 = pd.DataFrame({
            'marker1': [3, 4],
            'marker2': [5, 6],
            'region': "reg2"
        })

        #Global q75 = 3.25, 5.25 

        adata1 = ingest_cells(df1, "marker*", region = "region")
        adata2 = ingest_cells(df2, "marker*", region = "region")

        all_adata = concatinate_regions([adata1, adata2])

        normalized_layer = "q75_normalization"
        normalize(all_adata, normalized_layer, "Q75")

        ground_truth = np.array([[1.857142, 3.714285, 2.6, 3.466666], \
            [4.2, 5.6, 4.565217, 5.47826]]).transpose()

        normalized_array = all_adata.to_df(layer=normalized_layer).to_numpy()
        #print(normalized_array)
        #print(ground_truth)
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)


if __name__ == '__main__':
    unittest.main()

