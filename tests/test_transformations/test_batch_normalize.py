import unittest
import numpy as np
import pandas as pd
from spac.transformations import batch_normalize
from spac.data_utils import ingest_cells, concatinate_regions


class TestAnalysisMethods(unittest.TestCase):

    def test_nbatch_normalize_median_log(self):

        batch = "region"
        # Marker for first region
        # Medians of log2 (1+x)  = 2, 5
        df1 = pd.DataFrame({
            'marker1': [1, 3, 7],
            'marker2': [15, 31, 63],
            batch: "reg1"
        })

        # Median of log2 (1+x) = 5, 8
        df2 = pd.DataFrame({
            'marker1': [15, 31, 63],
            'marker2': [127, 255, 511],
            batch: "reg2"
        })

        # Global medians of log2 (1+x) = 3.5, 6.5

        adata1 = ingest_cells(df1, "^marker.*", obs=batch)
        adata2 = ingest_cells(df2, "^marker.*", obs=batch)

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "median_normalization"
        batch_normalize(
            all_adata,
            batch,
            median_normalized_layer,
            "median",
            log=True)

        ground_truth = np.array(
            [[2.5, 3.5, 4.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 5.5, 6.5, 7.5]]
            ).transpose()

        normalized_array = all_adata.to_df(
            layer=median_normalized_layer
            ).to_numpy()
        # print(normalized_array)
        # print(ground_truth)
        self.assertEqual(np.array_equal(ground_truth, normalized_array), True)

    def test_batch_normalize_median(self):

        batch = "region"
        # vMarker for first region
        # vMedians = 2, 5
        df1 = pd.DataFrame({
            'marker1': [1, 2, 3],
            'marker2': [4, 5, 6],
            batch: "reg1"
        })

        # Median = 5, 8
        df2 = pd.DataFrame({
            'marker1': [4, 5, 6],
            'marker2': [7, 8, 9],
            batch: "reg2"
            })

        # Global medians = 3.5, 6.5

        adata1 = ingest_cells(df1, "^marker.*", obs=batch)
        adata2 = ingest_cells(df2, "^marker.*", obs=batch)

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "median_normalization"
        batch_normalize(
            all_adata,
            batch,
            median_normalized_layer,
            "median")

        ground_truth = np.array(
            [[2.5, 3.5, 4.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 5.5, 6.5, 7.5]]
            ).transpose()

        normalized_array = all_adata.to_df(
            layer=median_normalized_layer
            ).to_numpy()
        # print(normalized_array)
        # print(ground_truth)
        self.assertEqual(np.array_equal(ground_truth, normalized_array), True)

    def test_batch_normalize_q50(self):

        batch = "region"
        # Marker for first region
        # Medians = 2, 5
        df1 = pd.DataFrame({
            'marker1': [1, 2, 3],
            'marker2': [4, 5, 6],
            batch: "reg1"
        })

        # Median = 5, 8
        df2 = pd.DataFrame({
            'marker1': [4, 5, 6],
            'marker2': [7, 8, 9],
            batch: "reg2"
        })

        # Global medians = 3.5, 6.5

        adata1 = ingest_cells(df1, "^marker.*", obs=batch)
        adata2 = ingest_cells(df2, "^marker.*", obs=batch)

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "q50_normalization"
        batch_normalize(all_adata, batch, median_normalized_layer, "Q50")

        ground_truth = np.array(
            [
                [1.75, 3.5, 5.25, 2.8, 3.5, 4.2],
                [5.2, 6.5, 7.8, 5.6875, 6.5, 7.3125]
            ]
            ).transpose()

        normalized_array = all_adata.to_df(
            layer=median_normalized_layer
            ).to_numpy()
        # print(normalized_array)
        # print(ground_truth)
        self.assertEqual(np.array_equal(ground_truth, normalized_array), True)

    def test_batch_normalize_q75(self):

        batch = "region"
        # Marker for first region
        # q75 = 1.75, 3.75
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [3, 4],
            batch: "reg1"
        })

        # q75 = 3.75, 5.75
        df2 = pd.DataFrame({
            'marker1': [3, 4],
            'marker2': [5, 6],
            batch: "reg2"
        })

        # Global q75 = 3.25, 5.25

        adata1 = ingest_cells(df1, "^marker.*", obs=batch)
        adata2 = ingest_cells(df2, "^marker.*", obs=batch)

        all_adata = concatinate_regions([adata1, adata2])

        normalized_layer = "q75_normalization"
        batch_normalize(
            all_adata,
            batch,
            normalized_layer,
            "Q75")

        ground_truth = np.array(
            [
                [1.857142, 3.714285, 2.6, 3.466666],
                [4.2, 5.6, 4.565217, 5.47826]
            ]
            ).transpose()

        normalized_array = all_adata.to_df(layer=normalized_layer).to_numpy()
        # print(normalized_array)
        # print(ground_truth)
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)


if __name__ == '__main__':
    unittest.main()
