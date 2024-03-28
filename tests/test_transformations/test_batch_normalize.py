import unittest
import numpy as np
import anndata as ad
import pandas as pd
from spac.transformations import batch_normalize
from spac.data_utils import ingest_cells, concatinate_regions


class TestAnalysisMethods(unittest.TestCase):

    def test_batch_normalize_median_log(self):

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

        adata1 = ingest_cells(df1, "^marker.*", annotation=batch)
        adata2 = ingest_cells(df2, "^marker.*", annotation=batch)

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "median_normalization"
        batch_normalize(
            all_adata,
            annotation=batch,
            output_layer=median_normalized_layer,
            method="median",
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

        adata1 = ingest_cells(df1, "^marker.*", annotation=batch)
        adata2 = ingest_cells(df2, "^marker.*", annotation=batch)

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "median_normalization"
        batch_normalize(
            all_adata,
            annotation=batch,
            output_layer=median_normalized_layer,
            method="median",
            log=False)

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

        adata1 = ingest_cells(df1, "^marker.*", annotation=batch)
        adata2 = ingest_cells(df2, "^marker.*", annotation=batch)

        all_adata = concatinate_regions([adata1, adata2])

        median_normalized_layer = "q50_normalization"
        batch_normalize(
            all_adata,
            annotation=batch,
            output_layer=median_normalized_layer,
            method="Q50")

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

        adata1 = ingest_cells(df1, "^marker.*", annotation=batch)
        adata2 = ingest_cells(df2, "^marker.*", annotation=batch)

        all_adata = concatinate_regions([adata1, adata2])

        normalized_layer = "q75_normalization"
        batch_normalize(
            all_adata,
            annotation=batch,
            output_layer=normalized_layer,
            method="Q75")

        ground_truth = np.array(
            [
                [1.857142, 3.714285, 2.6, 3.466666],
                [4.2, 5.6, 4.565217, 5.47826]
            ]
            ).transpose()

        normalized_array = all_adata.to_df(
            layer=normalized_layer
            ).to_numpy()
        # print(normalized_array)
        # print(ground_truth)
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)

    def test_cell_orders_is_preserved_median(self):
        # Marker values and annotations
        df = pd.DataFrame({
            'marker1': [1, 5, 2, 6],
            'batch_annotation': ["batch1", "batch2", "batch1", "batch2"]
        })

        # Assuming ingest_cells and concatinate_regions work as expected
        adata = ingest_cells(df, "marker1", annotation="batch_annotation")

        # Perform batch normalization
        median_normalized_layer = "median_normalization"
        batch_normalize(
            adata,
            annotation="batch_annotation",
            output_layer=median_normalized_layer,
            method="median")

        # Calculate the expected normalized values
        # Normalized values: [1+(3.5-1.5), 5+(3.5-5.5),
        # 2+(3.5-1.5), 6+(3.5-5.5)]
        ground_truth = np.array([3.0, 3.0, 4.0, 4.0])

        # Extract the normalized array
        normalized_df = adata.to_df(layer=median_normalized_layer)
        normalized_array = normalized_df.to_numpy().flatten()
        # print("Ground Truth:", ground_truth)
        # print("Normalized Array:", normalized_array)

        # Check if the normalized values match the ground truth
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)

    def test_cell_orders_is_preserved_q50(self):
        # Marker values and annotations
        df = pd.DataFrame({
            'marker1': [1, 5, 2, 6],
            'batch_annotation': ["batch1", "batch2", "batch1", "batch2"]
        })

        # Assuming ingest_cells and concatenate_regions work as expected
        adata = ingest_cells(df, "marker1", annotation="batch_annotation")

        # Perform batch normalization using Q50
        q50_normalized_layer = "q50_normalization"
        batch_normalize(
            adata,
            annotation="batch_annotation",
            output_layer=q50_normalized_layer,
            method="Q50")

        # Calculate the expected normalized values for Q50
        # Normalized values: [1*(3.5/1.5), 5*(3.5/5.5),
        # 2*(3.5/1.5), 6*(3.5/5.5)]
        ground_truth = np.array([
            2.333333, 3.181818, 4.666667, 3.818182
        ])

        # Extract the normalized array
        normalized_df = adata.to_df(layer=q50_normalized_layer)
        normalized_array = normalized_df.to_numpy().flatten()

        # Check if the normalized values match the ground truth
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)

    def test_cell_orders_is_preserved_q75(self):
        df = pd.DataFrame({
            'marker1': [1, 5, 2, 6],
            'batch_annotation': ["batch1", "batch2", "batch1", "batch2"]
        })

        # Assuming ingest_cells and concatenate_regions work as expected
        adata = ingest_cells(df, "marker1", annotation="batch_annotation")

        # Perform batch normalization using Q75
        q75_normalized_layer = "q75_normalization"
        batch_normalize(
            adata,
            annotation="batch_annotation",
            output_layer=q75_normalized_layer,
            method="Q75")

        # Calculate the expected normalized values for Q75
        # Normalized values: [1*(5.25/1.75), 5*(5.25/5.75),
        # 2*(5.25/1.75), 6*(5.25/5.75)]
        ground_truth = np.array([3.0, 4.565217, 6.0, 5.478261])

        # Extract the normalized array
        normalized_df = adata.to_df(layer=q75_normalized_layer)
        normalized_array = normalized_df.to_numpy().flatten()

        # Check if the normalized values match the ground truth
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)

    def test_batch_normalize_z_score(self):
        batch = "batch_annotation"
        df = pd.DataFrame({
            'marker1': [1, 5, 2, 6, 3, 7],
            'marker2': [2, 6, 3, 7, 4, 8],
            batch: ["batch1", "batch1", "batch2", "batch2", "batch3", "batch3"]
        })

        adata = ingest_cells(df, "^marker.*", annotation=batch)

        z_score_normalized_layer = "z_score_normalization"
        batch_normalize(
            adata,
            annotation=batch,
            output_layer=z_score_normalized_layer,
            method="z-score")

        # Calculate the z-score normalization manually:
        ground_truth = np.array([
            [-0.707107, -0.707107],
            [0.707107, 0.707107],
            [-0.707107, -0.707107],
            [0.707107, 0.707107],
            [-0.707107, -0.707107],
            [0.707107, 0.7071071],
        ])

        normalized_array = adata.to_df(
            layer=z_score_normalized_layer
            ).to_numpy()
        # print(normalized_array)
        # print(ground_truth)
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)

    def test_batch_normalize_log_type_check(self):
        data = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ])

        obs = pd.DataFrame({'batch': ['batch1', 'batch2', 'batch1']})
        adata = ad.AnnData(X=data, obs=obs)

        # Function call that should raise ValueError for non-boolean 'log'
        with self.assertRaises(ValueError):
            batch_normalize(
                adata=adata,
                annotation="batch",
                output_layer="test_layer",
                log="not_boolean",  # Intentionally incorrect type
                method="median")

    def test_batch_normalize_with_input_layer(self):

        batch = "region"
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

        adata1 = ingest_cells(df1, "^marker.*", annotation=batch)
        adata2 = ingest_cells(df2, "^marker.*", annotation=batch)

        all_adata = concatinate_regions([adata1, adata2])
        all_adata.layers["preprocessed"] = np.log2(1 + all_adata.X)

        normalized_layer = "normalized_from_preprocessed"
        batch_normalize(
            all_adata,
            annotation=batch,
            output_layer=normalized_layer,
            input_layer="preprocessed",
            method="median")

        ground_truth = np.array(
            [[2.5, 3.5, 4.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 5.5, 6.5, 7.5]]
            ).transpose()

        normalized_array = all_adata.to_df(
            layer=normalized_layer
            ).to_numpy()
        self.assertEqual(np.allclose(ground_truth, normalized_array), True)


if __name__ == '__main__':
    unittest.main()
