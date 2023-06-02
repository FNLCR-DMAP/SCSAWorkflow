import unittest
import pandas as pd
import numpy as np
from spac.data_utils import subtract_min_per_region
from spac.data_utils import ingest_cells, concatinate_regions


class TestAnalysisMethods(unittest.TestCase):

    def test_subtract_min_per_region(self):

        batch = "region"
        # Marker for first region
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [2, 4],
            batch: "reg1"
        })

        df2 = pd.DataFrame({
            'marker1': [3, 6],
            'marker2': [4, 8],
            batch: "reg2"
        })

        adata1 = ingest_cells(df1, "^marker.*", obs=batch)
        adata2 = ingest_cells(df2, "^marker.*", obs=batch)

        all_adata = concatinate_regions([adata1, adata2])

        min_normalized_layer = "min_subtracted"
        subtract_min_per_region(
            all_adata,
            batch,
            min_normalized_layer,
            min_quantile=0)

        ground_truth = np.array([[0, 1, 0, 3], [0, 2, 0, 4]]).transpose()

        min_subtracted_array = all_adata.to_df(
            layer=min_normalized_layer
            ).to_numpy()
        # print(min_subtracted_array)
        # print(ground_truth)
        # print(np.array_equal(ground_truth, min_subtracted_array))
        self.assertEqual(np.array_equal(
            ground_truth, min_subtracted_array
            ), True)


if __name__ == '__main__':
    unittest.main()
