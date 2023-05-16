import unittest
import pandas as pd
from spac.data_utils import ingest_cells

class TestAnalysisMethods(unittest.TestCase):

    def test_ingest_marker_regex(self):

        batch = "region"
        # Marker for first region
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [2, 4],
            batch: "reg1"
        })

        adata = ingest_cells(df1,
                             "marker*",
                             obs=batch)

        self.assertCountEqual(
            list(adata.var_names),
            ["marker1", "marker2"])

    def test_ingest_marker_list(self):

        batch = "region"
        # Marker for first region
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [2, 4],
            batch: "reg1"

        })

        adata = ingest_cells(df1,
                             ["marker1", "marker2"],
                             obs=batch)

        self.assertCountEqual(
            list(adata.var_names),
            ["marker1", "marker2"])

    def test_ingest_marker_name(self):

        batch = "region"
        # Marker for first region
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [2, 4],
            batch: "reg1"
        })

        adata = ingest_cells(df1,
                             "marker1",
                             obs=batch)
        self.assertCountEqual(
            list(adata.var_names),
            ["marker1"])

    def test_ingest_region_list(self):

        # Marker for first region
        df1 = pd.DataFrame({
            'marker1': [1, 2],
            'marker2': [2, 4],
            "region1": "reg1",
            "region2": "reg2"
        })

        adata = ingest_cells(df1,
                             "marker1",
                             obs=["region1", "region2"])
        self.assertCountEqual(
            list(adata.obs_keys()),
            ["region1", "region2"])


if __name__ == '__main__':
    unittest.main()
