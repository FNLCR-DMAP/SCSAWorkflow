import unittest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from spac.transformations import add_qc_metrics
from spac.transformations import get_qc_summary_table

class TestGetQCSummaryTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set a random seed for reproducibility
        np.random.seed(42)

    # Create a small AnnData object for testing
    def create_test_adata(self):
        X = np.array([
            [1, 0, 3, 0],
            [0, 2, 0, 4],
            [5, 0, 0, 6]
        ])
        var_names = ["MT-CO1", "MT-CO2", "GeneA", "GeneB"]
        obs_names = ["cell1", "cell2", "cell3"]
        adata = AnnData(X=X)
        adata.var_names = var_names
        adata.obs_names = obs_names
        # Compute QC metrics using the provided function
        add_qc_metrics(adata)
        return adata

    # Test that the summary table is created and has the correct structure
    def test_qc_summary_table_basic(self):
        adata = self.create_test_adata()
        get_qc_summary_table(adata)
        summary = adata.uns["qc_summary_table"]
        self.assertIn("qc_summary_table", adata.uns)
        self.assertTrue(isinstance(summary, pd.DataFrame))
        # Check that all expected columns are present
        self.assertIn("mean", summary.columns)
        self.assertIn("median", summary.columns)
        self.assertIn("upper_mad", summary.columns)
        self.assertIn("lower_mad", summary.columns)
        self.assertIn("upper_quantile", summary.columns)
        self.assertIn("lower_quantile", summary.columns)
        self.assertIn("Sample", summary.columns)
        # Check that the correct metrics are summarized
        self.assertEqual(set(summary["metric_name"]), 
                         {"nFeature", "nCount", "percent.mt"})
        # Check that the sample label is correct when not grouping
        self.assertEqual(summary["Sample"].iloc[0], "All")

    # Test that a TypeError is raised if a non-numeric column is included
    def test_qc_summary_table_non_numeric(self):
        adata = self.create_test_adata()
        adata.obs["non_numeric"] = ["a", "b", "c"]
        with self.assertRaises(TypeError) as exc_info:
            get_qc_summary_table(adata, 
                                 stat_columns_list=["nFeature", "non_numeric"])
        expected_msg = 'Column "non_numeric" must be numeric to compute statistics.'
        self.assertEqual(str(exc_info.exception), expected_msg)

    # Test that summary statistics are computed correctly for nFeature and nCount
    def test_qc_summary_table_statistics(self):
        adata = self.create_test_adata()
        get_qc_summary_table(adata)
        summary = adata.uns["qc_summary_table"]
        # Check mean, median, quantiles for nFeature (all values are 2)
        nfeature_row = summary[summary["metric_name"] == "nFeature"].iloc[0]
        self.assertEqual(nfeature_row["mean"], 2)
        self.assertEqual(nfeature_row["median"], 2)
        self.assertEqual(nfeature_row["upper_mad"], 2)
        self.assertEqual(nfeature_row["lower_mad"], 2)
        self.assertEqual(nfeature_row["upper_quantile"], 2)
        self.assertEqual(nfeature_row["lower_quantile"], 2)
        # Check nCount statistics
        # nCount per cell = [4, 6, 11] -> 
        # mean 7.0, median 6.0, 95th pct 10.5, 5th pct 4.2
        ncount_row = summary[summary["metric_name"] == "nCount"].iloc[0]
        self.assertAlmostEqual(ncount_row["mean"], 7.0) 
        self.assertAlmostEqual(ncount_row["median"], 6.0)  
        self.assertAlmostEqual(ncount_row["upper_quantile"], 10.5) 
        self.assertAlmostEqual(ncount_row["lower_quantile"], 4.2)  

    # Test that summary statistics is computed correctly with sample_column grouping
    def test_qc_summary_table_grouping(self):
        adata = self.create_test_adata()
        get_qc_summary_table(adata)
        # Add a sample column with two groups
        adata.obs["batch"] = ["A", "A", "B"]
        get_qc_summary_table(adata, sample_column="batch")
        summary = adata.uns["qc_summary_table"]
        # There should be two groups: A and B
        self.assertEqual(set(summary["Sample"]), {"A", "B"})
        # For group A (cells 0 and 1): nCount = [4, 6]
        group_a = summary[(summary["Sample"] == "A") & (summary["metric_name"] == "nCount")].iloc[0]
        self.assertAlmostEqual(group_a["mean"], 5.0)
        self.assertAlmostEqual(group_a["median"], 5.0)
        # For group B (cell 2): nCount = [11]
        group_b = summary[(summary["Sample"] == "B") & 
                          (summary["metric_name"] == "nCount")].iloc[0]
        self.assertAlmostEqual(group_b["mean"], 11.0)
        self.assertAlmostEqual(group_b["median"], 11.0)

    # Test that all-NaN columns are handled gracefully
    def test_qc_summary_table_all_nan_column(self):
        adata = self.create_test_adata()
        adata.obs["all_nan"] = [np.nan, np.nan, np.nan]
        with self.assertRaises(TypeError) as exc_info:
            get_qc_summary_table(adata, stat_columns_list=["all_nan"])
        expected_msg = (
            'Column "all_nan" must be numeric to compute statistics. '
            'All values are NaN.'
        )
        self.assertEqual(str(exc_info.exception), expected_msg)

if __name__ == "__main__":
    unittest.main()