import unittest
import numpy as np
import pandas as pd
from spac.utils import compute_boxplot_metrics


class TestComputeBoxplotMetrics(unittest.TestCase):

    def setUp(self):
        # For reproducibility
        np.random.seed(42)

        self.data = pd.DataFrame(
            {
                "Marker 1": [1, 3, 5, 7, 9, 11, 13],
                "Marker 2": [2, 4, 6, 8, 10, 12, 98],
            }
        )

        self.annotated_data = pd.DataFrame(
            {
                "Marker 1": [1, 3, 5, 7, 9, 11, 13],
                "Phenotype": ["A", "B", "A", "B", "A", "B", "A"],
            }
        )

        # Generate data with outliers
        normal_data = np.random.normal(0, 1, 10000)
        outliers = np.concatenate(
            [np.random.normal(100, 5, 5000), np.random.normal(-100, 5, 5000)]
        )

        self.flier_data = pd.DataFrame({
            "Marker 1": np.concatenate([normal_data, outliers])
        })

    def test_return_type(self):
        result = compute_boxplot_metrics(
            self.data,
        )

        assert isinstance(result, pd.DataFrame)

    def test_return_value(self):
        result = compute_boxplot_metrics(
            self.data,
            showfliers="all"
        )

        # The expected stats for each marker
        marker1_expected_stats = {
            "marker": "Marker 1",
            "whislo": 1.0,
            "q1": 4.0,
            "med": 7.0,
            "mean": 7.0,
            "q3": 10.0,
            "whishi": 13.0,
            "fliers": [],
        }
        marker2_expected_stats = {
            "marker": "Marker 2",
            "whislo": 2.0,
            "q1": 5.0,
            "med": 8.0,
            "mean": 20.0,
            "q3": 11.0,
            "whishi": 12.0,
            "fliers": [98],
        }

        # Extract the dictionaries for each marker
        marker1_result = result[result["marker"] == "Marker 1"].to_dict(
            orient="row"
        )[0]
        marker2_result = result[result["marker"] == "Marker 2"].to_dict(
            orient="row"
        )[0]

        # Compare the expected and result dictionaries
        self.assertEqual(marker1_expected_stats, marker1_result)
        self.assertEqual(marker2_expected_stats, marker2_result)

    def test_annotation(self):
        result = compute_boxplot_metrics(
            self.annotated_data,
            annotation="Phenotype",
        )

        # The expected stats for each annotation
        # for marker 1
        phenotypeA_expected_stats = {
            "marker": "Marker 1",
            "Phenotype": "A",
            "whislo": 1.0,
            "q1": 4.0,
            "med": 7.0,
            "mean": 7.0,
            "q3": 10.0,
            "whishi": 13.0,
        }
        phenotypeB_expected_stats = {
            "marker": "Marker 1",
            "Phenotype": "B",
            "whislo": 3.0,
            "q1": 5.0,
            "med": 7.0,
            "mean": 7.0,
            "q3": 9.0,
            "whishi": 11.0,
        }

        # Extract the dictionaries for each annotation
        # For marker 1
        phenotypeA_result = result[
            (result["marker"] == "Marker 1") & (result["Phenotype"] == "A")
        ].to_dict(orient="row")[0]
        phenotypeB_result = result[
            (result["marker"] == "Marker 1") & (result["Phenotype"] == "B")
        ].to_dict(orient="row")[0]

        # Compare the expected and result dictionaries
        self.assertEqual(phenotypeA_expected_stats, phenotypeA_result)
        self.assertEqual(phenotypeB_expected_stats, phenotypeB_result)

    def test_downsample_fliers(self):
        result = compute_boxplot_metrics(
            self.flier_data,
            showfliers="downsample",
        )

        # Extract the fliers for Marker 1
        fliers_result = result[result["marker"] == "Marker 1"].to_dict(
            orient="row"
        )[0]['fliers']

        # Get the min and max outliers
        min_outlier = np.min(self.flier_data['Marker 1'])
        max_outlier = np.max(self.flier_data['Marker 1'])

       # Check if the returned result contains the max and min outliers
        self.assertIn(min_outlier, fliers_result)
        self.assertIn(max_outlier, fliers_result)
    
    def test_invalid_options(self):
        # Test invalid options for showfliers
        with self.assertRaises(ValueError):
            compute_boxplot_metrics(
                self.data,
                showfliers="invalid_option",
            )
        
        # Test invalid annotation
        with self.assertRaises(ValueError):
            compute_boxplot_metrics(
                self.annotated_data,
                annotation="invalid_annotation",
            )




if __name__ == "__main__":
    unittest.main()
