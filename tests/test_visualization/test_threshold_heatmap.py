import anndata
import unittest
import numpy as np
import pandas as pd
from spac.visualization import threshold_heatmap
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestThresholdHeatmap(unittest.TestCase):
    def setUp(self):
        self.annotation_names = ['cell1', 'cell2', 'cell3']
        self.annotation = pd.DataFrame({'phenotype': ['A', 'B', 'A']},
                                       index=self.annotation_names)
        self.X = np.array([[0.1, 0.3], [0.5, 0.7], [1.0, 1.5]])
        self.var_names = ['marker1', 'marker2']
        self.var = pd.DataFrame(index=self.var_names)

        self.adata = anndata.AnnData(
            X=self.X, obs=self.annotation, var=self.var, dtype=self.X.dtype)
        self.adata.obs_names = self.annotation_names
        self.adata.var_names = self.var_names

        self.marker_cutoffs = {
            'marker1': (0.2, 0.8),
            'marker2': (0.4, 1.0),
        }

        self.phenotype = 'phenotype'

    def test_feature_cutoffs_values_are_nan(self):
        # Test low cutoff is NaN
        feature_cutoffs_nan_low = {
            'marker1': (float('nan'), 0.8),
            'marker2': (0.4, 1.0),
        }
        with self.assertRaises(ValueError):
            threshold_heatmap(
                self.adata,
                feature_cutoffs_nan_low,
                self.phenotype
            )

        # Test high cutoff is NaN
        feature_cutoffs_nan_high = {
            'marker1': (0.2, float('nan')),
            'marker2': (0.4, 1.0),
        }
        with self.assertRaises(ValueError):
            threshold_heatmap(
                self.adata,
                feature_cutoffs_nan_high,
                self.phenotype
            )

    def test_threshold_heatmap(self):
        threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype
        )

        self.assertEqual(
            self.adata.uns['feature_cutoffs'], self.marker_cutoffs
        )

        expected_intensity_data = np.array([[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            self.adata.layers["intensity"], expected_intensity_data
        )

    def test_swap_axes_through_kwargs(self):
        threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype,
            swap_axes=True
        )
        # Add assertions to check if axes are swapped.
        self.assertTrue(True)

    def test_threshold_heatmap_with_layer(self):
        # Add a new layer "new_layer" to adata for this specific test
        new_layer_data = np.array([[0.2, 0.4], [0.6, 0.8], [1.1, 1.6]])
        self.adata.layers["new_layer"] = new_layer_data

        # Using the new layer "new_layer"
        threshold_heatmap(
            self.adata,
            self.marker_cutoffs,
            self.phenotype,
            layer="new_layer"
        )

        # Check if the intensities are set correctly based on the new layer
        expected_intensity_data = np.array([[0, 0], [1, 1], [2, 2]])
        np.testing.assert_array_equal(
            self.adata.layers["intensity"], expected_intensity_data
        )


if __name__ == '__main__':
    unittest.main()
