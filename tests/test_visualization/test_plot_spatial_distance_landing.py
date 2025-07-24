import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
import unittest
import anndata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from spac.spatial_analysis import calculate_spatial_distance
from spac.visualization import plot_spatial_distance_landing


class TestPlotSpatialDistanceLanding(unittest.TestCase):
    def setUp(self):
        # Create a minimal AnnData object with several cells and two images
        data = np.array([[1.0], [2.0], [3.0], [4.0]])
        obs = pd.DataFrame(
            {
                'cell_type': ['type1', 'type2', 'type1', 'type2'],
                'imageid': ['img1', 'img1', 'img2', 'img2']
            },
            index=['CellA', 'CellB', 'CellC', 'CellD']
        )
        spatial_coords = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 11.0]
        ])
        self.adata = anndata.AnnData(X=data, obs=obs)
        self.adata.obsm['spatial'] = spatial_coords

        # Compute spatial distances
        # We'll use 'cell_type' as annotation and 'imageid' as imageid column
        from spac.spatial_analysis import calculate_spatial_distance
        calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            imageid='imageid',
            label='spatial_distance',
            verbose=False
        )

    def test_single_plot_mode(self):
        """
        Test plot_spatial_distance_landing with split_by_imageid=False.
        This should create a single plot, potentially facetted, but here
        we do not facet, just test if it runs without error.
        """
        df, fig = plot_spatial_distance_landing(
            adata=self.adata,
            annotation='cell_type',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            method='numeric',
            plot_type='box',
            return_df=True,
            split_by_imageid=False
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('group', df.columns)
        self.assertIn('distance', df.columns)

    def test_split_by_imageid_mode(self):
        """
        Test plot_spatial_distance_landing with split_by_imageid=True.
        Each image should be plotted separately. We have two images: img1 and img2.
        The returned df should contain data from both images.
        """
        df, fig = plot_spatial_distance_landing(
            adata=self.adata,
            annotation='cell_type',
            imageid='imageid',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            method='numeric',
            plot_type='box',
            return_df=True,
            split_by_imageid=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('imageid', df.columns)
        # We should have rows for both img1 and img2
        self.assertTrue(all(df['imageid'].isin(['img1', 'img2'])))

    def test_no_distance_from(self):
        """
        Test error message when 'distance_from' is not provided.
        """
        expected_msg = (
            "Please specify the 'distance_from' phenotype. This indicates "
            "the reference group from which distances are measured."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            plot_spatial_distance_landing(
                adata=self.adata,
                annotation='cell_type',
                spatial_distance='spatial_distance',
                distance_from=None,
                method='numeric'
            )

    def test_invalid_method(self):
        """
        Test error message when 'method' is invalid in landing function.
        """
        expected_msg = (
            "Invalid 'method'. Please choose 'numeric' "
            "or 'distribution'."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            plot_spatial_distance_landing(
                adata=self.adata,
                annotation='cell_type',
                spatial_distance='spatial_distance',
                distance_from='type1',
                method='invalid'
            )

    def test_faceting_in_single_plot_mode(self):
        """
        Test that faceting works when split_by_imageid=False and facet_by='imageid'.
        We should get a single figure that may be internally facetted.
        """
        df, fig = plot_spatial_distance_landing(
            adata=self.adata,
            annotation='cell_type',
            imageid='imageid',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            method='numeric',
            plot_type='box',
            return_df=True,
            split_by_imageid=False,
            col='imageid'
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(fig, plt.Figure)
        # Data should contain entries for both images
        self.assertTrue(all(df['imageid'].isin(['img1', 'img2'])))


if __name__ == '__main__':
    unittest.main()
