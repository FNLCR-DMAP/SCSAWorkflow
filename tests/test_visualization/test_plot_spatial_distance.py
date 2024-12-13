import unittest
import anndata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from spac.spatial_analysis import calculate_spatial_distance
from spac.visualization import plot_spatial_distance


class TestPlotSpatialDistance(unittest.TestCase):
    def setUp(self):
        # Create a minimal AnnData object with two cells of different
        # phenotypes
        data = np.array([[1.0], [2.0]])
        obs = pd.DataFrame(
            {
                'cell_type': ['type1', 'type2'],
                'imageid': ['img1', 'img1']
            },
            index=['CellA', 'CellB']
        )
        spatial_coords = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ])
        self.adata = anndata.AnnData(X=data, obs=obs)
        self.adata.obsm['spatial'] = spatial_coords

        # Compute spatial_distance. Store under 'spatial_distance' key
        calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            imageid='imageid',
            label='spatial_distance',
            verbose=False
        )

        # Print the keys in adata.uns to verify where spatial distances
        # are stored
        print("Keys in adata.uns after calculate_spatial_distance:",
              self.adata.uns.keys())

    def test_plot_spatial_distance_numeric(self):
        """
        Test plot_spatial_distance in numeric mode with return_df=True.
        Using 'box' plot which aggregates data. We therefore expect a single
        aggregated entry for the given group ('type2').
        """
        df, fig = plot_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            method='numeric',
            plot_type='box',
            return_df=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn('group', df.columns)
        self.assertIn('distance', df.columns)

        # Check that the group column indeed has entries with 'type2'
        self.assertIn('type2', df['group'].values)

        # Since this is aggregated data for a single group and single image,
        # we expect one aggregated row
        self.assertEqual(len(df), 1)

    def test_plot_spatial_distance_distribution(self):
        """
        Test plot_spatial_distance in distribution mode without returning df.
        Just check that a figure is produced and no error is raised.
        """
        df, fig = plot_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            method='distribution',
            plot_type='hist',
            return_df=False
        )

        self.assertIsNone(df)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_spatial_distance_with_imageid(self):
        """
        Test plot_spatial_distance adding imageid stratification.
        With box plot (aggregated), we expect one aggregated row per
        group-image combination. Since there's only one image ('img1') and one
        group ('type2'), len(df) should be 1.
        """
        df, _ = plot_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            imageid='imageid',
            method='numeric',
            plot_type='box',
            return_df=True
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('group', df.columns)
        self.assertIn('distance', df.columns)
        self.assertIn('imageid', df.columns)

        # Expect one aggregated row
        self.assertEqual(len(df), 1)
        self.assertTrue(all(df['imageid'] == 'img1'))

    def test_plot_spatial_distance_no_distance_from(self):
        """
        Test error message when 'distance_from' is not provided.
        """
        expected_msg = (
            "Please specify the 'distance_from' phenotype. This indicates "
            "the reference group from which distances are measured."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            plot_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                spatial_distance='spatial_distance',
                distance_from=None,
                method='numeric'
            )

    def test_invalid_method(self):
        """
        Test error message when 'method' is invalid.
        """
        expected_msg = (
            "Invalid 'method'. Please choose 'numeric' "
            "or 'distribution'."
        )
        with self.assertRaisesRegex(ValueError, expected_msg):
            plot_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                spatial_distance='spatial_distance',
                distance_from='type1',
                method='invalid'
            )

    def test_valid_labels(self):
        """
        Test axes labels and verify them in the figure.
        Since an aggregated plot(box) is used, y_axis is 'group'.
        """
        df, fig = plot_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            spatial_distance='spatial_distance',
            distance_from='type1',
            distance_to='type2',
            method='numeric',
            plot_type='box',
            x_axis='distance',
            y_axis='group',
            return_df=True
        )

        # Check axis labels
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'distance')
        self.assertEqual(ax.get_ylabel(), 'group')


if __name__ == '__main__':
    unittest.main()
