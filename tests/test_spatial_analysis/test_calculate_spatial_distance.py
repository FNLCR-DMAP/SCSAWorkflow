import unittest
import anndata
import numpy as np
from spac.spatial_analysis import calculate_spatial_distance
import pandas as pd
import io
from contextlib import redirect_stdout


class TestCalculateSpatialDistance(unittest.TestCase):

    def setUp(self):
        # Create a sample AnnData object
        self.adata = anndata.AnnData(X=np.random.rand(100, 10))
        # Add spatial coordinates to adata.obsm
        self.adata.obsm['spatial'] = np.random.rand(100, 2)
        # Add annotations and image IDs to adata.obs
        self.adata.obs['cell_type'] = np.random.choice(
            ['type1', 'type2'], size=100
        )
        self.adata.obs['imageid'] = np.random.choice(
            ['image1', 'image2'], size=100
        )

    def test_typical_case(self):
        """Test the function with default parameters."""
        calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            verbose=False
        )
        self.assertIn('spatial_distance', self.adata.uns)

    def test_output_structure(self):
        """Test that the output is stored as a pandas DataFrame."""
        calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            verbose=False
        )
        result = self.adata.uns['spatial_distance']
        self.assertIsInstance(result, pd.DataFrame)

    def test_subset_processing(self):
        """Test that the function correctly processes a subset of images."""
        # Run the function with subset=['image1']
        calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            subset=['image1'],
            imageid='imageid',
            verbose=False
        )
        result = self.adata.uns['spatial_distance']

        # Get all cell IDs in 'image1' as strings
        image1_cell_indices = self.adata.obs[
            self.adata.obs['imageid'] == 'image1'
        ].index.astype(str)

        # Get the cell indices in the result
        result_cell_indices = result.index.astype(str)

        # Check that all cell IDs in result are from 'image1'
        self.assertTrue(
            set(result_cell_indices).issubset(set(image1_cell_indices))
        )

    def test_invalid_subset_type(self):
        """Test that passing an invalid subset type raises a TypeError."""
        # Pass an integer as subset, which should raise TypeError
        with self.assertRaises(TypeError):
            calculate_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                subset=123,  # Invalid subset type
                imageid='imageid',
                verbose=False
            )

    def test_subset_imageid_not_found(self):
        """Test that passing a subset with non-existent image IDs
        raises a ValueError."""
        # Pass a subset with an image ID not present in the data
        with self.assertRaises(ValueError):
            calculate_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                subset=['nonexistent_image'],
                imageid='imageid',
                verbose=False
            )

    def test_missing_coordinates(self):
        """Test that the function raises a KeyError
        when spatial coordinates are missing."""
        # Remove the spatial coordinates
        del self.adata.obsm['spatial']
        with self.assertRaises(KeyError):
            calculate_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                verbose=False
            )

    def test_missing_coordinate_values(self):
        """Test that the function raises a ValueError
        when coordinate values are missing."""
        # Set first cell's x-coordinate to NaN
        self.adata.obsm['spatial'][0, 0] = np.nan
        with self.assertRaises(ValueError):
            calculate_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                verbose=False
            )

    def test_obsm_key_not_found(self):
        """Test that passing a non-existent obsm_key raises a KeyError."""
        with self.assertRaises(KeyError):
            calculate_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                obsm_key='nonexistent_obsm',
                verbose=False
            )

    def test_custom_obsm_key(self):
        """Test that the function works with a custom obsm_key."""
        # Rename spatial coordinates
        self.adata.obsm['custom_spatial'] = self.adata.obsm.pop('spatial')
        calculate_spatial_distance(
            adata=self.adata,
            obsm_key='custom_spatial',
            annotation='cell_type',
            verbose=False
        )
        self.assertIn('spatial_distance', self.adata.uns)

    def test_z_coordinate_handling(self):
        """Test that the function correctly handles Z-coordinate data."""
        # Add a Z-coordinate
        self.adata.obsm['spatial'] = np.random.rand(100, 3)
        calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            verbose=False
        )
        result = self.adata.uns['spatial_distance']
        self.assertIsInstance(result, pd.DataFrame)

    def test_label_parameter(self):
        """Test that the label parameter correctly stores the result."""
        calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            label='custom_label',
            verbose=False
        )
        self.assertIn('custom_label', self.adata.uns)

    def test_verbose_output(self):
        # Capture the print output
        f = io.StringIO()
        with redirect_stdout(f):
            calculate_spatial_distance(
                adata=self.adata,
                annotation='cell_type',
                verbose=True
            )
        output = f.getvalue()
        self.assertIn(
            'Preparing data for spatial distance calculation...',
            output
        )

    def test_return_type(self):
        """Test that the function returns the original AnnData object."""
        result = calculate_spatial_distance(
            adata=self.adata,
            annotation='cell_type',
            verbose=False
        )
        self.assertIs(result, self.adata)


if __name__ == '__main__':
    unittest.main()
