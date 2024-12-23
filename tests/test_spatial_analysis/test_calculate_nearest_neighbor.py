import unittest
import anndata
import numpy as np
from spac.spatial_analysis import calculate_nearest_neighbor
import pandas as pd
import io
from contextlib import redirect_stdout
from pandas.testing import assert_frame_equal


class TestCalculateNearestNeighbor(unittest.TestCase):

    def setUp(self):
        # Create a minimal deterministic AnnData object
        data = np.array([[1.0, 2.0],
                         [3.0, 4.0],
                         [5.0, 6.0],
                         [7.0, 8.0]])
        spatial_coords = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        obs = pd.DataFrame({
            'cell_type': ['type1', 'type1', 'type2', 'type2'],
            'imageid': ['image1', 'image1', 'image2', 'image2']
        }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])

        self.adata = anndata.AnnData(X=data, obs=obs)
        self.adata.obsm['spatial'] = spatial_coords

    def test_output_one_slide_one_phenotype(self):
        """Test output for one slide with a single phenotype."""
        adata1 = anndata.AnnData(
            X=np.array([[1.0]]),
            obs=pd.DataFrame(
                {'cell_type': ['type1'], 'imageid': ['image1']},
                index=['CellA']
            ),
            obsm={'spatial': np.array([[0.0, 0.0]])}
        )
        calculate_nearest_neighbor(
            adata=adata1,
            annotation='cell_type',
            verbose=False
        )
        result1 = adata1.obsm['spatial_distance']
        self.assertIsInstance(result1, pd.DataFrame)
        expected_df_1 = pd.DataFrame(
            data=[[0.0]],
            index=['CellA'],
            columns=['type1']
        )
        assert_frame_equal(result1, expected_df_1)

    def test_output_one_slide_two_phenotypes(self):
        """
        Test output for one slide with two different phenotypes.
        """
        adata2 = anndata.AnnData(
            X=np.array([[1.0], [2.0]]),
            obs=pd.DataFrame(
                {'cell_type': ['type1', 'type2'],
                 'imageid': ['image1', 'image1']},
                index=['CellA', 'CellB']
            ),
            obsm={'spatial': np.array([[0.0, 0.0], [1.0, 1.0]])}
        )
        calculate_nearest_neighbor(
            adata=adata2,
            annotation='cell_type',
            imageid='imageid',
            verbose=False
        )

        result2 = adata2.obsm['spatial_distance']
        self.assertIsInstance(result2, pd.DataFrame)

        # Distances:
        # CellA to type1 = 0.0 (itself), CellA to type2 = sqrt(2)
        # CellB to type1 = sqrt(2), CellB to type2 = 0.0 (itself)
        dist = np.sqrt(2)
        expected_df_2 = pd.DataFrame(
            data=[[0.0, dist],
                  [dist, 0.0]],
            index=['CellA', 'CellB'],
            columns=['type1', 'type2']
        )
        assert_frame_equal(result2, expected_df_2)

    def test_output_two_slides_one_phenotype(self):
        """Test output for two slides, each with one phenotype."""
        adata3 = anndata.AnnData(
            X=np.array([[1.0], [2.0]]),
            obs=pd.DataFrame(
                {'cell_type': ['type1', 'type1'],
                 'imageid': ['image1', 'image2']},
                index=['CellA', 'CellB']
            ),
            obsm={'spatial': np.array([[0.0, 0.0], [1.0, 1.0]])}
        )
        calculate_nearest_neighbor(
            adata=adata3,
            annotation='cell_type',
            imageid='imageid',
            verbose=False
        )

        result3 = adata3.obsm['spatial_distance']
        self.assertIsInstance(result3, pd.DataFrame)

        # Each slide processed separately, each slide has only one cell of
        # type1. Thus, each cell's distance to type1 is 0.0.
        expected_df_3 = pd.DataFrame(
            data=[[0.0],
                  [0.0]],
            index=['CellA', 'CellB'],
            columns=['type1']
        )
        assert_frame_equal(result3, expected_df_3)

    def test_typical_case_with_output(self):
        """Test typical case with default label and a custom label."""
        # Default label
        calculate_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            verbose=False
        )

        # Check if the result is stored under the default label
        self.assertIn('spatial_distance', self.adata.obsm)

        # Check if the output is a DataFrame
        result = self.adata.obsm['spatial_distance']
        self.assertIsInstance(result, pd.DataFrame)

        # Expect columns to have all phenotypes found
        self.assertIn('type1', result.columns)
        self.assertIn('type2', result.columns)
        self.assertEqual(len(result), 4)  # 4 cells total

        # Test custom label
        calculate_nearest_neighbor(
            adata=self.adata,
            annotation='cell_type',
            label='custom_label',
            verbose=False
        )

        # Check if the result is stored under the custom label
        self.assertIn('custom_label', self.adata.obsm)
        custom_result = self.adata.obsm['custom_label']
        self.assertIsInstance(custom_result, pd.DataFrame)

    def test_missing_coordinate_values(self):
        """Test ValueError when coordinate values are missing."""
        self.adata.obsm['spatial'][0, 0] = np.nan
        with self.assertRaises(ValueError) as context:
            calculate_nearest_neighbor(
                adata=self.adata,
                annotation='cell_type',
                verbose=False
            )
        expected_msg = (
            "Missing values found in spatial coordinates for cells at "
            "indices: [0]."
        )
        self.assertEqual(str(context.exception), expected_msg)

    def test_invalid_coordinate_dimensions(self):
        """Test ValueError when coordinates have insufficient dimensions."""
        self.adata.obsm['spatial'] = np.random.rand(4, 1)  # Only one dimension
        with self.assertRaises(ValueError) as context:
            calculate_nearest_neighbor(
                adata=self.adata,
                annotation='cell_type',
                verbose=False
            )
        expected_msg = (
            "The input data must include coordinates with at least "
            "two dimensions, such as X and Y positions."
        )
        self.assertEqual(str(context.exception), expected_msg)

    def test_z_coordinate_handling(self):
        """Test that the function correctly handles Z-coordinate data."""
        # Use a minimal dataset with Z-coordinates
        adata = anndata.AnnData(
            X=np.array([[1.0, 2.0], [3.0, 4.0]]),
            obs=pd.DataFrame(
                {'cell_type': ['type1', 'type2']}, index=['C1', 'C2']
            ),
            obsm={
                'spatial': np.array([
                    [0.0, 0.0, 0.5],
                    [1.0, 1.0, 0.5]
                ])
            }
        )

        calculate_nearest_neighbor(
            adata=adata,
            annotation='cell_type',
            verbose=False
        )

        result = adata.obsm['spatial_distance']
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('type1', result.columns)
        self.assertIn('type2', result.columns)
        self.assertEqual(len(result), 2)

    def test_verbose_output(self):
        """Test verbose output for progress messages."""
        f = io.StringIO()
        with redirect_stdout(f):
            calculate_nearest_neighbor(
                adata=self.adata,
                annotation='cell_type',
                verbose=True
            )
        output = f.getvalue()
        self.assertIn(
            'Preparing data for spatial distance calculation...',
            output
        )

    def test_no_imageid_scenario(self):
        """
        Test behavior when `imageid` is None.

        Without an `imageid` column, a dummy image column is created internally
        so that scimap can run. After computation, this dummy column should be
        removed.

        Setup:
        - Two cells, both "type1" at coordinates (0.0, 0.0) and (1.0, 1.0).
        Both are the same phenotype, so the zero-distance is replaced by
        sqrt(2).

        Expected:
        - `adata.obsm['spatial_distance']` is a DataFrame with one column
          ("type1").
        - Distances:
            CellA: sqrt(2)
            CellB: sqrt(2)
        """
        adata_no_imageid = anndata.AnnData(
            X=np.array([[1.0], [2.0]]),
            obs=pd.DataFrame(
                {'cell_type': ['type1', 'type1']}, index=['CellA', 'CellB']
            ),
            obsm={'spatial': np.array([[0.0, 0.0],
                                       [1.0, 1.0]])}
        )

        # Run the calculation with no imageid specified
        calculate_nearest_neighbor(
            adata=adata_no_imageid,
            annotation='cell_type',
            imageid=None,  # No imageid
            verbose=False
        )

        # Check if results are stored
        self.assertIn('spatial_distance', adata_no_imageid.obsm)
        result = adata_no_imageid.obsm['spatial_distance']
        self.assertIsInstance(result, pd.DataFrame)

        dist = np.sqrt(2)
        expected_df = pd.DataFrame(
            data=[[dist],
                  [dist]],
            index=['CellA', 'CellB'],
            columns=['type1']
        )

        # Assert the result matches the expected DataFrame
        assert_frame_equal(
            result, expected_df, check_exact=False, rtol=1e-5, atol=1e-5
        )

        # Check that the dummy column is removed
        self.assertNotIn('_dummy_imageid', adata_no_imageid.obs.columns)


if __name__ == '__main__':
    unittest.main()
