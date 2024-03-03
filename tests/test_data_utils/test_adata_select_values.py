import unittest
import anndata as ad
import numpy as np
from spac.data_utils import adata_select_values


class TestAdataSelectValues(unittest.TestCase):
    def setUp(self):
        """Set up for testing adata_select_values."""
        # Create an AnnData object with specific annotations
        self.adata = ad.AnnData(
            np.random.rand(10, 2),  # 10 cells, 2 genes
            obs={
                'cell_type': [
                    'T', 'T', 'B', 'B', 'NK', 'T', 'T', 'B', 'B', 'NK'
                ],
                'condition': [
                    'healthy', 'infected', 'healthy', 'infected', 'healthy',
                    'healthy', 'infected', 'infected', 'healthy', 'infected'
                ]
            }
        )
        # Add a layer with dummy data for testing layer-specific selection
        self.adata.layers["dummy_layer"] = np.random.rand(10, 2)

    def test_select_values_by_annotation(self):
        """
        Test selecting cells based on a single annotation.
        """
        result = adata_select_values(self.adata, 'cell_type', ['T', 'B'])
        # Expecting 8 cells where cell_type is either 'T' or 'B'
        self.assertEqual(result.n_obs, 8)

    def test_all_cells_no_values_given(self):
        """
        Test that all cells are returned when no specific values are given.
        """
        result = adata_select_values(self.adata, 'cell_type')
        # Expecting all cells to be returned
        self.assertEqual(result.n_obs, 10)

    def test_no_matching_values(self):
        """
        Test selecting cells with no matching values.
        """
        result = adata_select_values(self.adata, 'cell_type', ['Nonexistent'])
        # Expecting no cells to match
        self.assertEqual(result.n_obs, 0)

    def test_specific_layer(self):
        """
        Test selecting cells while specifying a layer.
        """
        result = adata_select_values(
            self.adata, 'condition', ['healthy'], layer="dummy_layer"
        )
        # Expecting 5 cells to match 'healthy' condition
        self.assertEqual(result.n_obs, 5)

    def test_invalid_layer(self):
        """
        Test handling of invalid layer specification.
        """
        with self.assertRaises(ValueError):
            adata_select_values(
                self.adata, 'cell_type', ['T'], layer="nonexistent_layer"
            )

    def test_invalid_annotation(self):
        """
        Test handling of invalid annotation.
        """
        with self.assertRaises(ValueError):
            adata_select_values(self.adata, 'nonexistent_annotation', ['T'])


if __name__ == '__main__':
    unittest.main()
