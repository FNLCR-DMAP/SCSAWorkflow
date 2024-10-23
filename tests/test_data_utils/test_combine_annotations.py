import unittest
import pandas as pd
from anndata import AnnData
from spac.data_utils import combine_annotations  
import numpy as np


class TestCombineAnnotations(unittest.TestCase):


    def setUp(self):
        # Set up test data with annotations in adata.obs
        obs_data = pd.DataFrame({
            'A': ['a1', 'a2', 'a3'],
            'B': ['b1', 'b2', 'b3'],
            'C': ['c1', 'c2', 'c3']
        })
      
        # Set up dummy real values for adata.X
        X_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        # Create AnnData object with .obs and .X
        self.adata = AnnData(X=X_data, obs=obs_data)

    def test_valid_combination(self):
        # Test valid combination of annotations
        result = combine_annotations(
            self.adata.copy(),
            ['A', 'B'],
            '-',
            'AB_combined'
        )
        self.assertTrue('AB_combined' in result.obs.columns)
        self.assertEqual(result.obs['AB_combined'][0], 'a1-b1')
        self.assertEqual(result.obs['AB_combined'][1], 'a2-b2')

    def test_invalid_annotations_type(self):
        # Test if annotations are provided as a string instead of a list
        with self.assertRaises(ValueError) as context:
            combine_annotations(self.adata.copy(), 'A', '-', 'AB_combined')
        self.assertEqual(
            str(context.exception), 
            "Annotations must be a list. Got <class 'str'>")

    def test_existing_new_annotation_name(self):
        # Test when the new annotation name already exists in the adata.obs
        self.adata.obs['AB_combined'] = ['a1-b1', 'a2-b2', 'a3-b3']
        with self.assertRaises(ValueError) as context:
            combine_annotations(
                self.adata.copy(), ['A', 'B'], '-', 'AB_combined')
        self.assertEqual(
            str(context.exception),
            "'AB_combined' already exists in adata.obs.")

    def test_empty_annotations_list(self):
        # Test if an empty list of annotations raises an error
        with self.assertRaises(ValueError) as context:
            combine_annotations(self.adata.copy(), [], '-', 'combined_empty')
    
        # Verify that the correct error message is returned
        self.assertEqual(str(context.exception),
                         "Annotations list cannot be empty.")


if __name__ == '__main__':
    unittest.main()
