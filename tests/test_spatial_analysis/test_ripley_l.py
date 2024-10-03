import unittest
import pandas as pd
import numpy as np
import anndata
from spac.spatial_analysis import ripley_l  # Replace with actual import


class TestRipleyL(unittest.TestCase):

    def create_dummy_dataset(self, dataframe=None):
        # Create a mock AnnData object for testing
        if dataframe is None:
            cells = [
                {'phenotype': 'A', 'feature': 10, "region": "region1",
                 'spatial_x': 0, 'spatial_y': 0},

                {'phenotype': 'A', 'feature': 10, "region": "region1",
                 'spatial_x': 1, 'spatial_y': 1},

                {'phenotype': 'A', 'feature': 10, "region": "region1",
                 'spatial_x': 0, 'spatial_y': 1},

                {'phenotype': 'A', 'feature': 10, "region": "region2",
                 'spatial_x': 0, 'spatial_y': 0},

                {'phenotype': 'B', 'feature': 10, "region": "region2",
                 'spatial_x': 1, 'spatial_y': 1},

                {'phenotype': 'B', 'feature': 10, "region": "region2",
                 'spatial_x': 0, 'spatial_y': 2},
            ]
            dataframe = pd.DataFrame(cells)

        features = dataframe['feature'] \
            .to_numpy() \
            .reshape(-1, 1) \
            .astype(np.float32)
        spatial_coords = dataframe[['spatial_x', 'spatial_y']].to_numpy()

        adata = anndata.AnnData(X=features)
        adata.obs["phenotype"] = pd.Categorical(dataframe["phenotype"])
        adata.obs["region"] = pd.Categorical(dataframe["region"])
        adata.obsm["spatial"] = spatial_coords
        return adata

    def test_no_regions(self):
        """
        Test I/O when all cells are in the same region.
        """
        adata = self.create_dummy_dataset()
        adata_region = adata[adata.obs.region == 'region1']
        distances = [5]
        phenotypes = ['A', 'A']

        result = ripley_l(
            adata=adata_region,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
        )

        # Check the output is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('ripley_l', result.columns)
        self.assertEqual(result['center_phenotype'].iloc[0], 'A')
        self.assertEqual(result['neighbor_phenotype'].iloc[0], 'A')
        self.assertEqual(result['region'].iloc[0], "all")
        self.assertEqual(result['region_cells'].iloc[0], 3)
        # Check that ripley results are not None
        self.assertIsNotNone(result['ripley_l'].iloc[0])

    def test_ripley_l_two_regions(self):
        adata = self.create_dummy_dataset()
        distances = list(np.linspace(0, 500, 10))
        phenotypes = ['A', 'B']

        result = ripley_l(
            adata=adata,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
            regions='region'
        )

        # List of expected regions
        expected_regions = ['region1', 'region2']

        # Get the unique regions from the DataFrame's 'region' column
        actual_regions = result['region'].unique().tolist()

        # Check if both 'region1' and 'region2' are in the 'region' column
        self.assertTrue(
            all(region in actual_regions for region in expected_regions),
            "Expected regions are not present in the 'region' column.")

    def test_results_saved_in_uns(self):
        adata = self.create_dummy_dataset()
        distances = [5]
        phenotypes = ['A', 'B']

        ripley_l(
            adata=adata,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
            regions='region'
        )

        # Check that the existing results are used or updated
        self.assertIn('ripley_l', adata.uns)

    def test_appending_uns_results(self):
        """
        Call the function twice with different phenoyptes and
        Check that results are appeneded correctly.
        """
        adata = self.create_dummy_dataset()
        distances = [5]
        phenotypes = ['A', 'B']

        ripley_l(
            adata=adata,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
            regions='region'
        )

        phenotypes = ['A', 'A']

        ripley_l(
            adata=adata,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
            regions='region'
        )

        # Check that the existing results are used or updated
        # The ripley_l should have four results (2 regions * 2 calls)
        self.assertEqual(adata.uns['ripley_l'].shape[0], 4)

    def test_no_phenotype(self):
        """
        Test that Ripley does not run if a region has a missing phenotype
        """
        adata = self.create_dummy_dataset()
        adata_region = adata[adata.obs.region == 'region1']
        distances = [5]
        phenotypes = ['A', 'C']

        result = ripley_l(
            adata=adata_region,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
        )

        # Check that ripley results are None
        self.assertIsNone(result.iloc[0]['ripley_l'])
        expected_message = (
            'WARNING, phenotype "C" not found in region "all",'
            ' skipping Ripley L.'
        )
        # Check that the expected message is printed
        self.assertEqual(result.iloc[0]['message'], expected_message)

    def test_two(self):
        """
        Test that Ripley does not run if a region has a missing phenotype
        """
        adata = self.create_dummy_dataset()
        adata_region = adata[adata.obs.region == 'region1']
        adata_region = adata_region[:2, :]
        distances = [5]
        phenotypes = ['A', 'A']

        result = ripley_l(
            adata=adata_region,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
        )

        # Check that ripley results are None
        self.assertIsNone(result.iloc[0]['ripley_l'])
        expected_message = (
            'WARNING, not enough cells in region "all".'
            ' Number of cells "2". Skipping Ripley L.'
        )
        # Check that the expected message is printed
        self.assertEqual(result.iloc[0]['message'], expected_message)


if __name__ == '__main__':
    unittest.main()
