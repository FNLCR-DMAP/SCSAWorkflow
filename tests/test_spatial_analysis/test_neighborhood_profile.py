import unittest
import numpy as np
import numpy.testing as npt
from spac.spatial_analysis import neighborhood_profile
import pandas as pd
import anndata


class TestNeighborhoodProfile(unittest.TestCase):

    def create_dummy_dataset(self, dataframe=None):
        # Create a mock AnnData object for testing
        # The phenotypes "A" and "B" are present in the dataset
        # Two slides are present
        # Each slide has two cells
        if dataframe is None:
            cells = [
                {'phenotype': 'A', 'feature': 10, "slide": "slide1",
                 'spatial_x': 0,  'spatial_y': 0},

                {'phenotype': 'A', 'feature': 10, "slide": "slide1",
                 'spatial_x': 1,  'spatial_y': 0},

                {'phenotype': 'A', 'feature': 10, "slide": "slide2",
                 'spatial_x': 0,  'spatial_y': 0},

                {'phenotype': 'B', 'feature': 10, "slide": "slide2",
                 'spatial_x': 1,  'spatial_y': 0},
            ]

            dataframe = pd.DataFrame(cells)

        # Extract the feature column as numpy array
        features = dataframe['feature'].to_numpy().reshape(-1, 1)

        # Convert features to float32 
        features = features.astype(np.float32)

        # Extract the spatial coordinates as numpy array
        spatial_coords = dataframe[['spatial_x', 'spatial_y']].to_numpy()

        # Create anndata object
        adata = anndata.AnnData(X=features)

        adata.obs["phenotype"] = pd.Categorical(dataframe["phenotype"])
        if "slide" in dataframe.columns:
            adata.obs["slide"] = pd.Categorical(dataframe["slide"])
        adata.obsm["spatial"] = spatial_coords
        return adata
    
    def test_neighborhood_profile_IO(self):
        # Test the input/outputs of the neighborhood_profile function
        adata = self.create_dummy_dataset()
        bins = [0, 1]
        neighborhood_profile(
            adata,
            regions="slide",
            phenotypes="phenotype",
            distances=bins,
        )

        output_name = "neighborhood_profile"
        # Check that adata.obsm contains the neighborhood profile
        self.assertIn(output_name, adata.obsm)
        # Check that adata.uns contains the summary
        self.assertIn(output_name, adata.uns)
        # Bins are stored correctly
        self.assertEqual(adata.uns[output_name]["bins"],
                         bins)
        
        expected_labels = np.array(['A', 'B'])
        # Get the actual values
        actual_labels = adata.uns[output_name]["labels"]
        # Check that the actual values match the expected values
        npt.assert_array_equal(actual_labels, expected_labels)

        # Check that size of the neighborhood profile is of shape (4, 2, 1)
        self.assertEqual(adata.obsm["neighborhood_profile"].shape, (4, 2, 1))


    def test_one_slide_total_cells_normalized(self):

        cells = [
            {'phenotype': 'A', 'feature': 10, "slide": "slide1",
             'spatial_x': 0,  'spatial_y': 0},

            {'phenotype': 'A', 'feature': 10, "slide": "slide1",
             'spatial_x': 0.9,  'spatial_y': 0},

            {'phenotype': 'B', 'feature': 10, "slide": "slide1",
             'spatial_x': 0.5,  'spatial_y': 0},
        ]

        dataframe = pd.DataFrame(cells)

        adata = self.create_dummy_dataset(dataframe)

        neighborhood_profile(
            adata,
            regions="slide",
            phenotypes="phenotype",
            distances=[0, 1],
            normalize="total_cells"
        )

        expected_profile = np.array([
            [0.5],
            [0.5]])

        # Check that that the array of the first cells is expected
        np.testing.assert_array_equal(
            adata.obsm["neighborhood_profile"][0, :, :], expected_profile
        )

    def test_one_implicit_slide_total_cells_normalized(self):
        # In this test, all the cells belong to one slide, hence
        # the "regions" input argument is not passed

        cells = [
            {'phenotype': 'A', 'feature': 10,
             'spatial_x': 0,  'spatial_y': 0},

            {'phenotype': 'A', 'feature': 10,
             'spatial_x': 0.9,  'spatial_y': 0},

            {'phenotype': 'B', 'feature': 10,
             'spatial_x': 0.5,  'spatial_y': 0},
        ]

        dataframe = pd.DataFrame(cells)

        adata = self.create_dummy_dataset(dataframe)

        neighborhood_profile(
            adata,
            phenotypes="phenotype",
            distances=[0, 1],
            normalize="total_cells"
        )

        expected_profile = np.array([
            [0.5],
            [0.5]])

        # Check that that the array of the first cells is expected
        np.testing.assert_array_equal(
            adata.obsm["neighborhood_profile"][0, :, :], expected_profile
        )



    def test_two_slides_scrambled_cells(self):

        # Cells are ordered alternately in slide 1 and slide 2
        cells = [
            {'phenotype': 'A', 'feature': 10, "slide": "slide1",
             'spatial_x': 0,  'spatial_y': 0},

            {'phenotype': 'B', 'feature': 10, "slide": "slide2",
             'spatial_x': 1,  'spatial_y': 0},

            {'phenotype': 'A', 'feature': 10, "slide": "slide2",
             'spatial_x': 0,  'spatial_y': 0},

            {'phenotype': 'C', 'feature': 10, "slide": "slide1",
             'spatial_x': 1,  'spatial_y': 0},
        ]

        dataframe = pd.DataFrame(cells)

        adata = self.create_dummy_dataset(dataframe)

        neighborhood_profile(
            adata,
            regions="slide",
            phenotypes="phenotype",
            distances=[0, 1.1],
        )

        # Cell 1 has one neighbor of type C (slide 1)
        # Cell 2 has one neighbor of type A (slide 2)
        # Cell 3 has one neighbor of type B (slide 2)
        # Cell 4 has one neighbor of type A (slide 1)
        expected_profile = np.array([
            [
                [0],
                [0],
                [1]
            ],
            [
                [1],
                [0],
                [0]

            ],
            [
                [0],
                [1],
                [0]
            ],
            [
                [1],
                [0],
                [0]

            ]])

        # Check that that the array of the first cells is expected
        np.testing.assert_array_equal(
            adata.obsm["neighborhood_profile"], expected_profile
        )

    def test_invalid_distance_type(self):
        adata = self.create_dummy_dataset()

        with self.assertRaises(TypeError) as context:
            neighborhood_profile(
                adata,
                regions="slide",
                phenotypes="phenotype",
                distances=123
            )

        expected_message = ("distances must be a list, tuple, or numpy array."
                            " Got <class 'int'>")
        self.assertEqual(expected_message, str(context.exception))

    def test_invalid_distance_values(self) -> None:
        adata = self.create_dummy_dataset()

        with self.assertRaises(ValueError) as context:
            neighborhood_profile(
                adata,
                regions="slide",
                phenotypes="phenotype",
                distances=[0, -2, 3],
            )
        self.assertEqual(
            str(context.exception),
            "distances must be a list of positive numbers. Got [0, -2, 3]",
        )

    def test_non_monotonic_distances(self):
        adata = self.create_dummy_dataset()

        with self.assertRaises(ValueError) as context:
            neighborhood_profile(
                adata,
                regions="slide",
                phenotypes="phenotype",
                distances=[1, 3, 2]
            )

        expected_message = ("distances must be monotonically increasing."
                            " Got [1, 3, 2]")
        self.assertEqual(expected_message, str(context.exception))


    def test_invalid_normalization(self):
        adata = self.create_dummy_dataset()

        with self.assertRaises(ValueError) as context:
            neighborhood_profile(
                adata,
                regions="slide",
                phenotypes="phenotype",
                distances=[1, 2],
                normalize="invalid"
            )
        expected_message = ('normalize must be "total_cells", "bin_area" or'
                            ' None. Got "invalid"')
        self.assertEqual(expected_message, str(context.exception))



if __name__ == '__main__':
    unittest.main()
