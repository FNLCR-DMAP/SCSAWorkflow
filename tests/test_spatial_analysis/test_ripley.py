
import pandas as pd
import numpy as np
import anndata
import unittest
from spac._ripley import ripley


class TestRipleyL(unittest.TestCase):

    def create_dummy_dataset(self, dataframe=None):
        # Create a mock AnnData object for testing

        # Add four points to form a rectangle that is 1 * 2
        if dataframe is None:
            dictionary = [
                {'phenotype': 'A', 'feature': 10, 'spatial_x': 0,  'spatial_y': 0},
                {'phenotype': 'A', 'feature': 20, 'spatial_x': 1,  'spatial_y': 0},
                {'phenotype': 'A', 'feature': 20, 'spatial_x': 0,  'spatial_y': 2},
                {'phenotype': 'A', 'feature': 20, 'spatial_x': 1,  'spatial_y': 2},
            ]

            dataframe = pd.DataFrame(dictionary)

        # Extract the feature column as numpy array
        features = dataframe['feature'].to_numpy().reshape(-1, 1)

        # Convert features to float32
        features = features.astype(np.float32)

        # Extract the spatial coordinates as numpy array
        spatial_coords = dataframe[['spatial_x', 'spatial_y']].to_numpy()

        # Create anndata object
        adata = anndata.AnnData(X=features)

        adata.obs["phenotype"] = pd.Categorical(dataframe["phenotype"])
        adata.obsm["spatial"] = spatial_coords
        return adata

    def setUp(self):
        # Create a mock AnnData object for testing
        self.adata = self.create_dummy_dataset()

    def test_one_phenotype_with_passed_area_and_support(self):

        # Get the number of row in adata
        n_cells = self.adata.shape[0]

        # Set area to n_cells * pi, so that the L statitc
        # at a given radius is: sqrt(average number of neighbor)
        area = np.pi * n_cells

        radius = 1.1

        result = ripley(self.adata,
                        cluster_key="phenotype",
                        mode="L",
                        n_observations=4,
                        n_simulations=0,
                        support=[radius],
                        area=area,
                        copy=True)

        # Assert that one radius is returned
        self.assertEquals(len(result["L_stat"]), 1)

        # Assert that the radius is correct
        self.assertEquals(result["L_stat"].at[0, "bins"], radius)

        # Assert that the L statistic is 1.0
        self.assertAlmostEqual(result["L_stat"].at[0, "stats"], 1.0, places=5)

    def test_one_phenotype_with_multiple_support(self):

        # Get the number of row in adata
        n_cells = self.adata.shape[0]

        # Set area to n_cells * pi, so that the L statitc
        # at a given radius is: sqrt(average number of neighbor)
        area = np.pi * n_cells

        radii = [0, 1.1, 2.1, 3.1]

        ground_truth_l_stats = [0, 1,
                                np.sqrt(2),
                                np.sqrt(3),
                                ]

        result = ripley(self.adata,
                        cluster_key="phenotype",
                        mode="L",
                        n_observations=4,
                        n_simulations=0,
                        support=radii,
                        area=area,
                        copy=True)

        # Get the L statistics
        returned_l_stats = result["L_stat"]["stats"].to_numpy()

        for returned_l_stat, gt_l_stat in zip(returned_l_stats,
                                              ground_truth_l_stats):
            self.assertAlmostEqual(gt_l_stat, returned_l_stat, places=5)

    def test_two_phenotypes_using_same_phenotype(self):
        """
        This test will run the two phenotypes using the same phenotype key.

        This test checks that the function will return the right statistic
        for two phenotypes using the same phenotype key.
        """
        # Get the number of row in adata
        n_cells = self.adata.shape[0]

        # Set area to n_cells * pi, so that the L statitc
        # at a given radius is: sqrt(average number of neighbor)
        area = np.pi * n_cells

        radii = [0, 1.1, 2.1, 3.1]

        ground_truth_l_stats = [0, 1,
                                np.sqrt(2),
                                np.sqrt(3),
                                ]

        result = ripley(self.adata,
                        cluster_key="phenotype",
                        mode="L",
                        n_observations=4,
                        n_simulations=0,
                        support=radii,
                        area=area,
                        phenotypes=("A", "A"),
                        copy=True)

        # Get the L statistics
        returned_l_stats = result["L_stat"]["stats"].to_numpy()

        # Check that the returned statistic for the two phenotypes is "A_A"
        unique_phenotypes = result["L_stat"]["phenotype"].unique()
        self.assertEqual(len(unique_phenotypes), 1,
                         "The function should return one unique "
                         "phenotype for the two phenotypes.")
        self.assertEqual(unique_phenotypes[0], "A_A",
                         "The returned phenotype should be 'A_A'")

        for returned_l_stat, gt_l_stat in zip(returned_l_stats,
                                              ground_truth_l_stats):
            self.assertAlmostEqual(gt_l_stat, returned_l_stat, places=5,
                                   msg="The ground truth L statistics "
                                       "should match the returned statistics")

    def test_two_different_phenotypes(self):
        """
        This test will run the two different phenotypes
        """

        # Add four points to form a rectangle that is 1 * 2
        dataframe = [
                    {'phenotype': 'A', 'feature': 10, 'spatial_x': 0,  'spatial_y': 0},
                    {'phenotype': 'B', 'feature': 20, 'spatial_x': 1,  'spatial_y': 0},
                    {'phenotype': 'B', 'feature': 20, 'spatial_x': 0,  'spatial_y': 2},
                    {'phenotype': 'A', 'feature': 20, 'spatial_x': 1,  'spatial_y': 2},
                ]

        dataframe = pd.DataFrame(dataframe)
        self.adata = self.create_dummy_dataset(dataframe)

        center_phenotype = "A"
        neighbor_phenotype = "B"
        # Set area to n_center_cells * pi, so that the L statitc
        # at a given radius is: sqrt(average number of neighbor)
        n_center_cells = self.adata[
            self.adata.obs["phenotype"] == center_phenotype].shape[0]
        area = np.pi * n_center_cells

        radii = [0, 1.1, 2.1]

        ground_truth_l_stats = [0,
                                1,
                                np.sqrt(2),
                                ]

        phenotypes = (center_phenotype, neighbor_phenotype)
        result = ripley(self.adata,
                        cluster_key="phenotype",
                        mode="L",
                        n_observations=4,
                        n_simulations=0,
                        support=radii,
                        area=area,
                        phenotypes=phenotypes,
                        copy=True)

        # Get the L statistics
        returned_l_stats = result["L_stat"]["stats"].to_numpy()

        # Check that the returned statistic for the two phenotypes is "A_B"
        unique_phenotypes = result["L_stat"]["phenotype"].unique()
        self.assertEqual(len(unique_phenotypes), 1,
                         "The function should return one unique "
                         "phenotype for the two phenotypes.")
        self.assertEqual(unique_phenotypes[0], "A_B",
                         "The returned phenotype should be 'A_B'")

        for returned_l_stat, gt_l_stat in zip(returned_l_stats,
                                              ground_truth_l_stats):
            self.assertAlmostEqual(gt_l_stat, returned_l_stat, places=5,
                                   msg="The ground truth L statistics "
                                       "should match the returned statistics")

    def test_csr_same_phenotype(self):
        """
        Test the L statitic under complete spatial randomness
        while using the same phenotype
        """

        phenotype_name = ["A"]
        n_cells = 2000
        phenotypes = phenotype_name * n_cells
        features = np.random.rand(n_cells)
        # Generate spatial_x at random float position in the square
        x_max = 200
        y_max = 200
        spatial_x = np.random.rand(n_cells) * x_max
        spatial_y = np.random.rand(n_cells) * y_max

        # Keep radius relatively small to avoid boundary adjustment
        radii = [0, 1, 2,  3, 4, 5]

        # Create a dataframe out of phenotypes, features, spatial coordinates
        dictionary = {'phenotype': phenotypes, 'feature': features,
                      'spatial_x': spatial_x, 'spatial_y': spatial_y}
        dataframe = pd.DataFrame(dictionary)

        self.adata = self.create_dummy_dataset(dataframe)

        result = ripley(self.adata,
                        cluster_key="phenotype",
                        mode="L",
                        n_simulations=100,
                        support=radii,
                        area=x_max*y_max,
                        phenotypes=("A", "A"),
                        copy=True)

        # Get the L statistics
        sim_stats_mean = result["sims_stat"].groupby("bins")["stats"].mean()

        # Loop over all raddi and ground truth L statistics
        # L(R) = R under complete spatial randomness ignoring boundry
        # conditions
        for r, gt_l_stat_mean in zip(radii, sim_stats_mean):
            difference = abs(r-gt_l_stat_mean)
            self.assertLess(
                difference,
                0.1,
                msg="The ground truth L statistics should match the returned statistics"
            )


    def test_csr_two_phenotype(self):
        """
        Test the L statitic under complete spatial randomness
        while using the same phenotype
        """

        center_phenotype = ["A"]
        n_center_cells = 1000
        neighbor_phenotype = ["B"]
        n_neighbor_cells = 1000

        # As the cells are randomly placed, assign them
        # random phenotypes
        phenotypes = center_phenotype * n_center_cells + \
            neighbor_phenotype * n_neighbor_cells

        features = np.random.rand(n_center_cells + n_neighbor_cells)

        # Generate spatial_x at random float position in the square
        x_max = 200
        y_max = 200
        total_cells = n_center_cells + n_neighbor_cells
        spatial_x = np.random.rand(total_cells) * x_max
        spatial_y = np.random.rand(total_cells) * y_max

        # Keep radius relatively small to avoid boundary adjustment
        radii = [0, 1, 2,  3, 4, 5]

        # Create a dataframe out of phenotypes, features, spatial coordinates
        dictionary = {'phenotype': phenotypes, 'feature': features,
                      'spatial_x': spatial_x, 'spatial_y': spatial_y}
        dataframe = pd.DataFrame(dictionary)

        self.adata = self.create_dummy_dataset(dataframe)

        phenotypes = (center_phenotype, neighbor_phenotype)
        result = ripley(self.adata,
                        cluster_key="phenotype",
                        mode="L",
                        n_simulations=100,
                        support=radii,
                        area=x_max*y_max,
                        phenotypes=phenotypes,
                        copy=True)

        # Get the L statistics
        sim_stats_mean = result["sims_stat"].groupby("bins")["stats"].mean()

        # Loop over all raddi and ground truth L statistics
        # L(R) = R under complete spatial randomness ignoring boundry
        # conditions
        for r, gt_l_stat_mean in zip(radii, sim_stats_mean):
            difference = abs(r-gt_l_stat_mean)
            self.assertLess(
                difference,
                0.1,
                msg="The ground truth L statistics should match the returned statistics"
            )


    def test_n_cells_returned(self):
        """
        Make sure n_center and n_neighbor cells are returned
        """

        # Add four points to form a rectangle that is 1 * 2
        dataframe = [
                    {'phenotype': 'A', 'feature': 10, 'spatial_x': 0,  'spatial_y': 0},
                    {'phenotype': 'B', 'feature': 20, 'spatial_x': 1,  'spatial_y': 0},
                    {'phenotype': 'B', 'feature': 20, 'spatial_x': 0,  'spatial_y': 2},
                    {'phenotype': 'B', 'feature': 20, 'spatial_x': 1,  'spatial_y': 2},
                    {'phenotype': 'A', 'feature': 20, 'spatial_x': 1,  'spatial_y': 2},
                ]

        dataframe = pd.DataFrame(dataframe)
        self.adata = self.create_dummy_dataset(dataframe)

        center_phenotype = "A"
        neighbor_phenotype = "B"

        phenotypes = (center_phenotype, neighbor_phenotype)
        result = ripley(self.adata,
                        cluster_key="phenotype",
                        mode="L",
                        n_simulations=0,
                        phenotypes=phenotypes,
                        copy=True)

        # Check that the correct number of cells are returned
        self.assertEqual(result["n_center"], 2)
        self.assertEqual(result["n_neighbor"], 3)

