import unittest
import numpy as np
from spac.spatial_analysis import _neighborhood_profile_core

class TestNeighborhoodProfileCore(unittest.TestCase):

    def test_one_cell_no_neighbors(self):

        # Define input parameters for one cell with no neighbors
        coord = np.array([[0, 0]])  # One cell at origin
        phenotypes = np.array([0])  # Phenotype of the cell
        n_phenotypes = 1  # Number of unique phenotypes
        distances_bins = [0, 1.01]  # Distance bins, assuming no neighbors
        normalize = None

        # Call the function
        neighborhood_profile = _neighborhood_profile_core(coord,
                                                          phenotypes, 
                                                          n_phenotypes, 
                                                          distances_bins, 
                                                          normalize)

        # Assert that the neighborhood profile is all zeros
        print(neighborhood_profile)
        expected_profile_shape = (1, n_phenotypes, len(distances_bins) - 1)
        self.assertEqual(neighborhood_profile.shape, expected_profile_shape)
        self.assertTrue(np.allclose(neighborhood_profile, 0))

    def test_one_cell_one_neighbor(self):
        from scipy.spatial import distance_matrix

        # Define input parameters for one cell with one neighbor
        # Two cells, one at origin and one neighbor
        coord = np.array([[0, 0], [1, 0]])
        # Phenotypes of the cells
        phenotypes = np.array([0, 1])  
        n_phenotypes = 2  
        distances_bins = [0, 1.01]  
        normalize = None

        # Call the function
        neighborhood_profile = _neighborhood_profile_core(coord, 
                                                          phenotypes, 
                                                          n_phenotypes, 
                                                          distances_bins, 
                                                          normalize)

        # Assert that the neighborhood profile is as expected
        expected_profile = np.array([[[0], [1]], 
                                     [[1], [0]]])
        np.testing.assert_array_equal(neighborhood_profile, expected_profile)


    def test_missing_phenotype(self):
        """
        Run the test with n_phenotypes = 3 where only phenoytpes 0, and 2 are
        present 
        """
        from scipy.spatial import distance_matrix

        # Define input parameters for one cell with one neighbor
        # Two cells, one at origin and one neighbor
        coord = np.array([[0, 0], [1, 0]])
        # Phenotypes of the cells
        phenotypes = np.array([0, 2])  
        n_phenotypes = 3
        distances_bins = [0, 1.01]  
        normalize = None

        # Call the function
        neighborhood_profile = _neighborhood_profile_core(coord, 
                                                          phenotypes, 
                                                          n_phenotypes, 
                                                          distances_bins, 
                                                          normalize)

        # Assert that the neighborhood profile is as expected
        expected_profile = np.array([[[0], [0], [1]], 
                                     [[1], [0], [0]]])
        np.testing.assert_array_equal(neighborhood_profile, expected_profile)




    def test_three_bins(self):

        # Define input parameters for one cell with one neighbor
        # Two cells, one at origin and one neighbor
        coord = np.array([[0, 0], [1, 0]])
        # Phenotypes of the cells
        phenotypes = np.array([0, 1])  
        n_phenotypes = 2  
        # The neighbor cell lies in the center bin
        distances_bins = [0, 0.99, 1.01, 3] 
        normalize = None

        # Call the function
        neighborhood_profile = _neighborhood_profile_core(coord, 
                                                          phenotypes, 
                                                          n_phenotypes, 
                                                          distances_bins, 
                                                          normalize)

        # Assert that the neighborhood profile is as expected
        expected_profile = np.array([[[0, 0, 0], [0, 1, 0]], 
                                     [[0, 1, 0], [0, 0, 0]]])
        np.testing.assert_array_equal(neighborhood_profile, expected_profile)

    def test_three_phenotypes_two_bins(self):

        # Define input parameters for one cell at origin with phenotype 0
        # 1 cell in first bin with phenotype 1
        # 3 cells in first bin with phenotype 2
        # 1 cell in second bin with phenotype 0
        # 1 cell in second bin with phenotype 1
        coord = np.array([[0, 0], 
                          [0.5, 0],
                          [0, 0.5],
                          [0, 0.6],
                          [0, 0.7],
                          [1.5, 0],
                          [1.6, 0]
                          ])
        # Phenotypes of the cells
        phenotypes = np.array([0,
                               1,
                               2, 
                               2, 
                               2, 
                               0,
                               1]) 
        n_phenotypes = 3  
        # The neighbor cell lies in the center bin
        distances_bins = [0, 1, 2]
        normalize = None 

        # Call the function
        neighborhood_profile = _neighborhood_profile_core(coord, 
                                                          phenotypes, 
                                                          n_phenotypes, 
                                                          distances_bins, 
                                                          normalize)

        # Assert that the neighborhood profile is as expected for the 
        # center cell only
        expected_profile = np.array([[0, 1],
                                     [1, 1],
                                     [3, 0]])

        np.testing.assert_array_equal(neighborhood_profile[0, :, :],
                                      expected_profile)


    def test_three_phenotypes_two_bins_count_normalized(self):

        # Define input parameters for one cell at origin with phenotype 0
        # 1 cell in first bin with phenotype 1
        # 3 cells in first bin with phenotype 2
        # 1 cell in second bin with phenotype 0
        # 1 cell in second bin with phenotype 1

        # Third bins with no cells
        # Normalize by the counts per bin
        coord = np.array([[0, 0], 
                          [0.5, 0],
                          [0, 0.5],
                          [0, 0.6],
                          [0, 0.7],
                          [1.5, 0],
                          [1.6, 0]
                          ])
        # Phenotypes of the cells
        phenotypes = np.array([0,
                               1,
                               2, 
                               2, 
                               2, 
                               0,
                               1]) 
        n_phenotypes = 3  
        # The neighbor cell lies in the center bin
        distances_bins = [0, 1, 2, 3]
        normalize = "total_cells" 

        # Call the function
        neighborhood_profile = _neighborhood_profile_core(coord, 
                                                          phenotypes, 
                                                          n_phenotypes, 
                                                          distances_bins, 
                                                          normalize)

        # Assert that the neighborhood profile is as expected for the 
        # center cell only
        expected_profile = np.array([[0/4, 1/2, 0],
                                     [1/4, 1/2, 0],
                                     [3/4, 0/2, 0]])

        np.testing.assert_array_equal(neighborhood_profile[0, :, :],
                                      expected_profile)





if __name__ == '__main__':
    unittest.main()
