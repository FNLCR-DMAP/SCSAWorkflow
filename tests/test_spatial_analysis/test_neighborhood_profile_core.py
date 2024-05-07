import unittest
import numpy as np
from spac.spatial_analysis import _neighborhood_profile_core
import matplotlib.pyplot as plt
import os


def plot_cells_with_distances(coord, phenotypes, distances_bins, save_path):
    plt.figure()
    for i, (x, y) in enumerate(coord):
        plt.scatter(x, y, c='k' if phenotypes[i] == 0 else 'r' if phenotypes[i] == 1 else 'b')

    for radius in distances_bins:
        circle = plt.Circle((0, 0), radius, color='gray', fill=False)
        plt.gca().add_artist(circle)

    plt.xlim(-1, 2)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    plt.savefig(save_path)


class TestNeighborhoodProfileCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a directory to store plots if it doesn't exist
        module_dir = os.path.dirname(os.path.abspath(__file__))
        cls.plot_dir = os.path.join(module_dir, 'plots')
        os.makedirs(cls.plot_dir, exist_ok=True)

    def setUp(self):
        # Remove the figure before each test
        for filename in os.listdir(self.plot_dir):
            file_path = os.path.join(self.plot_dir, filename)
            os.remove(file_path)

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
        expected_profile = np.array([[[0, 0, 0],
                                      [0, 1, 0]],
                                     [[0, 1, 0],
                                      [0, 0, 0]]])
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

        # Save the plot as a PNG file
        save_path = os.path.join(self.plot_dir, 
                                 'three_phenotypes_two_bins.png')

        plot_cells_with_distances(coord, phenotypes, distances_bins, save_path)


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


    def test_three_bins_area_normalized(self):

        # Define input parameters for one cell with one neighbor
        # Four cells, three bins
        coord = np.array([[0, 0], [0.5, 0], [1.5, 0], [2.5, 0]])
        # Phenotypes of the cells
        phenotypes = np.array([0, 0, 0, 0])  
        n_phenotypes = 1  
        # The neighbor cell lies in the center bin
        distances_bins = [0, 1, 2, 3] 
        normalize = 'bin_area'

        # Call the function
        neighborhood_profile = _neighborhood_profile_core(coord, 
                                                          phenotypes, 
                                                          n_phenotypes, 
                                                          distances_bins, 
                                                          normalize)

        bin1_area = np.pi 
        bin2_area = np.pi * (2*2-1)
        bin3_area = np.pi * (3*3 - 2*2)
        # Assert that the neighborhood profile for the center cell as expected 
        expected_profile = np.array([1/bin1_area,
                                     1/bin2_area,
                                     1/bin3_area])

        np.testing.assert_array_equal(neighborhood_profile[0, 0, :],
                                      expected_profile)





if __name__ == '__main__':
    unittest.main()
