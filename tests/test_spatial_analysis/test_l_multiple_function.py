import unittest
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Import the function under test; adjust the path as needed.
from spac._ripley import _l_multiple_function


class TestLMultipleFunction(unittest.TestCase):

    def _setup_geometry(self, center_coord, neighbor_coord):
        """
        Helper function to compute the distance matrix, convex hull,
        and area from the given center and neighbor coordinates.
        """
        # Compute pairwise distance matrix between center and neighbor points.
        distances = cdist(center_coord, neighbor_coord)
        # Stack center and neighbor points for convex hull computation.
        all_points = np.vstack([center_coord, neighbor_coord])
        hull = ConvexHull(all_points)
        # For 2D convex hulls, the 'volume' attribute gives the area.
        area = hull.volume
        return distances, hull, area

    def _create_minimal_scenario(self):
        # Define center points manually.
        center_coord = np.array([
            # 4 corners
            [0, 0], [0, 100], [100, 0], [100, 100],
            # Expected to be excluded at support=5
            [5.1, 4.9], [95.1, 5.1], [5.1, 95.1], [95.1, 94.1],
            # Expected to be excluded at support=10
            [10.1, 9.9], [90.1, 10.1], [8.9, 89.9], [90.1, 89.9],
            # Center point that should not be excluded at support=5 or 10
            [50, 50]
        ])

        # Define neighbor points manually.
        neighbor_coord = np.array([
            [2, 2],    # Too close to border at support=5
            [12, 12],  # Counted at support=5 but excluded at support=10
        ])

        # Define support values (distance thresholds).
        support = np.array([5, 10])

        # Setup geometry: compute pairwise distance matrix, convex hull, and area.
        distances, hull, area = self._setup_geometry(
            center_coord,
            neighbor_coord
        )

        return center_coord, neighbor_coord, support, distances, hull, area

    def test_l_multiple_function_edge_effect(self):
        """
        Test the _l_multiple_function with a minimal scenario that handles
        edge correction.
        """
        center_coord, neighbor_coord, support, distances, hull, area = \
            self._create_minimal_scenario()

        # Set the expected valid mask as a 2D numpy array (2 * 13):
        expected_valid_mask = np.array([
            # First row for support=5:
            # Corners are excluded:
            [0, 0, 0, 0,
             # Four points are excluded at support 5
             0, 0, 0, 0,
             # Rest of points are included
             1, 1, 1, 1, 1],
            # Second row for support=10:
            # all points are excluded except for the last center point
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=bool)

        num_center = center_coord.shape[0]
        num_neighbor = neighbor_coord.shape[0]

        # Call the function under test.
        result = _l_multiple_function(
            distances=distances,
            support=support,
            n_center=num_center,
            n_neighbor=num_neighbor,
            area=area,
            remove_diagonal=False,
            center_coord=center_coord,
            hull=hull,
            edge_correction=True,
            return_mask=True
        )

        # Check that the result is a dictionary with expected keys.
        expected_keys = {
            "support", "l_estimate", "used_center_points", "valid_mask"
        }
        self.assertTrue(expected_keys.issubset(result.keys()))

        # Check that "l_estimate" has the same shape as support.
        self.assertEqual(result["l_estimate"].shape, support.shape)

        # Check that "used_center_points" has one value per support level.
        self.assertEqual(result["used_center_points"].shape, (len(support),))

        # Check that the valid_mask has shape (num_support, num_center)
        self.assertEqual(result["valid_mask"].shape, (len(support), num_center))

        # Verify that the valid_mask matches our expectations
        np.testing.assert_array_equal(
            result["valid_mask"],
            expected_valid_mask
        )

        # Verify the Ripley L calculation
        # used_central_points = np.array([5, 1])
        # n_pairs_less_than_support = np.array([1, 0])
        # n_neighbors = 2
        # k_estimate = np.array([1000, 0])
        # l_estimate = np.sqrt(k_estimate / np.pi)
        expected_ripley_l = np.array([
            np.sqrt(1000 / np.pi),
            0
        ])

        np.testing.assert_allclose(
            result["l_estimate"],
            expected_ripley_l,
            rtol=1e-5
        )

    def test_l_multiple_function_no_edge(self):
        """
        Test the _l_multiple_function with a minimal scenario
        without edge correction.
        """
        center_coord, neighbor_coord, support, distances, hull, area = \
            self._create_minimal_scenario()

        # All points are included in the valid mask.
        expected_valid_mask = np.ones((len(support), center_coord.shape[0]), dtype=bool)

        num_center = center_coord.shape[0]
        num_neighbor = neighbor_coord.shape[0]

        # Call the function under test.
        result = _l_multiple_function(
            distances=distances,
            support=support,
            n_center=num_center,
            n_neighbor=num_neighbor,
            area=area,
            remove_diagonal=False,
            center_coord=center_coord,
            hull=hull,
            edge_correction=False,
            return_mask=True
        )

        # Check that the valid_mask has shape (num_support, num_center)
        self.assertEqual(result["valid_mask"].shape, (len(support), num_center))

        # Verify that the valid_mask matches our expectations
        np.testing.assert_array_equal(
            result["valid_mask"],
            expected_valid_mask
        )

        # Verify the Ripley L calculation
        # used_central_points = np.array([13, 13])
        # n_pairs_less_than_support (neibhors are counted twice) = np.array([3, 4])
        # n_neighbors = 2
        expected_ripley_l = np.array([
            19.164567,
            22.129336
        ])

        np.testing.assert_allclose(
            result["l_estimate"],
            expected_ripley_l,
            rtol=1e-5
        )




if __name__ == '__main__':
    unittest.main()
