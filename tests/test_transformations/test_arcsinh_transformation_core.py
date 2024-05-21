import unittest
import numpy as np
from spac.transformations import arcsinh_transformation_core


class TestArcsinhTransformationCore(unittest.TestCase):

    def setUp(self):
        self.data = np.array([
            [2, 5],
            [4, 10],
            [6, 15],
            [8, 20],
            [10, 25],
            [12, 30]
        ])

    def test_arcsinh_transformation_with_percentile(self):
        transformed_data = arcsinh_transformation_core(self.data,
                                                       percentile=20)
        expected_data = np.array([
            [0.48121183, 0.48121183],
            [0.88137359, 0.88137359],
            [1.19476322, 1.19476322],
            [1.44363548, 1.44363548],
            [1.64723115, 1.64723115],
            [1.81844646, 1.81844646]
        ])
        # print("Actual:", transformed_data)
        # print("Expected:", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_data, expected_data, decimal=5)

    def test_arcsinh_transformation_with_fixed_co_factor(self):
        transformed_data = arcsinh_transformation_core(self.data,
                                                       co_factor=5.0)
        # Hardcoded expected transformation using fixed co-factor of 5
        # expected_data = np.arcsinh(self.data / 5.0)
        expected_data = np.array([
            [0.39003533, 0.88137359],
            [0.73266826, 1.44363548],
            [1.01597313, 1.81844646],
            [1.24898253, 2.09471255],
            [1.44363548, 2.31243834],
            [1.60943791, 2.49177985]
        ])
        # print("Actual:", transformed_data)
        # print("Expected:", expected_data)
        np.testing.assert_array_almost_equal(
            transformed_data, expected_data, decimal=5)

    def test_arcsinh_transformation_invalid_percentile(self):
        with self.assertRaises(ValueError) as context:
            arcsinh_transformation_core(self.data, percentile=-0.1)
        self.assertEqual(str(context.exception),
                         "Percentile should be between 0 and 100.")

        with self.assertRaises(ValueError) as context:
            arcsinh_transformation_core(self.data, percentile=100.1)
        self.assertEqual(str(context.exception),
                         "Percentile should be between 0 and 100.")

    def test_arcsinh_transformation_missing_parameters(self):
        with self.assertRaises(ValueError) as context:
            arcsinh_transformation_core(self.data)
        self.assertEqual(str(context.exception),
                         "Either co_factor or percentile must be provided.")

    def test_arcsinh_transformation_error_on_both_parameters(self):
        with self.assertRaises(ValueError) as context:
            arcsinh_transformation_core(self.data, co_factor=5.0,
                                        percentile=20)
        self.assertEqual(str(context.exception),
                         "Please specify either co_factor or percentile, "
                         "not both.")


if __name__ == "__main__":
    unittest.main()
