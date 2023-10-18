import unittest
from spac.visualization import interative_spatial_plot
import plotly.graph_objs as go
import anndata
import pandas as pd
import numpy as np


class TestInteractiveSpatialPlot(unittest.TestCase):

    def setUp(self):
        # Create a mock `adata` object with the necessary attributes
        X = np.array([[1], [2], [3]])
        self.adata = anndata.AnnData(X)
        self.adata.obs = pd.DataFrame(
            {'annotation_1': ['a', 'b', 'c'],
             'annotation_2': ['x', 'y', 'z']}
        )
        spatial_coords = np.array([[0, 0], [1, 1], [2, 2]])
        self.adata.obsm["spatial"] = spatial_coords

    def test_error_raised(self):
        # Test if the function raises the expected error message
        error_msg = "Provided annotation should be a string " + \
            "or a list of strings, get <class 'int'> for 1234 entry."

        with self.assertRaisesRegex(
            TypeError,
            error_msg

        ):
            interative_spatial_plot(self.adata, [1234])  # Invalid annotation

    def test_return_object(self):
        # Test if the return object is of the correct type
        fig = interative_spatial_plot(self.adata, 'annotation_1')
        self.assertIsInstance(fig, go.Figure)

    def test_image_configuration(self):
        # Assuming you expect certain configurations
        # like a specific width and height
        fig = interative_spatial_plot(
            self.adata,
            'annotation_1',
            figure_width=15,
            figure_height=10,
            figure_dpi=100)
        self.assertEqual(fig.layout.width, 15 * 100)
        self.assertEqual(fig.layout.height, 10 * 100)

    def test_correct_image(self):
        # Still in progress
        fig = interative_spatial_plot(self.adata, 'annotation_1')
        # For demonstration, let's just check
        # a single property of the plotly figure:
        self.assertEqual(len(fig.data), 3)


if __name__ == "__main__":
    unittest.main()
