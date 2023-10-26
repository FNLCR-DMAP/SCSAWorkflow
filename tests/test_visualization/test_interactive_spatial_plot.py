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

    def test_return_object(self):
        # Test if the return object is of the correct type
        fig = interative_spatial_plot(self.adata, 'annotation_1')
        self.assertIsInstance(fig, go.Figure)

    def test_image_configuration(self):
        # Assuming you expect certain configurations
        # like a specific width and height
        fig = interative_spatial_plot(
            self.adata,
            ['annotation_1','annotation_2'],
            figure_width=15,
            figure_height=10,
            figure_dpi=100)
        self.assertEqual(fig.layout.width, 15 * 100)
        self.assertEqual(fig.layout.height, 10 * 100)

    def test_correct_image(self):
        fig = interative_spatial_plot(self.adata, 'annotation_1')

        # Check that all plots are scatter plots
        for trace in fig.data:
            self.assertEqual(trace.type, 'scatter')

        # Check the x and y data
        expected_x = self.adata.obsm['spatial'][:, 0]
        expected_y = self.adata.obsm['spatial'][:, 1]
        for idx, trace in enumerate(fig.data):
            self.assertEqual(trace.x[0], expected_x[idx])
            self.assertEqual(trace.y[0], expected_y[idx])

        # Check the annotations/colors
        # Assuming 'annotation_1' in your adata has unique colors
        # that are represented in the plot
        expected_colors = [
            "annotation_1_a",
            "annotation_1_b",
            "annotation_1_c",
            "annotation_2_x",
            "annotation_2_y",
            "annotation_2_z"            
        ]
        for idx, trace in enumerate(fig.data):
            self.assertEqual(trace.customdata[0], expected_colors[idx])


if __name__ == "__main__":
    unittest.main()
