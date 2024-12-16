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
        fig_list = interative_spatial_plot(self.adata, 'annotation_1')
        self.assertIsInstance(fig_list, list)
        self.assertIsInstance(fig_list[0], dict)
        self.assertIsInstance(fig_list[0]['image_name'], str)
        self.assertIsInstance(fig_list[0]['image_object'], go.Figure)

    def test_image_configuration(self):
        # Assuming you expect certain configurations
        # like a specific width and height
        fig_list = interative_spatial_plot(
            self.adata,
            ['annotation_1','annotation_2'],
            figure_width=15,
            figure_height=10,
            figure_dpi=100)
        fig = fig_list[0]['image_object']
        self.assertEqual(fig.layout.width, 15 * 100)
        self.assertEqual(fig.layout.height, 10 * 100)

    def test_correct_image(self):
        fig_list = interative_spatial_plot(
            self.adata,
            'annotation_1'
        )
        fig = fig_list[0]['image_object']

        # Check that all plots are scatter plots
        for trace in fig.data:
            print(trace)
            self.assertEqual(trace.type, 'scattergl')
        # Check the x and y data
        expected_x = self.adata.obsm['spatial'][:, 0]
        expected_y = self.adata.obsm['spatial'][:, 1]
        for idx, trace in enumerate(fig.data):
            if idx == 0:
                continue
                # The first trace is the group label, data is the following
            else:
                self.assertEqual(trace.x[0], expected_x[idx - 1])
                self.assertEqual(trace.y[0], expected_y[idx - 1])

        # Check the annotations/colors
        # Assuming 'annotation_1' in your adata has unique colors
        # that are represented in the plot
        expected_colors = [
            "<b>annotation_1</b>",
            "a",
            "b",
            "c"      
        ]
        for idx, trace in enumerate(fig.data):
            self.assertEqual(trace.name, expected_colors[idx])




if __name__ == "__main__":
    unittest.main()
