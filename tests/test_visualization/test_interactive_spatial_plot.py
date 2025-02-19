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
            self.assertEqual(trace.type, 'scattergl')
        # Check the x and y data
        expected_x = self.adata.obsm['spatial'][:, 0]
        expected_y = self.adata.obsm['spatial'][:, 1]

        for idx, trace in enumerate(fig.data):
            if idx == len(fig.data) - 1:
                continue
                # The last trace is the group label
            else:
                self.assertEqual(trace.x[0], expected_x[idx])
                self.assertEqual(trace.y[0], expected_y[idx])

        # Every plot has a "hiddnen" point for the group label
        # Here it is called "annotation_1"
        # and it is appended to the end of the points

        expected_names = [
            "a",
            "b",
            "c",      
            "<b>annotation_1</b>"
        ]
        for idx, trace in enumerate(fig.data):
            self.assertEqual(trace.name, expected_names[idx])

    def test_stratify_by(self):
        fig_list = interative_spatial_plot(
            self.adata,
            'annotation_1',
            stratify_by='annotation_2'
        )

        self.assertEqual(len(fig_list), 3)
        figure_name_list = [
            'Highlighting_annotation_2_x.html',
            'Highlighting_annotation_2_y.html',
            'Highlighting_annotation_2_z.html'
        ]

        # Every plot has a "hiddnen" point for the group label
        # Here it is called "annotation_1"
        # and it is appended to the end of the points
        expected_labels = [
                ["a", "<b>annotation_1</b>"],
                ["b", "<b>annotation_1</b>"],
                ["c", "<b>annotation_1</b>"]
        ]
        expected_colors = {
            'a': 'rgb(127,0,255)',
            'b': 'rgb(128,254,179)',
            'c': 'rgb(255,0,0)'
        }
        for i, itr_fig in enumerate(fig_list):
            
            fig_name = itr_fig['image_name']
            self.assertEqual(fig_name, figure_name_list[i])

            fig = itr_fig['image_object']
            for idx, trace in enumerate(fig.data):
                self.assertEqual(trace.name, expected_labels[i][idx])
                if trace.name in expected_colors.keys():
                    self.assertEqual(
                        trace.marker.color,
                        expected_colors[trace.name]
                    )

    def test_color_mapping_type_check(self):
        defined_color_map = {
                'a': 'red',
                'b': 'blue',
                'c': 'green'
        }
        err_msg = 'The "degfined_color_map" should be ' + \
                "a string getting <class 'dict'>."
        with self.assertRaisesRegex(TypeError, err_msg):
            interative_spatial_plot(
                self.adata,
                'annotation_1',
                defined_color_map=defined_color_map
            )

    def test_color_mapping_value_error(self):
        # Test correct error message is generated when
        # the anndata object does not cotain uns attribute
        defined_color_map = 'test_color_mapping'
        err_msg = (
            "The given color map name: test_color_mapping is not found "
            "in current analysis, available items are: ['example_key']"
        )

        self.adata.uns['example_key'] = 'example_value'

        with self.assertRaises(ValueError) as cm:
            interative_spatial_plot(
                self.adata,
                'annotation_1',
                defined_color_map=defined_color_map
            )
        self.assertEqual(str(cm.exception), err_msg)

    def test_color_mapping_key_error(self):
        # Test correct error message is generated when defined_color_map 
        # does not exist in adata.uns
        defined_color_map = 'test_color_mapping'
        err_msg = (
            'No existing color map found, '
            'please make sure the Append Pin '
            'Color Rules template had been ran '
            'prior to the current visualization node.'
        )


        # Check that 1- the exception is reaised and 2- the error message is correct
        with self.assertRaisesRegex(ValueError, err_msg):
            interative_spatial_plot(
                self.adata,
                'annotation_1',
                defined_color_map=defined_color_map
            )

    def test_color_mapping(self):
        defined_color_map = {
                'a': 'red',
                'b': 'blue',
                'c': 'green'
            }
        self.adata.uns['test_color_mapping'] = defined_color_map
        fig_list = interative_spatial_plot(
            self.adata,
            'annotation_1',
            defined_color_map='test_color_mapping'
        )
        fig = fig_list[0]['image_object']
        for trace in fig.data:
            if trace.name in defined_color_map.keys():
                self.assertEqual(
                    trace.marker.color,
                    defined_color_map[trace.name]
                )
    
    def test_multiple_annotations_legend_order(self):
        defined_color_map = {
                'a': 'red',
                'b': 'blue',
                'c': 'green',
                'x': 'yellow',
                'y': 'pink',
                'z': 'black'
            }
        fig_list = interative_spatial_plot(
            self.adata,W
            ['annotation_1','annotation_2']
        )
        legend_order = [
            'annotation_1',
            'a',
            'b',
            'c',
            'annotation_2',
            'x',
            'y',
            'z'
        ]
        color_order = [
            'white',
            'red',
            'blue',
            'green',
            'white',
            'yellow',
            'pink',
            'black'
        ]
        fig = fig_list[0]['image_object']

        # The order of trace is the order of legend
        # Thus the trace name should follow the designed
        # legend order
        for i, trace in enumerate(fig.data):
            self.assertEqual(
                trace.name,
                legend_order[i]
            )
            self.assertEqual(
                trace.marker.color,
                color_order[i]
            )

if __name__ == "__main__":
    unittest.main()
