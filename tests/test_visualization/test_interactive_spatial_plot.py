import unittest
from spac.visualization import interative_spatial_plot
import plotly.graph_objs as go
import anndata
import pandas as pd
import numpy as np
import numbers


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

    def test_hover_values_annotation(self):
        # Use the existing adata from setUp with annotation_1.
        fig_list = interative_spatial_plot(self.adata, 'annotation_1')
        self.assertIsInstance(fig_list, list)
        fig = fig_list[0]['image_object']
        # In annotation plots, we set a hovertemplate that displays customdata.
        # Our implementation sets hovertemplate "%{customdata[0]}<extra></extra>"
        for trace in fig.data:
            # Dummy annotation trace may not have customdata, so check only for data traces.
            if trace.name != "<b>annotation_1</b>":
                self.assertEqual(trace.hovertemplate, "%{customdata[0]}<extra></extra>")
                # customdata should contain the annotation values.
                # Since our setUp has ['a', 'b', 'c'], check that the first customdata value is one of them.
                self.assertIn(trace.customdata[0][0], ['a', 'b', 'c'])

    def test_annotation_plot(self):
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

            if idx == 0:
                continue
                # The first trace is the group label
            else:
                self.assertEqual(trace.x[0], expected_x[idx - 1])
                self.assertEqual(trace.y[0], expected_y[idx - 1])

        # Every plot has a "hiddnen" point for the group label
        # Here it is called "annotation_1"
        # and it is appended to the end of the points

        expected_names = [
            "<b>annotation_1</b>",
            "a",
            "b",
            "c"
        ]
        for idx, trace in enumerate(fig.data):
            self.assertEqual(trace.name, expected_names[idx])
    
    def test_feature_plot(self):
        # Create a mock adata for continuous feature plots.
        X = np.array([[5], [15], [25]])
        adata_feature = anndata.AnnData(X)
        adata_feature.var_names = ['gene1']
        spatial_coords = np.array([[0, 0], [1, 1], [2, 2]])
        adata_feature.obsm["spatial"] = spatial_coords
        # Call interative_spatial_plot using a feature (continuous)
        fig_list = interative_spatial_plot(
            adata_feature,
            feature='gene1',
            dot_size=10,
            feature_colorscale="balance")
        self.assertIsInstance(fig_list, list)
        fig = fig_list[0]['image_object']
        # For a continuous plot, there should be only one trace
        self.assertEqual(len(fig.data), 1)
        # Check that the trace is a scatterplot using continuous scale
        trace = fig.data[0]
        self.assertEqual(trace.type, 'scattergl')
        # The color values should be numeric and based on the gene expression
        for val in trace.marker.color:
            self.assertIsInstance(val, numbers.Number)

        # Save the figure as HTML
        fig.write_html("test_feature_plot.html")

    def test_stratify_feature_plot(self):
        # Create a mock adata with continuous feature and a stratification column.
        X = np.array([[5], [15], [25], [35]])
        adata_strat = anndata.AnnData(X)
        adata_strat.var_names = ['gene1']
        adata_strat.obs = pd.DataFrame({
            'group': ['A', 'B', 'A', 'B']
        })
        spatial_coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        adata_strat.obsm["spatial"] = spatial_coords

        # Now call interative_spatial_plot with feature parameter
        # and stratify_by "group"
        fig_list = interative_spatial_plot(
            adata_strat,
            feature='gene1',
            stratify_by='group'
        )
        # Expect two figures: one for group A and one for group B
        self.assertEqual(len(fig_list), 2)
        for fig_dict in fig_list:
            fig = fig_dict['image_object']
            # In continuous feature plots all traces should be scatter traces
            for trace in fig.data:
                self.assertEqual(trace.type, 'scattergl')
                # Since using continuous coloring,
                # the marker color should be numeric;
                # we check the first value in marker.color if available.
                if isinstance(trace.marker.color, (list, np.ndarray)):
                    self.assertIsInstance(
                        trace.marker.color[0], numbers.Number)
                else:
                    self.assertIsInstance(trace.marker.color, numbers.Number)

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
                if idx != 0:
                    self.assertEqual(trace.name, expected_labels[i][idx - 1])
                    if trace.name in expected_colors.keys():
                        self.assertEqual(
                            trace.marker.color,
                            expected_colors[trace.name]
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

    def test_multiple_annotations_legend_and_color_order(self):
        defined_color_map = {
                'a': 'red',
                'b': 'blue',
                'c': 'green',
                'x': 'yellow',
                'y': 'pink',
                'z': 'black'
            }
        self.adata.uns['test_color_mapping'] = defined_color_map
        fig_list = interative_spatial_plot(
            self.adata,
            ['annotation_1','annotation_2'],
            defined_color_map='test_color_mapping',
            dot_size=10,
            reverse_y_axis=True
        )
        legend_order = [
            "<b>annotation_1</b>",
            'a',
            'b',
            'c',
            "<b>annotation_2</b>",
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
        # Save the figure as HTML
        fig.write_html("test_multiple_annotations_legend_and_color_order.html")
    

if __name__ == "__main__":
    unittest.main()
