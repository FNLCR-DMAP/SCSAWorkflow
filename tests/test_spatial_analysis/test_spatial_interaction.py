import anndata
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spac.spatial_analysis import spatial_interaction


class TestSpatialInteraction(unittest.TestCase):
    def setUp(self):
        # Create a mock AnnData object for testing
        obs = pd.DataFrame({
            "cluster_num": [1, 2, 1, 2, 1, 2, 1, 2],
            "cluster_str": [
                "A", "B", "A", "B",
                "A", "B", "A", "B"
                ]
        })
        features = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
                [1.6, 1.7, 1.8],
                [1.9, 2.0, 2.1],
                [2.2, 2.3, 2.4]
            ]
        )
        spatial_coords = np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
                [0.7, 0.8],
                [0.9, 1.0],
                [1.1, 1.2],
                [1.3, 1.4],
                [1.5, 1.6]
            ]
        )
        self.adata = anndata.AnnData(X=features, obs=obs)
        self.adata.obsm['spatial'] = spatial_coords
        self.run_CI = False

    def test_spatial_interaction_invalid_data_type(self):
        # Invalid data type test
        invalid_data = "not an AnnData object"
        observation = "valid_observation"
        analysis_method = "Neighborhood Enrichment"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                adata=invalid_data,
                observation=observation,
                analysis_method=analysis_method
            )

        self.assertIsInstance(cm.exception, ValueError)
        self.assertEqual(
            str(cm.exception),
            "Input data is not an AnnData object. Got <class 'str'>"
            )

    def test_spatial_interaction_observation_not_found(self):
        # Feature not found test
        observation = "nonexistent_observation"
        analysis_method = "Cluster Interaction Matrix"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                self.adata,
                observation,
                analysis_method
            )

        expect_string = "Observation nonexistent_observation not " + \
            "found in the dataset. Existing observations " + \
            "are: cluster_num,cluster_str"
        self.assertIsInstance(cm.exception, ValueError)
        print(str(cm.exception))
        self.assertEqual(
            str(cm.exception),
            expect_string
        )

    def test_spatial_interaction_invalid_analysis_method(self):
        # Invalid analysis method test
        observation = "cluster_str"
        invalid_analysis_method = "Invalid Method"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                self.adata,
                observation,
                invalid_analysis_method
            )

        expect_string = "Method Invalid Method is not supported " + \
            "currently. Available methods are: " + \
            "Neighborhood Enrichment,Cluster Interaction Matrix"

        self.assertIsInstance(cm.exception, ValueError)
        self.assertEqual(
            str(cm.exception),
            expect_string
        )

    def test_spatial_interaction_invalid_ax_type(self):
        # Invalid ax type test
        observation = "cluster_str"
        analysis_method = "Neighborhood Enrichment"
        invalid_ax = "not an Axes object"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                self.adata,
                observation,
                analysis_method,
                ax=invalid_ax
            )

        error_str = "Invalid 'ax' argument. Expected an instance " + \
            "of matplotlib.axes.Axes. Got <class 'str'>"

        self.assertIsInstance(cm.exception, ValueError)
        self.assertEqual(
            str(cm.exception),
            error_str
        )

    def test_neighborhood_enrichment_analysis(self):
        # Test Neighborhood Enrichment analysis
        ax = plt.gca()
        spatial_interaction(
            self.adata,
            "cluster_num",
            "Neighborhood Enrichment",
            ax=ax)

        # Verify that Neighborhood Enrichment analysis is performed and plotted
        # Assertion 1: Check if Neighborhood Enrichment analysis is performed
        self.assertTrue("cluster_num_plot" in self.adata.obs)

        # Assertion 2: Check if the resulting plot is displayed
        self.assertTrue(plt.gcf().get_axes())

    def test_custom_axes_provided(self):
        # Test custom matplotlib Axes provided
        fig, ax = plt.subplots()
        # Set the desired x-axis limits
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        returned_ax = spatial_interaction(
            self.adata,
            "cluster_str",
            "Neighborhood Enrichment",
            ax=ax)

        # Assert that the returned ax object is the same
        # as the input ax object
        self.assertEqual(id(returned_ax), id(ax))

        # Verify that the provided Axes is used for plotting
        self.assertEqual(returned_ax.get_xlim(), (-0.5, 1.5))
        self.assertEqual(returned_ax.get_ylim(), (-0.5, 0.5))

        # Clean up the figure
        plt.close(fig)

    def test_no_axes_provided(self):
        # Test no matplotlib Axes provided
        spatial_interaction(
            self.adata,
            "cluster_str",
            "Neighborhood Enrichment")
        # Verify that a new Axes is created and used for plotting
        self.assertTrue(plt.gcf().get_axes())

    def test_additional_kwargs(self):
        # Test additional keyword arguments for matplotlib.pyplot.text()
        kwargs = {"color": "red"}
        spatial_interaction(
            self.adata,
            "cluster_str",
            "Neighborhood Enrichment",
            **kwargs)
        # Verify that the additional keyword arguments are
        # passed to matplotlib.pyplot.text()
        # Assertion goes here

    def test_returned_ax_has_right_titles(self):
        observation = "cluster_str"
        analysis_method = "Neighborhood Enrichment"

        # Create a blank figure
        fig = plt.figure()

        # Get the ax object associated with the figure
        ax = fig.add_subplot(111)

        # Call the function
        returned_ax = spatial_interaction(
            self.adata,
            observation,
            analysis_method,
            ax=ax
        )

        # Assert that the returned ax object is not None
        self.assertIsNotNone(returned_ax)

        if self.run_CI:
            figure = returned_ax.figure
            axes_list = figure.axes

            current_values = [
                axes_list[2].get_title(),
                axes_list[1].get_ylabel(),
                axes_list[1].get_yticklabels()[1].get_text(),
                axes_list[1].get_yticklabels()[0].get_text()
            ]

            expect_values = [
                'Neighborhood enrichment',
                observation + "_plot",
                'A',
                'B'
            ]
            for i in range(len(current_values)):
                error_msg = f"Value at index {i} " + \
                    f"is different. Got '{current_values[i]}', " + \
                    f"expected '{expect_values[i]}'"

                self.assertEqual(
                    current_values[i],
                    expect_values[i],
                    error_msg
                )

    def test_new_ax_has_right_titles(self):
        observation = "cluster_num"
        analysis_method = "Neighborhood Enrichment"

        # Call the function
        returned_ax = spatial_interaction(
            self.adata,
            observation,
            analysis_method
        )

        # Assert that the returned ax object is not None
        self.assertIsNotNone(returned_ax)

        # Assert that the returned ax object is the same
        # as the input ax object
        self.assertIsInstance(returned_ax, plt.Axes)

        if self.run_CI:
            figure = returned_ax.figure
            axes_list = figure.axes

            current_values = [
                axes_list[2].get_title(),
                axes_list[1].get_ylabel(),
                axes_list[1].get_yticklabels()[1].get_text(),
                axes_list[1].get_yticklabels()[0].get_text()
            ]

            expect_values = [
                'Neighborhood enrichment',
                observation + "_plot",
                '1',
                '2'
            ]
            for i in range(len(current_values)):
                error_msg = f"Value at index {i} " + \
                    f"is different. Got '{current_values[i]}', " + \
                    f"expected '{expect_values[i]}'"

                self.assertEqual(
                    current_values[i],
                    expect_values[i],
                    error_msg
                )

    def tearDown(self):
        del self.adata


if __name__ == '__main__':
    unittest.main()
