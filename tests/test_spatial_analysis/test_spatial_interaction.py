import anndata
import unittest
import numpy as np
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
from spac.spatial_analysis import spatial_interaction


class TestSpatialInteraction(unittest.TestCase):
    def setUp(self):
        # Create a mock AnnData object for testing
        repetition = 2
        annotation = pd.DataFrame({
                    "cluster_num": [1, 1, 1, 1, 2, 2, 2, 2] * repetition,
                    "cluster_str": [
                        "A", "A", "B", "A",
                        "B", "B", "A", "B"
                        ] * repetition,
                    "cluster_str2": [
                        "Un", "De", "De", "De",
                        "Un", "Un", "Un", "De"
                        ] * repetition
                })

        features = np.array(
                    [
                        [1, 3, 5],
                        [1, 3, 6],
                        [1, 4, 5],
                        [1, 4, 6],
                        [2, 3, 5],
                        [2, 3, 6],
                        [2, 4, 5],
                        [2, 4, 6]
                    ])

        n_features = np.tile(features, (repetition, 1))

        spatial_coords = np.array([
            [1, 1],
            [1, 11],
            [1, 11],
            [1, 1],
            [2, 1],
            [2, 22],
            [2, 22],
            [2, 1]
        ])
        n_spatial_coords = np.tile(spatial_coords, (repetition, 1))
        self.adata = anndata.AnnData(X=n_features, obs=annotation)
        self.adata.obsm['spatial'] = n_spatial_coords
        self.run_CI = False

    def test_spatial_interaction_invalid_data_type(self):
        # Invalid data type test
        invalid_data = "not an AnnData object"
        annotation = "valid_annotation"
        analysis_method = "Neighborhood Enrichment"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                adata=invalid_data,
                annotation=annotation,
                analysis_method=analysis_method
            )

        self.assertIsInstance(cm.exception, ValueError)
        self.assertEqual(
            str(cm.exception),
            "Input data is not an AnnData object. Got <class 'str'>"
            )

    def test_spatial_interaction_annotation_not_found(self):
        # Feature not found test
        annotation = "nonexistent_annotation"
        analysis_method = "Cluster Interaction Matrix"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                self.adata,
                annotation,
                analysis_method
            )

        expect_string = "The annotation 'nonexistent_annotation' " + \
                        "does not exist in the provided dataset.\n" + \
                        "Existing annotations are:\n" + \
                        "cluster_num\ncluster_str\ncluster_str2"
        self.assertIsInstance(cm.exception, ValueError)
        print(str(cm.exception))
        self.assertEqual(
            str(cm.exception),
            expect_string
        )

    def test_spatial_interaction_invalid_analysis_method(self):
        # Invalid analysis method test
        annotation = "cluster_str"
        invalid_analysis_method = "Invalid Method"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                self.adata,
                annotation,
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
        annotation = "cluster_str"
        analysis_method = "Neighborhood Enrichment"
        invalid_ax = "not an Axes object"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                self.adata,
                annotation,
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
        returned_ax_dict = spatial_interaction(
            self.adata,
            "cluster_str",
            "Neighborhood Enrichment",
            ax=ax)

        # Assert that the returned ax object is the same
        # as the input ax object
        returned_ax = returned_ax_dict['Full']
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
        annotation = "cluster_str"
        analysis_method = "Neighborhood Enrichment"

        # Create a blank figure
        fig = plt.figure()

        # Get the ax object associated with the figure
        ax = fig.add_subplot(111)

        # Call the function
        returned_ax = spatial_interaction(
            self.adata,
            annotation,
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
                annotation + "_plot",
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
        annotation = "cluster_num"
        analysis_method = "Neighborhood Enrichment"

        # Call the function
        returned_ax_dict = spatial_interaction(
            self.adata,
            annotation,
            analysis_method
        )

        returned_ax = returned_ax_dict['Full']

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
                annotation + "_plot",
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

    def test_sinlge_stratify_by(self):
        ax_dict = spatial_interaction(
            self.adata,
            "cluster_num",
            "Neighborhood Enrichment",
            stratify_by="cluster_str"
        )
        unique_cluster_str_values = self.adata.obs["cluster_str"].unique()

        # Get the keys (unique cluster values) from the dictionary
        keys = list(ax_dict.keys())

        # Assert that we have at least two keys (clusters)
        self.assertEqual(len(keys), 2)

        # Assert that the axes associated
        # with the first and second keys are different
        self.assertNotEqual(ax_dict[keys[0]], ax_dict[keys[1]])

        for value in unique_cluster_str_values:
            # Expect each unique value as a key in the returned dict
            self.assertIn(value, ax_dict.keys())

            # Each should be a matplotlib axis object
            self.assertIsInstance(ax_dict[value], plt.Axes)

    def test_list_stratify_by(self):
        ax_dict = spatial_interaction(
            self.adata,
            "cluster_str",
            "Neighborhood Enrichment",
            stratify_by=["cluster_num"]
            )
        combined_keys = self.adata.obs[
                "cluster_num"
            ].astype(str).unique()

        for key in combined_keys:
            # Expect each combined key as a key in the returned dict
            self.assertIn(key, ax_dict.keys())

            # Each should be a matplotlib axis object
            self.assertIsInstance(ax_dict[key], plt.Axes)

    def test_return_matrix_and_stratify_by_combinations(self):
        annotation = "cluster_num"
        analysis_method = "Cluster Interaction Matrix"
        stratify_options = [None, "cluster_str"]

        for stratify_by in stratify_options:
            for return_matrix in [False, True]:
                with self.subTest(
                        stratify_by=stratify_by,
                        return_matrix=return_matrix
                ):
                    result = spatial_interaction(
                        self.adata,
                        annotation,
                        analysis_method,
                        stratify_by=stratify_by,
                        return_matrix=return_matrix
                    )

                    # Assert that the result is a
                    # list when return_matrix is True
                    if return_matrix:
                        self.assertIsInstance(result, list)
                        self.assertEqual(len(result), 2)
                        # Expect two dictionaries

                        self.assertIsInstance(result, list)
                        self.assertIn("Ax", result[0].keys())
                        self.assertIn("Matrix", result[1].keys())

                        if stratify_by is not None:
                            # If stratification is used, assert that
                            # the keys correspond to the unique values
                            # in the stratification column
                            unique_values = self.adata.obs[
                                stratify_by
                            ].unique()
                            for item in result:
                                for value in unique_values:
                                    if value in item:
                                        self.assertIsInstance(
                                                item["Ax"][value],
                                                plt.Axes
                                            )
                                        self.assertIsInstance(
                                                item["Matrix"][value],
                                                np.ndarray
                                            )
                    else:
                        # When return_matrix is False, assert
                        # that the result is a dictionary
                        self.assertIsInstance(result, dict)

                        if stratify_by is not None:
                            # If stratification is used, assert
                            # that the keys correspond to the unique values
                            # in the stratification column
                            unique_values = self.adata.obs[
                                stratify_by
                            ].unique()
                            for value in unique_values:
                                self.assertIn(value, result)
                                self.assertIsInstance(
                                        result[value],
                                        plt.Axes
                                    )
                        else:
                            # If no stratification is used, assert
                            # that there is only one key
                            self.assertEqual(len(result), 1)
                            self.assertIn("Full", result.keys())

    def test_interaction_matrix_stratify_compute(self):
        ax_dict = spatial_interaction(
            self.adata,
            "cluster_str2",
            "Cluster Interaction Matrix",
            stratify_by="cluster_num",
            return_matrix=True
            )

        expected_ax_dict = {
                    1: array([[24., 10.], [12., 2.]]),
                    2: array([[2.,  8.], [10., 28.]])
                }

        for key, value in ax_dict[1]['Matrix'].items():
            self.assertIn(
                key,
                expected_ax_dict.keys()
            )

            self.assertTrue(
                np.array_equal(
                    value,
                    expected_ax_dict[key]
                    )
                )

    def test_interaction_matrix_no_stratify_compute(self):
        ax_dict = spatial_interaction(
            self.adata,
            "cluster_num",
            "Cluster Interaction Matrix",
            return_matrix=True
            )

        expected_array = array([[36., 24.], [12., 24.]])

        self.assertTrue(
            np.array_equal(
                ax_dict[1]['Matrix'],
                expected_array
            )
        )

    def test_interaction_matrix_compute(self):
        ax_dict = spatial_interaction(
            self.adata,
            "cluster_str2",
            "Neighborhood Enrichment",
            stratify_by="cluster_num",
            return_matrix=True,
            seed=42
            )

        expected_ax_dict = {
            1: (
                array([
                            [-0.89849487, -0.52617258],
                            [0.89849487,  0.52617258]
                        ]),
                array([
                            [24, 10],
                            [12,  2]
                        ])
                ),
            2: (
                array([
                        [0.53420803, -1.1833491],
                        [-0.53420803, 1.1833491]
                    ]),
                array([
                        [2,  8],
                        [10, 28]
                    ])
                )
            }

        for key, tuple_values in ax_dict[1]['Matrix'].items():
            self.assertIn(
                key,
                expected_ax_dict.keys()
            )
            for i in range(len(tuple_values)):
                self.assertTrue(
                    np.allclose(
                        tuple_values[i],
                        expected_ax_dict[key][i]
                    )
                )

    def tearDown(self):
        del self.adata


if __name__ == '__main__':
    unittest.main()
