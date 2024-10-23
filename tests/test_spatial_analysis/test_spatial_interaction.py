import sys
import anndata
import unittest
import numpy as np
import pandas as pd
import matplotlib as mpl
from numpy import array
import matplotlib.pyplot as plt
from spac.spatial_analysis import spatial_interaction
mpl.use('agg')

class TestSpatialInteraction(unittest.TestCase):

    def create_dummy_dataset(
            self,
            repetition=1
    ):
        cluster_num = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2] * repetition

        phenotypes = [
            "A", "A", "A", "B", "A", "B", "A",
            "B", "A", "B", "A", "B", "A", "B"
        ] * repetition

        analysis_region = (["Region_A"] * 7  + ["Region_B"] * 7) * repetition

        annotation = pd.DataFrame({
            "cluster_num": cluster_num,
            "Phenotypes": phenotypes,
            "Analysis_Region": analysis_region
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
                    ] * 2)

        n_features = np.tile(features, (repetition, 1))

        spatial_coords = np.array([
            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
            [50, 1], [50, 2], [50, 3], [50,4], [50, 5], [50, 6], [50, 7]])

        n_spatial_coords = np.tile(spatial_coords, (repetition, 1))
        adata = anndata.AnnData(X=n_features, obs=annotation)
        adata.obsm['spatial'] = n_spatial_coords
        return adata

    def setUp(self):
        # Create a mock AnnData object for testing
        self.adata = self.create_dummy_dataset(repetition=1)

    def test_spatial_interaction_invalid_analysis_method(self):
        # Invalid analysis method test
        annotation = "Phenotypes"
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
        self.assertEqual(str(cm.exception), expect_string)

    def test_spatial_interaction_invalid_ax_type(self):
        # Invalid ax type test
        annotation = "Phenotypes"
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
            "Phenotypes",
            "Neighborhood Enrichment",
            ax=ax)

        # Assert that the returned ax object is the same
        # as the input ax object
        returned_ax = returned_ax_dict['Ax']
        self.assertEqual(id(returned_ax), id(ax))

        # Verify that the provided Axes is used for plotting
        self.assertEqual(returned_ax.get_xlim(), (-0.5, 1.5))
        self.assertEqual(returned_ax.get_ylim(), (-0.5, 0.5))

        # Clean up the figure
        # plt.close(fig)

    def test_no_axes_provided(self):
        # Test no matplotlib Axes provided
        spatial_interaction(
            self.adata,
            "Phenotypes",
            "Neighborhood Enrichment")
        # Verify that a new Axes is created and used for plotting
        self.assertTrue(plt.gcf().get_axes())

    def test_returned_ax_has_right_titles(self):
        annotation = "Phenotypes"
        analysis_method = "Neighborhood Enrichment"

        # Create a blank figure
        fig, ax = plt.subplots()

        # Call the function
        returned_ax = spatial_interaction(
            self.adata,
            annotation,
            analysis_method,
            ax=ax
        )        

        # Assert that the returned ax object is not None
        self.assertIsNotNone(returned_ax)

        # For some reason on CI, the image ax is registered to plt
        # but the object was not able to accessed through
        # fig.axes nor ax.get_figure().axes
        # However, the information can be acquired with
        # plt.gcf -> plt. get current figure.
        axes_list = plt.gcf().get_axes()
    
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

        returned_ax = returned_ax_dict['Ax']

        # Assert that the returned ax object is not None
        self.assertIsNotNone(returned_ax)

        # Assert that the returned ax object is the same
        # as the input ax object
        self.assertIsInstance(returned_ax, plt.Axes)

        # For some reason on CI, the image ax is registered to plt
        # but the object was not able to accessed through
        # fig.axes nor ax.get_figure().axes
        # However, the information can be acquired with
        # plt.gcf -> plt. get current figure.
        axes_list = plt.gcf().get_axes()

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
        # Use a for loop with subtests to check each value
        for i in range(len(current_values)):
            with self.subTest(
                index=i,
                current=current_values[i],
                expected=expect_values[i]
            ):
                error_msg = f"Value at index {i} is different. " + \
                    "Got '{current_values[i]}', " + \
                    "expected '{expect_values[i]}'"
                self.assertEqual(
                    current_values[i],
                    expect_values[i],
                    error_msg)

    def test_sinlge_stratify_by(self):
        ax_dict = spatial_interaction(
            self.adata,
            "Phenotypes",
            "Neighborhood Enrichment",
            stratify_by="Analysis_Region"
        )
        unique_values = self.adata.obs["Analysis_Region"].unique()

        # Get the keys (unique cluster values) from the dictionary
        keys = list(ax_dict["Ax"].keys())

        # Assert that we have at least two keys (clusters)
        self.assertEqual(len(keys), 2)

        for value in unique_values:
            # Expect each unique value as a key in the returned dict
            self.assertIn(value, ax_dict["Ax"].keys())

            # Each should be a matplotlib axis object
            self.assertIsInstance(ax_dict["Ax"][value], plt.Axes)

    def test_list_stratify_by(self):
        ax_dict = spatial_interaction(
            self.adata,
            "Phenotypes",
            "Neighborhood Enrichment",
            stratify_by=["cluster_num"]
            )

        combined_keys = self.adata.obs[
                "cluster_num"
            ].astype(str).unique()

        ax_keys_list = list(ax_dict["Ax"].keys())

        self.assertListEqual(ax_keys_list, ['1', '2'])

        for key in combined_keys:
            # Expect each combined key as a key in the returned dict
            self.assertIn(key, ax_dict["Ax"].keys())

            # Each should be a matplotlib axis object
            self.assertIsInstance(ax_dict["Ax"][key], plt.Axes)

    def test_return_matrix_and_stratify_by_combinations(self):
        annotation = "cluster_num"
        analysis_method = "Cluster Interaction Matrix"
        stratify_options = [None, "Analysis_Region"]

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
                        self.assertIsInstance(result, dict)
                        self.assertEqual(len(result), 2)
                        # Expect two dictionaries
                        self.assertIn("Ax", result.keys())
                        self.assertIn("Matrix", result.keys())

                        if stratify_by is not None:
                            # If stratification is used, assert that
                            # the keys correspond to the unique values
                            # in the stratification column
                            unique_values = [
                                "Region_A",
                                "Region_B"
                            ]

                            for value in unique_values:
                                self.assertIsInstance(
                                        result["Ax"][value],
                                        plt.Axes
                                    )
                                self.assertIsInstance(
                                        result["Matrix"][value],
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
                            unique_values = [
                                "Region_A",
                                "Region_B"
                            ]
                            for value in unique_values:
                                self.assertIn(value, result["Ax"])
                                self.assertIsInstance(
                                        result["Ax"][value],
                                        plt.Axes
                                    )
                        else:
                            # If no stratification is used, assert
                            # that there is only one key
                            self.assertEqual(len(result), 1)
                            self.assertIn("Ax", result.keys())

    def test_stratify_function(self):

        # The values in the "startify by" column divides the
        # "stratify_test_annotation" column into exact same two
        # sets of labels, hence the result of spatial interaction on 
        # Stratify_test_annotation for dataset with one repitition(8 datapoints)
        # should be exactly the same as running Stratify_test_annotation with
        # stratify_by "Stratify_test_stratify_by" column on dataset with
        # two repitation (16 datapoints). The ground truth matrix
        # can be acquired by:
        # ground_truth = spatial_interaction(
        #     self.create_dummy_dataset(repetition=1),
        #     "Phenotypes",
        #     "Cluster Interaction Matrix",
        #     return_matrix=True
        #     )  
        # Check if stratify works.
        ax_dict = spatial_interaction(
            self.adata,
            "Phenotypes",
            "Cluster Interaction Matrix",
            stratify_by="Analysis_Region",
            return_matrix=True
            )

        # The first 7 cells are in Region_A are on a row and
        # not connected to the last 7 cells in Region_B.
        # Given the default squidpy using KNN to make the graph
        # with 6 neighbors, the ground truth matrix is:
        region_a_ground_truth = array(
            [[20., 10.],
             [10., 2.]])

        self.assertTrue(
            np.array_equal(
                ax_dict['Matrix']['Region_A'],
                region_a_ground_truth
            )
        )

        region_b_ground_truth = array(
            [[6., 12.],
             [12., 12.]])

        self.assertTrue(
            np.array_equal(
                ax_dict['Matrix']['Region_B'],
                region_b_ground_truth 
            )
        )

        # Check if stratify=None works
        ax_dict = spatial_interaction(
            self.adata,
            "Phenotypes",
            "Cluster Interaction Matrix",
            stratify_by=None,
            return_matrix=True
            )

        # As Region_A cells are not connected to Region_B cells, the
        # ground truth matrix is:
        all_regions_ground_truth = \
            region_a_ground_truth + region_b_ground_truth

        self.assertTrue(
            np.array_equal(
                ax_dict['Matrix'],
                all_regions_ground_truth
            )
        )

    def test_n_neighbor_function(self):
        # Using the previous stratify example as reference

        ax_dict = spatial_interaction(
            self.adata,
            "Phenotypes",
            "Cluster Interaction Matrix",
            stratify_by="Analysis_Region",
            return_matrix=True,
            n_neighs=2
            )

        # The first 7 cells are in Region_A are on a row and not 
        # connected to the last 7 cells in Region_B.
        # Given the squidpy using KNN to make the graph
        # with 2 neighbors, the ground truth matrix is:
        region_a_ground_truth = array(
            [[6., 4.],
             [4., 0.]])

        self.assertTrue(
            np.array_equal(
                ax_dict['Matrix']['Region_A'],
                region_a_ground_truth
            )
        )

        region_b_ground_truth = array(
            [[0., 6.],
             [6., 2.]])

        self.assertTrue(
            np.array_equal(
                ax_dict['Matrix']['Region_B'],
                region_b_ground_truth 
            )
        )

    def test_radius_function(self):
        # Using the previous stratify example as reference

        ax_dict = spatial_interaction(
            self.adata,
            "Phenotypes",
            "Cluster Interaction Matrix",
            stratify_by="Analysis_Region",
            return_matrix=True,
            radius=1
            )

        # The first 7 cells are in Region_A are on a row and not 
        # connected to the last 7 cells in Region_B.
        # Given the squidpy using KNN to make the graph
        # with radius = 1, the ground truth matrix is:
        region_a_ground_truth = array(
            [[4., 4.],
             [4., 0.]])

        self.assertTrue(
            np.array_equal(
                ax_dict['Matrix']['Region_A'],
                region_a_ground_truth
            )
        )

        region_b_ground_truth = array(
            [[0., 6.],
             [6., 0.]])

        self.assertTrue(
            np.array_equal(
                ax_dict['Matrix']['Region_B'],
                region_b_ground_truth 
            )
        )


    def tearDown(self):
        del self.adata


if __name__ == '__main__':
    unittest.main()
