import anndata
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spac.spatial_analysis import spatial_interaction


class TestSpatialInteraction(unittest.TestCase):
    def setUp(self):
        # Create a mock AnnData object for testing
        annotation = pd.DataFrame({
            "cluster_num": [1, 2, 2, 1, 1, 2, 1, 2] * 5,
            "cluster_str": [
                "A", "B", "A", "B",
                "A", "B", "A", "B"
                ] * 5,
            "labels": [
                "Un", "De", "De", "De",
                "Un", "Un", "De", "Un",
                "Un", "Un", "Un", "De",
                "De", "De", "De", "Un",
                "Un", "De", "Un", "De"
                ] * 2
        })
        features = np.array(
            [
                [0.74676753, 0.46408353, 1.57989469],
                [-0.70239818, 0.72763876, -0.97137637],
                [0.7668926, 1.28649109, 0.00439471],
                [0.1626119, -0.67745485, 0.133223],
                [-0.76999373, 0.68962473, 1.21965841],
                [0.30440098, -0.21207851, 0.06921215],
                [2.02199453, 1.26504621, 0.1908807],
                [-1.29639493, 0.24391637, 0.99816421],
                [1.78321933, 0.17752491, -1.09286752],
                [-0.30869161, 1.0884462, 0.57758739],
                [0.48271656, -0.20947287, -0.69081094],
                [-0.37602375, -0.46224323, -0.10643137],
                [1.38178975, -1.15352462, 1.33530468],
                [-0.89947323, 2.71234519, -1.2043588],
                [1.03564018, -0.39764877, 1.07528684],
                [-2.18967361, -0.8862305, 0.23347924],
                [-0.14314478, -0.51110142, -0.59105464],
                [-1.4396892, 0.98799238, 1.07133854],
                [-0.69406544, 1.53425436, 0.06986254],
                [-2.05036209, 0.0242316, -1.97612371],
                [0.67404544, -0.36911984, 0.50482678],
                [-0.6759334, -0.61206631, 1.60611651],
                [-0.13852431, -0.68823812, 0.90864488],
                [-0.40529142, -2.28500196, -0.49695203],
                [-0.82450205, 0.08434124, 0.13808722],
                [-0.0394261, 0.6574769, 0.99616492],
                [-2.53752236, 0.6023303, -0.59271762],
                [0.79551212, -0.69543986, -0.45754289],
                [0.0151175, -1.54801625, 1.2467797],
                [0.79318999, 2.11103018, 0.54927102],
                [-0.14915621, -0.38113325, -0.11962863],
                [-1.52999662, -0.99229833, -1.79421771],
                [0.41009931, -1.76932615, 0.89048498],
                [1.38687137, -0.51228094, 0.41986469],
                [0.32971825, -1.45386377, -0.60882985],
                [-0.5973023, -1.05662806, -1.44705104],
                [-0.43171187, 0.68328617, 0.15106893],
                [-0.2422716, -0.27264463, 2.87836626],
                [0.3660139, 0.33811382, 0.20605574],
                [-0.22853678, 0.89296266, -1.08729805]
            ]
        )
        spatial_coords = np.array(
            [
                [-0.50022065, -0.07619456],
                [0.80629849, 1.3394628],
                [-1.76129432, -0.05689653],
                [0.5263589, 0.44512444],
                [0.5464698, 0.33697916],
                [0.38636183, 1.02076166],
                [-0.0772483, -0.61370415],
                [-0.67429891, -2.04728207],
                [1.64487174, 2.00882995],
                [2.18092253, 0.59974797],
                [-0.57880274, 1.15883317],
                [-1.29041131, -0.62178807],
                [0.33430965, -0.24713342],
                [0.03485657, -0.1041557],
                [0.65980237, -1.23234249],
                [0.81474365, -1.22294307],
                [-0.14663293, 1.51689158],
                [0.09116057, 0.36915328],
                [-0.91581849, 1.0130385],
                [0.72576781, 0.00269482],
                [0.01994862, 1.12024613],
                [-0.13322151, -1.78447807],
                [0.26643831, -0.42013604],
                [-0.37204659, 0.13233973],
                [-0.62107268, 0.56311141],
                [0.60191124, -0.62487495],
                [-0.77451973, -0.35540268],
                [0.6807837, -0.14351254],
                [0.06994339, 0.89862981],
                [0.32838978, 0.07389066],
                [-0.4867984, -1.09153204],
                [1.41409357, -0.78526777],
                [0.83992281, 1.78054052],
                [-0.75939187, -0.72872904],
                [-0.60859419, -0.3976462],
                [0.51635987, 0.58870188],
                [0.31449339, 0.31681774],
                [-0.42982571, -0.70367001],
                [-1.48978518, -0.1277682],
                [-1.0395289, -0.50693076]
            ]
        )
        self.adata = anndata.AnnData(X=features, obs=annotation)
        self.adata.obsm['spatial'] = spatial_coords
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
                        "cluster_num\ncluster_str\nlabels"
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

    def test_List_stratify_by(self):
        ax_dict = spatial_interaction(
            self.adata,
            "cluster_str",
            "Neighborhood Enrichment",
            stratify_by=[
                "cluster_num",
                "labels"
                ]
            )
        combined_keys = self.adata.obs[
            [
                "cluster_num",
                "labels"
            ]
             ].astype(str).agg('_'.join, axis=1).unique()

        for key in combined_keys:
            # Expect each combined key as a key in the returned dict
            self.assertIn(key, ax_dict.keys())

            # Each should be a matplotlib axis object
            self.assertIsInstance(ax_dict[key], plt.Axes)

    def tearDown(self):
        del self.adata


if __name__ == '__main__':
    unittest.main()
