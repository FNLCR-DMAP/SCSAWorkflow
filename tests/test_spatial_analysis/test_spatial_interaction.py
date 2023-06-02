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
            "cell_id": ["cell1", "cell2",
                        "cell3", "cell4",
                        "cell5", "cell6",
                        "cell7", "cell8"],
            "feature": [1, 2, 1, 2, 1, 2, 1, 2],
            "cluster": ["A", "B", "A", "B",
                        "A", "B", "A", "B"],
            "x": [0, 1, 0, 1, 0, 1, 0, 1],
            "y": [0, 0, 1, 1, 0, 1, 0, 1]
        })
        X = np.array(
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
        self.adata = anndata.AnnData(X=X, obs=obs)
        self.adata.obsm['spatial'] = spatial_coords

    def tearDown(self):
        del self.adata

    def test_spatial_interaction_invalid_data_type(self):
        # Invalid data type test
        invalid_data = "not an AnnData object"
        feature = "valid_feature"
        analysis_method = "Neighborhood Enrichment"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                invalid_data,
                feature,
                analysis_method
            )

            self.assertIsInstance(cm.exception, ValueError)
            self.assertEqual(
                str(cm.exception),
                "Input data is not an AnnData object. Got <class 'str'>"
                )

    def test_spatial_interaction_feature_not_found(self):
        # Feature not found test
        adata = anndata.AnnData(pd.DataFrame({"existing_feature": [1, 2, 3]}))
        feature = "nonexistent_feature"
        analysis_method = "Cluster Interaction Matrix"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                adata,
                feature,
                analysis_method
            )

            expect_string = "Feature nonexistent_feature not " + \
                "found in the dataset. Existing columns are: existing_feature"
            self.assertIsInstance(cm.exception, ValueError)
            self.assertEqual(
                str(cm.exception),
                expect_string
            )

    def test_spatial_interaction_invalid_analysis_method(self):
        # Invalid analysis method test
        adata = anndata.AnnData(pd.DataFrame({"valid_feature": [1, 2, 3]}))
        feature = "valid_feature"
        invalid_analysis_method = "Invalid Method"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                adata,
                feature,
                invalid_analysis_method
            )

            expect_string = "Method Invalid Method is not supported " + \
                "currently. Available methods are: " + \
                "Neighborhood_Enrichment,Cluster_Interaction_Matrix"
            self.assertIsInstance(cm.exception, ValueError)
            self.assertEqual(
                str(cm.exception),
                expect_string
            )

    def test_spatial_interaction_invalid_ax_type(self):
        # Invalid ax type test
        adata = anndata.AnnData(pd.DataFrame({"valid_feature": [1, 2, 3]}))
        feature = "valid_feature"
        analysis_method = "Neighborhood Enrichment"
        invalid_ax = "not an Axes object"

        with self.assertRaises(ValueError) as cm:
            spatial_interaction(
                adata,
                feature,
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

    def test_invalid_input(self):
        # Test case: Invalid input data type
        invalid_data = anndata.AnnData()
        with self.assertRaises(ValueError):
            spatial_interaction(invalid_data,
                                'feature1',
                                'Neighborhood Enrichment')

        # Test case: Invalid feature not present in dataset
        with self.assertRaises(ValueError):
            spatial_interaction(self.adata,
                                'feature3',
                                'Neighborhood Enrichment')

        # Test case: Invalid analysis method
        with self.assertRaises(ValueError):
            spatial_interaction(self.adata,
                                'feature1',
                                'Invalid Method')

        # Test case: Invalid 'ax' argument type
        with self.assertRaises(ValueError):
            spatial_interaction(self.adata,
                                'feature1',
                                'Neighborhood Enrichment',
                                ax='invalid_ax')

    def test_neighborhood_enrichment_analysis(self):
        # Test Neighborhood Enrichment analysis
        ax = plt.gca()
        spatial_interaction(
            self.adata,
            "feature",
            "Neighborhood Enrichment",
            ax=ax)

        # Verify that Neighborhood Enrichment analysis is performed and plotted
        # Assertion 1: Check if Neighborhood Enrichment analysis is performed
        self.assertTrue("feature_plot" in self.adata.obs)

        # Assertion 2: Check if the resulting plot is displayed
        self.assertTrue(plt.gcf().get_axes())

    def test_cluster_interaction_matrix_analysis(self):
        # Test Cluster Interaction Matrix analysis
        ax = plt.gca()
        spatial_interaction(
            self.adata,
            "cluster",
            "Cluster Interaction Matrix",
            ax=ax)

        # Verify that Cluster Interaction Matrix
        # analysis is performed and plotted
        # Assertion 1: Check if Cluster Interaction
        # Matrix analysis is performed
        self.assertTrue("cluster_plot" in self.adata.obs)

        # Assertion 2: Check if the resulting plot is displayed
        self.assertTrue(plt.gcf().get_axes())

    def test_invalid_anndata_object(self):
        # Test invalid AnnData object
        invalid_adata = pd.DataFrame({"feature": [1, 2, 3]})
        with self.assertRaises(ValueError):
            spatial_interaction(
                invalid_adata,
                "feature",
                "Neighborhood Enrichment")

    def test_feature_not_found(self):
        # Test feature not found in the dataset
        with self.assertRaises(ValueError):
            spatial_interaction(
                self.adata,
                "invalid_feature",
                "Neighborhood Enrichment")

    def test_invalid_analysis_method(self):
        # Test invalid analysis method
        with self.assertRaises(ValueError):
            spatial_interaction(self.adata, "feature", "Invalid Method")

    def test_custom_axes_provided(self):
        # Test custom matplotlib Axes provided
        fig, ax = plt.subplots()
        # Set the desired x-axis limits
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        spatial_interaction(
            self.adata,
            "feature",
            "Neighborhood Enrichment",
            ax=ax)
        # Verify that the provided Axes is used for plotting
        self.assertEqual(ax.get_xlim(), (-0.5, 1.5))
        self.assertEqual(ax.get_ylim(), (-0.5, 0.5))

        # Clean up the figure
        plt.close(fig)

    def test_no_axes_provided(self):
        # Test no matplotlib Axes provided
        spatial_interaction(
            self.adata,
            "feature",
            "Neighborhood Enrichment")
        # Verify that a new Axes is created and used for plotting
        self.assertTrue(plt.gcf().get_axes())

    def test_additional_kwargs(self):
        # Test additional keyword arguments for matplotlib.pyplot.text()
        kwargs = {"color": "red"}
        spatial_interaction(
            self.adata,
            "feature",
            "Neighborhood Enrichment",
            **kwargs)
        # Verify that the additional keyword arguments are
        # passed to matplotlib.pyplot.text()
        # Assertion goes here

    def test_anndata_x_attribute(self):
        # Test presence of the X attribute in AnnData object
        self.assertTrue(hasattr(self.adata, "X"))


if __name__ == '__main__':
    unittest.main()
