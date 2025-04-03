import unittest
from spac.visualization import relational_heatmap
from anndata import AnnData
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from pandas.testing import assert_frame_equal



class TestRelationalHeatmap(unittest.TestCase):
    def setUp(self):
        # Create a simple AnnData object for testing
        self.adata = AnnData(
            np.random.rand(10, 10),
            obs=pd.DataFrame(
                {
                    'source_annotation': ['source1', 'source2'] * 5,
                    'target_annotation': ['target1', 'target2'] * 5,
                },
                index=[f'cell_{i}' for i in range(10)]
            )
        )

    def test_valid_type(self):
        # Test the function with valid inputs
        result_dict = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues',
            font_size=12
        )
        self.assertIsNotNone(result_dict)
        self.assertIn("figure", result_dict)
        self.assertIn("data", result_dict)
        self.assertIn("file_name", result_dict)
        self.assertIsInstance(result_dict["figure"], Figure)

    def test_sames_source_type(self):
        # Test the function with valid inputs
        result_dict = relational_heatmap(
            self.adata,
            'source_annotation',
            'source_annotation',
            'Blues',
            font_size=12
        )
        self.assertIsNotNone(result_dict)
        self.assertIn("figure", result_dict)
        self.assertIn("data", result_dict)
        self.assertIn("file_name", result_dict)
        self.assertIsInstance(result_dict["figure"], Figure)

    def test_valid_labels(self):
        # Test the function with valid inputs
        result_dict = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues'
        )

        fig = result_dict["figure"]

        self.assertCountEqual(
            fig.layout.xaxis.ticktext,
            ["target_target1", "target_target2"]
        )
        self.assertCountEqual(
            fig.layout.yaxis.ticktext,
            ["source_source1", "source_source2"]
        )

    def test_valid_heatmap_values(self):
        # Test the function with valid inputs
        result_dict = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues'
        )

        fig = result_dict["figure"]

        # Check the source and target in fig.data[0].link
        self.assertCountEqual(fig.layout.annotations[0].text, "100.0")
        self.assertCountEqual(fig.layout.annotations[1].text, "0")
        self.assertCountEqual(fig.layout.annotations[2].text, "0")
        self.assertCountEqual(fig.layout.annotations[3].text, "100.0")

    def test_valid_dataframe(self):
        # Test the function with valid inputs
        result_dict = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues'
        )

        df = result_dict["data"]

        # Define the expected DataFrame
        expected_df = pd.DataFrame({
            'target_target1': [100.0, 0],
            'target_target2': [0, 100.0],
            'total': [100.0, 100.0]
        }, index=['source_source1', 'source_source2'])

        assert_frame_equal(expected_df, df)

    def test_invalid_source_annotation(self):
        # Test the function with an invalid source_annotation
        with self.assertRaises(ValueError):
            relational_heatmap(
                self.adata,
                'invalid_source_annotation',
                'target_annotation',
                'Blues'
            )

    def test_invalid_target_annotation(self):
        # Test the function with an invalid target_annotation
        with self.assertRaises(ValueError):
            relational_heatmap(
                self.adata,
                'source_annotation',
                'invalid_target_annotation',
                'Blues'
            )

    def test_axis_labels(self):
        """
        Test that the x-axis and y-axis labels are correct in the relational heatmap.
        """
        # Generate the relational heatmap
        result_dict = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues'
        )

        # Extract the figure from the result
        fig = result_dict["figure"]

        # Check the x-axis label
        self.assertEqual(fig.layout.yaxis.title.text, "source_annotation")

        # Check the y-axis label
        self.assertEqual(fig.layout.xaxis.title.text, "target_annotation")


if __name__ == '__main__':
    unittest.main()
