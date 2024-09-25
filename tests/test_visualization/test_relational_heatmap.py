import unittest
from spac.visualization import relational_heatmap
from anndata import AnnData
import numpy as np
import pandas as pd
from plotly.graph_objs import Figure


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
        fig = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues',
            font_size=12
        )
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, Figure)

    def test_sames_source_type(self):
        # Test the function with valid inputs
        fig = relational_heatmap(
            self.adata,
            'source_annotation',
            'source_annotation',
            'Blues',
            font_size=12
        )
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, Figure)

    def test_valid_labels(self):
        # Test the function with valid inputs
        fig = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues'
        )

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
        fig = relational_heatmap(
            self.adata,
            'source_annotation',
            'target_annotation',
            'Blues'
        )

        # Check the source and target in fig.data[0].link
        self.assertCountEqual(fig.layout.annotations[0].text, "100.0")
        self.assertCountEqual(fig.layout.annotations[1].text, "0")
        self.assertCountEqual(fig.layout.annotations[2].text, "0")
        self.assertCountEqual(fig.layout.annotations[3].text, "100.0")

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


if __name__ == '__main__':
    unittest.main()
