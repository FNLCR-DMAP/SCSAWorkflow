import unittest
import pandas as pd
import plotly.graph_objects as go

from spac.data_utils import summarize_dataframe
from spac.visualization import present_summary_as_figure


class TestIntegrationSummaryToFigure(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with one numeric and one categorical 
        # column.
        self.df = pd.DataFrame({
            'num_col': [5, 15, None, 25, 35],
            'cat_col': ['X', 'Y', 'X', 'Z', 'Y']
        })

    def test_full_figure_workflow(self):
        # Generate the summary from the DataFrame.
        summary = summarize_dataframe(self.df,
                                      columns=['num_col', 'cat_col'],
                                      print_nan_locations=False)
        # Generate the Plotly figure from the summary.
        fig = present_summary_as_figure(summary)
        
        # Verify that the returned figure is a Plotly Figure.
        self.assertIsInstance(fig, go.Figure)
        # Check that a Table trace exists.
        table_traces = [trace for trace in fig.data if trace.type == "table"]
        self.assertGreater(len(table_traces), 0)
        # Validate that the table header contains expected values.
        header_vals = table_traces[0].header.values
        expected_headers = ["Column", "Data Type", "Missing Count",
                            "Missing Indices", "Summary"]
        for header in expected_headers:
            self.assertIn(header, header_vals)

        fig.write_image("test_figure.png")           


if __name__ == '__main__':
    unittest.main()
