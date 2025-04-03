import unittest
import pandas as pd

from spac.data_utils import summarize_dataframe
from spac.visualization import present_summary_as_html


class TestIntegrationSummaryToHtml(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame with a numeric and a categorical column,
        # including missing values.
        self.df = pd.DataFrame({
            'num_col': [10, 20, None, 40, 50],
            'cat_col': ['A', 'B', 'A', None, 'B']
        })

    def test_full_html_workflow(self):
        # Generate the summary dictionary from the DataFrame.
        summary = summarize_dataframe(self.df,
                                      columns=['num_col', 'cat_col'],
                                      print_nan_locations=False)
        # Generate HTML from the summary.
        html_output = present_summary_as_html(summary)
        
        # Check that the HTML string contains key elements.
        self.assertIsInstance(html_output, str)
        self.assertIn("<html>", html_output)
        self.assertIn("Data Summary", html_output)
        self.assertIn("Column: num_col", html_output)
        self.assertIn("Column: cat_col", html_output)

        # Save the HTML output to a file.
        with open('summary.html', 'w') as f:
            f.write(html_output)


if __name__ == '__main__':
    unittest.main()
