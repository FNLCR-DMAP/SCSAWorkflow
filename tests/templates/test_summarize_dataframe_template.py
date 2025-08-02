# tests/templates/test_summarize_dataframe_template.py
"""Unit tests for the Summarize DataFrame template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.summarize_dataframe_template import run_from_json


def mock_dataframe(n_rows: int = 10) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame for fast tests."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "file_name": [f"file_{i}.csv" for i in range(n_rows)],
        "CD3D": rng.random(n_rows) * 100,
        "FOXP3": rng.random(n_rows) * 50,
        "PDL1": rng.random(n_rows) * 75,
    })
    # Add some NaN values for testing
    df.loc[2, "CD3D"] = np.nan
    df.loc[5, "FOXP3"] = np.nan
    return df


class TestSummarizeDataFrameTemplate(unittest.TestCase):
    """Unit tests for the Summarize DataFrame template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.csv"
        )
        self.out_file = "summary"

        # Save minimal mock data
        mock_dataframe().to_csv(self.in_file, index=False)

        # Minimal parameters - from the example
        self.params = {
            "Calculate_Centroids": self.in_file,
            "Columns": ["file_name", "CD3D", "FOXP3", "PDL1"],
            "Print_Missing_Location": True,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            with patch('spac.templates.summarize_dataframe_template.'
                      'summarize_dataframe') as mock_summarize:
                with patch('spac.templates.summarize_dataframe_template.'
                          'present_summary_as_figure') as mock_fig:
                    # Mock the summary dataframe
                    mock_summarize.return_value = pd.DataFrame(
                        {"test": [1, 2, 3]}
                    )
                    
                    # Mock the Plotly figure
                    mock_plotly_fig = MagicMock()
                    mock_plotly_fig.write_html = MagicMock()
                    mock_plotly_fig.show = MagicMock()
                    mock_fig.return_value = mock_plotly_fig
                    
                    result = run_from_json(self.params)
                    self.assertIsInstance(result, dict)
                    self.assertIn("summary.html", result)
                    
                    # Verify figure methods were called
                    mock_plotly_fig.show.assert_called_once()
                    mock_plotly_fig.write_html.assert_called_once_with(
                        "summary.html"
                    )
            
            # Test 2: Run without saving
            with patch('spac.templates.summarize_dataframe_template.'
                      'summarize_dataframe') as mock_summarize:
                with patch('spac.templates.summarize_dataframe_template.'
                          'present_summary_as_figure') as mock_fig:
                    mock_summarize.return_value = pd.DataFrame(
                        {"test": [1, 2, 3]}
                    )
                    mock_plotly_fig = MagicMock()
                    mock_fig.return_value = mock_plotly_fig
                    
                    result_no_save = run_from_json(
                        self.params, save_results=False
                    )
                    # Should return tuple of (figure, dataframe)
                    self.assertIsInstance(result_no_save, tuple)
                    self.assertEqual(len(result_no_save), 2)
                    fig, summary = result_no_save
                    self.assertEqual(fig, mock_plotly_fig)
                    self.assertIsInstance(summary, pd.DataFrame)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            with patch('spac.templates.summarize_dataframe_template.'
                      'summarize_dataframe') as mock_summarize:
                with patch('spac.templates.summarize_dataframe_template.'
                          'present_summary_as_figure') as mock_fig:
                    mock_summarize.return_value = pd.DataFrame(
                        {"test": [1, 2, 3]}
                    )
                    mock_plotly_fig = MagicMock()
                    mock_plotly_fig.write_html = MagicMock()
                    mock_plotly_fig.show = MagicMock()
                    mock_fig.return_value = mock_plotly_fig
                    
                    result_json = run_from_json(json_path)
                    self.assertIsInstance(result_json, dict)

    @patch('spac.templates.summarize_dataframe_template.'
           'summarize_dataframe')
    @patch('spac.templates.summarize_dataframe_template.'
           'present_summary_as_figure')
    def test_function_calls(self, mock_present, mock_summarize) -> None:
        """Test that main functions are called with correct parameters."""
        # Mock the functions
        mock_summary = pd.DataFrame({"test": [1, 2, 3]})
        mock_summarize.return_value = mock_summary
        
        mock_fig = MagicMock()
        mock_fig.write_html = MagicMock()
        mock_fig.show = MagicMock()
        mock_present.return_value = mock_fig
        
        run_from_json(self.params)
        
        # Verify summarize_dataframe was called correctly
        mock_summarize.assert_called_once()
        call_args = mock_summarize.call_args
        
        # Check the dataframe was passed
        df_arg = call_args[0][0]
        self.assertIsInstance(df_arg, pd.DataFrame)
        self.assertEqual(len(df_arg), 10)
        
        # Check keyword arguments
        self.assertEqual(
            call_args[1]['columns'],
            ["file_name", "CD3D", "FOXP3", "PDL1"]
        )
        self.assertEqual(call_args[1]['print_nan_locations'], True)
        
        # Verify present_summary_as_figure was called
        mock_present.assert_called_once_with(mock_summary)

    def test_pickle_input(self) -> None:
        """Test loading from pickle file."""
        pickle_file = os.path.join(self.tmp_dir.name, "input.pickle")
        with open(pickle_file, 'wb') as f:
            pickle.dump(mock_dataframe(), f)
        
        params_pickle = self.params.copy()
        params_pickle["Calculate_Centroids"] = pickle_file
        
        with patch('spac.templates.summarize_dataframe_template.'
                  'summarize_dataframe') as mock_summarize:
            with patch('spac.templates.summarize_dataframe_template.'
                      'present_summary_as_figure') as mock_fig:
                mock_summarize.return_value = pd.DataFrame(
                    {"test": [1, 2, 3]}
                )
                mock_plotly_fig = MagicMock()
                mock_plotly_fig.write_html = MagicMock()
                mock_plotly_fig.show = MagicMock()
                mock_fig.return_value = mock_plotly_fig
                
                result = run_from_json(params_pickle)
                self.assertIsInstance(result, dict)

    def test_optional_parameter_defaults(self) -> None:
        """Test that optional parameters use correct defaults."""
        params_minimal = {
            "Calculate_Centroids": self.in_file,
            "Columns": ["file_name", "CD3D"]
            # Print_Missing_Location not specified
        }
        
        with patch('spac.templates.summarize_dataframe_template.'
                  'summarize_dataframe') as mock_summarize:
            with patch('spac.templates.summarize_dataframe_template.'
                      'present_summary_as_figure') as mock_fig:
                mock_summarize.return_value = pd.DataFrame(
                    {"test": [1, 2, 3]}
                )
                mock_plotly_fig = MagicMock()
                mock_plotly_fig.show = MagicMock()
                mock_fig.return_value = mock_plotly_fig
                
                run_from_json(params_minimal, save_results=False)
                
                # Check default value for Print_Missing_Location
                mock_summarize.assert_called_once()
                call_args = mock_summarize.call_args
                self.assertEqual(
                    call_args[1]['print_nan_locations'],
                    False  # Default from template JSON
                )

    @patch('builtins.print')
    @patch('spac.templates.summarize_dataframe_template.'
           'summarize_dataframe')
    def test_console_output(self, mock_summarize, mock_print) -> None:
        """Test that NaN locations are printed when requested."""
        # Create summary with NaN info
        mock_summary = pd.DataFrame({
            'column': ['CD3D', 'FOXP3'],
            'missing_indices': [[2], [5]]
        })
        mock_summarize.return_value = mock_summary
        
        with patch('spac.templates.summarize_dataframe_template.'
                  'present_summary_as_figure') as mock_fig:
            mock_plotly_fig = MagicMock()
            mock_plotly_fig.show = MagicMock()
            mock_fig.return_value = mock_plotly_fig
            
            run_from_json(self.params, save_results=False)
        
        # The summarize_dataframe function itself handles printing
        # We just verify it was called with print_nan_locations=True
        mock_summarize.assert_called_once()
        self.assertTrue(
            mock_summarize.call_args[1]['print_nan_locations']
        )


if __name__ == "__main__":
    unittest.main()