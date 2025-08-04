# tests/templates/test_append_annotation_template.py
"""Unit tests for the Append Annotation template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock

import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.append_annotation_template import run_from_json


def mock_dataframe(n_rows: int = 10) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame for fast tests."""
    data = {
        "cell_id": list(range(n_rows)),
        "intensity": [i * 0.5 for i in range(n_rows)],
        "existing_annotation": (["TypeA", "TypeB"] * 
                               ((n_rows + 1) // 2))[:n_rows]
    }
    return pd.DataFrame(data)


class TestAppendAnnotationTemplate(unittest.TestCase):
    """Unit tests for the Append Annotation template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.csv"
        )
        self.out_file = "append_observations.csv"

        # Save minimal mock data as CSV
        mock_df = mock_dataframe()
        mock_df.to_csv(self.in_file, index=False)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Dataset": self.in_file,
            "Annotation_Pair_List": ["region:region-A", "day:day1"],
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with CSV input and save results
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertIn(self.out_file, result)
            
            # Verify the output file exists
            output_path = result[self.out_file]
            self.assertTrue(os.path.exists(output_path))
            
            # Load and verify content
            output_df = pd.read_csv(output_path)
            self.assertIn("region", output_df.columns)
            self.assertIn("day", output_df.columns)
            self.assertTrue(all(output_df["region"] == "region-A"))
            self.assertTrue(all(output_df["day"] == "day1"))
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type
            self.assertIsInstance(result_no_save, pd.DataFrame)
            self.assertIn("region", result_no_save.columns)
            self.assertIn("day", result_no_save.columns)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid annotation pair format (missing colon)
        params_bad = self.params.copy()
        params_bad["Annotation_Pair_List"] = ["invalidformat"]
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad, save_results=False)
        
        # The error will come from the split operation
        self.assertIn("not enough values to unpack", str(context.exception))

    @patch('spac.templates.append_annotation_template.append_annotation')
    @patch('spac.templates.append_annotation_template.check_column_name')
    def test_function_calls(self, mock_check, mock_append) -> None:
        """Test that main functions are called with correct parameters."""
        # Mock the append_annotation function to return a dataframe
        # with the expected new columns
        output_df = mock_dataframe()
        output_df["region"] = "region-A"
        output_df["day"] = "day1"
        mock_append.return_value = output_df
        
        result = run_from_json(self.params, save_results=False)
        
        # Verify check_column_name was called for each pair
        self.assertEqual(mock_check.call_count, 2)
        mock_check.assert_any_call("region", "region:region-A")
        mock_check.assert_any_call("day", "day:day1")
        
        # Verify append_annotation was called correctly
        mock_append.assert_called_once()
        call_args = mock_append.call_args
        expected_dict = {"region": "region-A", "day": "day1"}
        self.assertEqual(call_args[0][1], expected_dict)
        
        # Verify result is correct
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("region", result.columns)
        self.assertIn("day", result.columns)

    def test_pickle_input(self) -> None:
        """Test that pickle input files work correctly."""
        # Create pickle input file
        pickle_file = os.path.join(self.tmp_dir.name, "input.pkl")
        mock_df = mock_dataframe()
        with open(pickle_file, 'wb') as f:
            pickle.dump(mock_df, f)
        
        params_pickle = self.params.copy()
        params_pickle["Upstream_Dataset"] = pickle_file
        
        result = run_from_json(params_pickle, save_results=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("region", result.columns)
        self.assertIn("day", result.columns)
        self.assertEqual(len(result), 10)

    def test_dataframe_input(self) -> None:
        """Test that direct DataFrame input works correctly."""
        # Pass DataFrame directly
        params_df = self.params.copy()
        params_df["Upstream_Dataset"] = mock_dataframe()
        
        result = run_from_json(params_df, save_results=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("region", result.columns)
        self.assertIn("day", result.columns)
        self.assertEqual(len(result), 10)

    def test_unsupported_file_format(self) -> None:
        """Test error for unsupported file formats."""
        # Create a file with unsupported extension
        bad_file = os.path.join(self.tmp_dir.name, "input.txt")
        with open(bad_file, 'w') as f:
            f.write("dummy content")
        
        params_bad = self.params.copy()
        params_bad["Upstream_Dataset"] = bad_file
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        self.assertIn("Unsupported file format: .txt", str(context.exception))
        self.assertIn("Supported formats: .csv, .pickle, .pkl", 
                     str(context.exception))

    def test_invalid_input_type(self) -> None:
        """Test error for invalid input types."""
        params_bad = self.params.copy()
        params_bad["Upstream_Dataset"] = 12345  # Invalid type
        
        with self.assertRaises(TypeError) as context:
            run_from_json(params_bad)
        
        self.assertIn("Upstream_Dataset must be DataFrame or file path", 
                     str(context.exception))
        self.assertIn("Got <class 'int'>", str(context.exception))

    @patch('builtins.print')
    def test_console_output(self, mock_print) -> None:
        """Test that expected console output is produced."""
        run_from_json(self.params, save_results=False)
        
        # Collect all printed output as strings
        print_output = []
        for call in mock_print.call_args_list:
            if call[0]:  # If there are positional arguments
                print_output.append(str(call[0][0]))
        
        # Join all output for easier searching
        all_output = '\n'.join(print_output)
        
        # Check for key expected outputs
        # Should print the annotation pairs dictionary
        self.assertIn("region", all_output)
        self.assertIn("region-A", all_output)
        self.assertIn("day", all_output)
        self.assertIn("day1", all_output)
        
        # Should show DataFrame info or similar output
        self.assertTrue(
            "DataFrame" in all_output or 
            "Returning DataFrame" in all_output or
            "columns" in all_output
        )


if __name__ == "__main__":
    unittest.main()