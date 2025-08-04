# tests/templates/test_binary_to_categorical_annotation_template.py
"""Unit tests for the Binary to Categorical Annotation template."""

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

from spac.templates.binary_to_categorical_annotation_template import (
    run_from_json
)


def mock_dataframe(n_cells: int = 10) -> pd.DataFrame:
    """Return a minimal synthetic DataFrame for fast tests."""
    # Create binary annotation columns with mutually exclusive values
    data = {
        "cell_id": [f"cell_{i}" for i in range(n_cells)],
        "Normal_Cells": [0] * n_cells,
        "Cancer_Cells": [0] * n_cells,
        "Immuno_Cells": [0] * n_cells,
        "x_centroid": [100.0 + i for i in range(n_cells)],
        "y_centroid": [200.0 + i for i in range(n_cells)]
    }
    
    # Make mutually exclusive binary annotations
    for i in range(n_cells):
        if i % 3 == 0:
            data["Normal_Cells"][i] = 1
        elif i % 3 == 1:
            data["Cancer_Cells"][i] = 1
        else:
            data["Immuno_Cells"][i] = 1
    
    return pd.DataFrame(data)


class TestBinaryToCategoricalAnnotationTemplate(unittest.TestCase):
    """Unit tests for the Binary to Categorical Annotation template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.csv"
        )
        self.out_file = "converted_annotations.csv"

        # Save minimal mock data
        mock_df = mock_dataframe()
        mock_df.to_csv(self.in_file, index=False)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Dataset": self.in_file,
            "Binary_Annotation_Columns": [
                "Normal_Cells",
                "Cancer_Cells", 
                "Immuno_Cells"
            ],
            "New_Annotation_Name": "cell_labels",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters (CSV input)
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertIn(self.out_file, result)
            
            # Verify the output file exists and has expected content
            output_path = result[self.out_file]
            self.assertTrue(os.path.exists(output_path))
            
            # Load and check the converted dataframe
            converted_df = pd.read_csv(output_path)
            self.assertIn("cell_labels", converted_df.columns)
            # Check that categorical values were created
            unique_labels = set(converted_df["cell_labels"])
            expected_labels = {"Normal_Cells", "Cancer_Cells", "Immuno_Cells"}
            self.assertEqual(unique_labels, expected_labels)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            self.assertIsInstance(result_no_save, pd.DataFrame)
            self.assertIn("cell_labels", result_no_save.columns)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)
            
            # Test 4: Pickle file input
            pickle_file = os.path.join(self.tmp_dir.name, "input.pkl")
            mock_df = mock_dataframe()
            with open(pickle_file, 'wb') as f:
                pickle.dump(mock_df, f)
            
            params_pickle = self.params.copy()
            params_pickle["Upstream_Dataset"] = pickle_file
            result_pickle = run_from_json(params_pickle, save_results=False)
            self.assertIsInstance(result_pickle, pd.DataFrame)
            self.assertIn("cell_labels", result_pickle.columns)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test 1: Invalid column name with special characters
        params_bad = self.params.copy()
        params_bad["New_Annotation_Name"] = "cell@labels!"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check that check_column_name was triggered
        self.assertIn("New Annotation Name", str(context.exception))
        
        # Test 2: Unsupported file format
        params_bad_format = self.params.copy()
        params_bad_format["Upstream_Dataset"] = "input.txt"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad_format)
        
        expected_msg = (
            "Unsupported file format: .txt. "
            "Supported formats: .csv, .pickle, .pkl, .p"
        )
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.binary_to_categorical_annotation_template.bin2cat')
    def test_function_calls(self, mock_bin2cat) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the bin2cat function to return a dataframe
        mock_result_df = mock_dataframe()
        mock_result_df["cell_labels"] = ["Normal_Cells"] * len(mock_result_df)
        mock_bin2cat.return_value = mock_result_df
        
        run_from_json(self.params, save_results=False)
        
        # Verify function was called correctly
        mock_bin2cat.assert_called_once()
        call_args = mock_bin2cat.call_args
        
        # Check the arguments
        self.assertIsInstance(call_args[1]['data'], pd.DataFrame)
        self.assertEqual(
            call_args[1]['one_hot_annotations'],
            ["Normal_Cells", "Cancer_Cells", "Immuno_Cells"]
        )
        self.assertEqual(call_args[1]['new_annotation'], "cell_labels")


if __name__ == "__main__":
    unittest.main()