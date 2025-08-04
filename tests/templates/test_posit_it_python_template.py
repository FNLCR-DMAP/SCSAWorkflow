# tests/templates/test_posit_it_python_template.py
"""Unit tests for the Posit-It-Python template."""

import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import matplotlib
matplotlib.use("Agg")  # Headless backend for CI
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.posit_it_python_template import run_from_json


class TestPostItPythonTemplate(unittest.TestCase):
    """Unit tests for the Posit-It-Python template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_file = "graphicsFile.png"

        # Minimal parameters from NIDAP template
        self.params = {
            "Label": "Post-It",
            "Label_font_size": "80",
            "Label_font_type": "normal",
            "Label_Bold": "False",
            "Label_font_color": "Black",
            "Label_font_family": "Arial",
            "Background_fill_color": "Yellow1",
            "Background_fill_opacity": "10",
            "Page_width": "18",
            "Page_height": "6",
            "Page_DPI": "300",
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()
        # Clean up any matplotlib figures
        plt.close('all')

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            self.assertIn(self.out_file, result)
            # Check file was created
            self.assertTrue(os.path.exists(result[self.out_file]))
            # Clean up
            os.remove(result[self.out_file])
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type - should be a figure
            self.assertIsInstance(result_no_save, plt.Figure)
            plt.close(result_no_save)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)
            # Clean up
            if os.path.exists(result_json[self.out_file]):
                os.remove(result_json[self.out_file])

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid integer conversion for font size
        params_bad = self.params.copy()
        params_bad["Label_font_size"] = "invalid_number"
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check exact error message
        expected_msg = (
            "Error: can't convert Label_font_size to integer. "
            "Received:\"invalid_number\""
        )
        self.assertEqual(str(context.exception), expected_msg)

    def test_function_calls(self) -> None:
        """Test that function is called with correct parameters."""
        # Test with custom text and colors
        params_custom = self.params.copy()
        params_custom["Label"] = "Test Label"
        params_custom["Label_font_color"] = "Red1"
        params_custom["Background_fill_color"] = "Blue1"
        params_custom["Label_Bold"] = "True"
        
        result = run_from_json(params_custom, save_results=False)
        
        # Verify figure was created with correct properties
        self.assertIsInstance(result, plt.Figure)
        # Check figure size
        self.assertEqual(result.get_figwidth(), 18.0)
        self.assertEqual(result.get_figheight(), 6.0)
        
        plt.close(result)

    def test_minimal_params(self) -> None:
        """Test with minimal parameters using defaults."""
        minimal_params = {}  # All defaults from JSON
        
        result = run_from_json(minimal_params, save_results=False)
        
        # Should still create a valid figure
        self.assertIsInstance(result, plt.Figure)
        plt.close(result)


if __name__ == "__main__":
    unittest.main()