# tests//templates/test_template_utils.py
"""Unit tests for template utilities."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.template_utils import (
    load_input,
    save_outputs,
    text_to_value,
    convert_pickle_to_h5ad,
    convert_to_floats
)


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": ["TypeA", "TypeB"] * (n_cells // 2)
    })
    x_mat = rng.normal(size=(n_cells, 2))
    adata = ad.AnnData(X=x_mat, obs=obs)
    return adata


def mock_dataframe(n_rows: int = 5) -> pd.DataFrame:
    """Return a minimal DataFrame for fast tests."""
    return pd.DataFrame({
        "col1": range(n_rows),
        "col2": [f"value_{i}" for i in range(n_rows)]
    })


class TestTemplateUtils(unittest.TestCase):
    """Unit tests for template utility functions."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_adata = mock_adata()
        self.test_df = mock_dataframe()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering all input/output scenarios."""
        # Suppress warnings for cleaner test output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Test 1: Load h5ad file
            h5ad_path = os.path.join(self.tmp_dir.name, "test.h5ad")
            self.test_adata.write_h5ad(h5ad_path)
            loaded_h5ad = load_input(h5ad_path)
            self.assertEqual(loaded_h5ad.n_obs, 10)
            self.assertIn("cell_type", loaded_h5ad.obs.columns)

            # Test 2: Load pickle file
            pickle_path = os.path.join(self.tmp_dir.name, "test.pickle")
            with open(pickle_path, "wb") as f:
                pickle.dump(self.test_adata, f)
            loaded_pickle = load_input(pickle_path)
            self.assertEqual(loaded_pickle.n_obs, 10)

            # Test 3: Load .pkl extension
            pkl_path = os.path.join(self.tmp_dir.name, "test.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(self.test_adata, f)
            loaded_pkl = load_input(pkl_path)
            self.assertEqual(loaded_pkl.n_obs, 10)

            # Test 4: Load .p extension
            p_path = os.path.join(self.tmp_dir.name, "test.p")
            with open(p_path, "wb") as f:
                pickle.dump(self.test_adata, f)
            loaded_p = load_input(p_path)
            self.assertEqual(loaded_p.n_obs, 10)

            # Test 5: Save outputs - multiple formats
            outputs = {
                "result.pickle": self.test_adata,  # Now preferred format
                "data.csv": self.test_df,
                "adata.pkl": self.test_adata,
                "adata.h5ad": self.test_adata,  # Still supported
                "other_data": {"key": "value"}  # Defaults to pickle
            }
            saved_files = save_outputs(outputs, self.tmp_dir.name)

            # Verify all files were saved
            self.assertEqual(len(saved_files), 5)
            for filename, filepath in saved_files.items():
                self.assertTrue(os.path.exists(filepath))
                self.assertIn(filename, saved_files)

            # Verify CSV content
            csv_path = saved_files["data.csv"]
            loaded_df = pd.read_csv(csv_path)
            self.assertEqual(len(loaded_df), 5)
            self.assertIn("col1", loaded_df.columns)

            # Test 6: Convert pickle to h5ad
            pickle_src = os.path.join(
                self.tmp_dir.name, "convert_src.pickle"
            )
            with open(pickle_src, "wb") as f:
                pickle.dump(self.test_adata, f)

            h5ad_dest = convert_pickle_to_h5ad(pickle_src)
            self.assertTrue(os.path.exists(h5ad_dest))
            self.assertTrue(h5ad_dest.endswith(".h5ad"))

            # Test with custom output path
            custom_dest = os.path.join(
                self.tmp_dir.name, "custom_output.h5ad"
            )
            h5ad_custom = convert_pickle_to_h5ad(pickle_src, custom_dest)
            self.assertEqual(h5ad_custom, custom_dest)
            self.assertTrue(os.path.exists(custom_dest))

            # Test 7: Load file with no extension (content detection)
            no_ext_path = os.path.join(self.tmp_dir.name, "noextension")
            with open(no_ext_path, "wb") as f:
                pickle.dump(self.test_adata, f)
            loaded_no_ext = load_input(no_ext_path)
            self.assertEqual(loaded_no_ext.n_obs, 10)

    def test_text_to_value_conversions(self) -> None:
        """Test all text_to_value conversion scenarios."""
        # Test 1: Convert to float
        result = text_to_value("3.14", to_float=True)
        self.assertEqual(result, 3.14)
        self.assertIsInstance(result, float)

        # Test 2: Convert to int
        result = text_to_value("42", to_int=True)
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

        # Test 3: None text handling
        result = text_to_value("None", value_to_convert_to=None)
        self.assertIsNone(result)

        # Test 4: Empty string handling
        result = text_to_value("", value_to_convert_to=-1)
        self.assertEqual(result, -1)

        # Test 5: Case insensitive None
        result = text_to_value("none", value_to_convert_to=0)
        self.assertEqual(result, 0)

        # Test 6: Custom none text
        result = text_to_value(
            "NA", default_none_text="NA", value_to_convert_to=999
        )
        self.assertEqual(result, 999)

        # Test 7: No conversion
        result = text_to_value("keep_as_string")
        self.assertEqual(result, "keep_as_string")
        self.assertIsInstance(result, str)

        # Test 8: Whitespace handling
        result = text_to_value("  None  ", value_to_convert_to=None)
        self.assertIsNone(result)

        # Test 9: Non-string input
        result = text_to_value(123, to_float=True)
        self.assertEqual(result, 123.0)
        self.assertIsInstance(result, float)

    def test_convert_to_floats(self) -> None:
        """Test convert_to_floats function."""
        # Test 1: String list
        result = convert_to_floats(["1.5", "2.0", "3.14"])
        self.assertEqual(result, [1.5, 2.0, 3.14])
        self.assertTrue(all(isinstance(x, float) for x in result))

        # Test 2: Mixed numeric types
        result = convert_to_floats([1, "2.5", 3.0])
        self.assertEqual(result, [1.0, 2.5, 3.0])

        # Test 3: Invalid value
        with self.assertRaises(ValueError) as context:
            convert_to_floats(["1.0", "invalid", "3.0"])
        expected_msg = "Failed to convert value : 'invalid' to float"
        self.assertIn(expected_msg, str(context.exception))

        # Test 4: Empty list
        result = convert_to_floats([])
        self.assertEqual(result, [])

    def test_load_input_missing_file_error_message(self) -> None:
        """Test exact error message for missing input file."""
        missing_path = "/nonexistent/path/file.h5ad"

        with self.assertRaises(FileNotFoundError) as context:
            load_input(missing_path)

        expected_msg = f"Input file not found: {missing_path}"
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_load_input_unsupported_format_error_message(self) -> None:
        """Test exact error message for unsupported file format."""
        # Create a text file with unsupported content
        txt_path = os.path.join(self.tmp_dir.name, "test.txt")
        with open(txt_path, "w") as f:
            f.write("This is not a valid data file")

        with self.assertRaises(ValueError) as context:
            load_input(txt_path)

        actual_msg = str(context.exception)
        self.assertTrue(actual_msg.startswith("Unable to load file"))
        self.assertIn("Supported formats: h5ad, pickle", actual_msg)

    def test_text_to_value_float_conversion_error_message(self) -> None:
        """Test exact error message for invalid float conversion."""
        with self.assertRaises(ValueError) as context:
            text_to_value(
                "not_a_number", to_float=True, param_name="test_param"
            )

        expected_msg = (
            'Error: can\'t convert test_param to float. '
            'Received:"not_a_number"'
        )
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_text_to_value_int_conversion_error_message(self) -> None:
        """Test exact error message for invalid integer conversion."""
        with self.assertRaises(ValueError) as context:
            text_to_value("3.14", to_int=True, param_name="count")

        expected_msg = (
            'Error: can\'t convert count to integer. '
            'Received:"3.14"'
        )
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_convert_pickle_to_h5ad_missing_file_error_message(self) -> None:
        """Test exact error message for missing pickle file."""
        missing_pickle = "/nonexistent/file.pickle"

        with self.assertRaises(FileNotFoundError) as context:
            convert_pickle_to_h5ad(missing_pickle)

        expected_msg = f"Pickle file not found: {missing_pickle}"
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_convert_pickle_to_h5ad_wrong_type_error_message(self) -> None:
        """Test exact error message when pickle doesn't contain AnnData."""
        # Create pickle with wrong type
        wrong_pickle = os.path.join(self.tmp_dir.name, "wrong_type.pickle")
        with open(wrong_pickle, "wb") as f:
            pickle.dump({"not": "anndata"}, f)

        with self.assertRaises(TypeError) as context:
            convert_pickle_to_h5ad(wrong_pickle)

        expected_msg = "Loaded object is not AnnData, got <class 'dict'>"
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_default_pickle_extension(self) -> None:
        """Test that files without extension default to pickle."""
        outputs = {
            "no_extension": self.test_adata
        }
        saved_files = save_outputs(outputs, self.tmp_dir.name)

        # Should have .pickle extension
        filepath = saved_files["no_extension"]
        self.assertTrue(filepath.endswith('.pickle'))
        self.assertTrue(os.path.exists(filepath))


if __name__ == "__main__":
    unittest.main()
