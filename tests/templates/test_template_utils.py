# tests/templates/test_template_utils.py
"""Unit tests for template utilities."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch
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
    convert_to_floats,
    spell_out_special_characters,
    load_csv_files
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
        expected_msg = "Failed to convert value: 'invalid' to float"
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

    def test_spell_out_special_characters(self) -> None:
        """Test spell_out_special_characters function."""
        from spac.templates.template_utils import spell_out_special_characters

        # Test space replacement
        result = spell_out_special_characters("Cell Type")
        self.assertEqual(result, "Cell_Type")

        # Test special units
        result = spell_out_special_characters("Area µm²")
        self.assertEqual(result, "Area_um2")

        # Test hyphen between letters
        result = spell_out_special_characters("CD4-positive")
        self.assertEqual(result, "CD4_positive")

        # Test plus/minus
        result = spell_out_special_characters("CD4+")
        self.assertEqual(result, "CD4_pos")  # Trailing underscore is stripped
        result = spell_out_special_characters("CD8-")
        self.assertEqual(result, "CD8_neg")  # Trailing underscore is stripped

        # Test combination markers
        result = spell_out_special_characters("CD4+CD20-")
        self.assertEqual(result, "CD4_pos_CD20_neg")
        
        # Test edge cases with special separators
        result = spell_out_special_characters("CD4+/CD20-")
        self.assertEqual(result, "CD4_pos_slashCD20_neg")
        
        result = spell_out_special_characters("CD4+ CD20-")
        self.assertEqual(result, "CD4_pos_CD20_neg")
        
        result = spell_out_special_characters("CD4+,CD20-")
        self.assertEqual(result, "CD4_pos_CD20_neg")
        
        # Test parentheses removal
        result = spell_out_special_characters("CD4+ (bright)")
        self.assertEqual(result, "CD4_pos_bright")

        # Test special characters
        result = spell_out_special_characters("Cell@100%")
        self.assertEqual(result, "Cellat100percent")

        # Test multiple underscores
        result = spell_out_special_characters("Cell___Type")
        self.assertEqual(result, "Cell_Type")

        # Test leading/trailing underscores
        result = spell_out_special_characters("_Cell_Type_")
        self.assertEqual(result, "Cell_Type")

        # Test complex case
        result = spell_out_special_characters("CD4+ T-cells (µm²)")
        self.assertEqual(result, "CD4_pos_T_cells_um2")

        # Test empty string
        result = spell_out_special_characters("")
        self.assertEqual(result, "")

        # Additional edge cases
        result = spell_out_special_characters("CD3+CD4+CD8-")
        self.assertEqual(result, "CD3_pos_CD4_pos_CD8_neg")
        
        result = spell_out_special_characters("PD-1/PD-L1")
        self.assertEqual(result, "PD_1slashPD_L1")
        
        result = spell_out_special_characters("CD45RA+CD45RO-")
        self.assertEqual(result, "CD45RA_pos_CD45RO_neg")
        
        result = spell_out_special_characters("CD4+CD25+FOXP3+")
        self.assertEqual(result, "CD4_pos_CD25_pos_FOXP3_pos")

        # Test with multiple special characters
        result = spell_out_special_characters("CD4+ & CD8+ (double positive)")
        self.assertEqual(result, "CD4_pos_and_CD8_pos_double_positive")
        
        # Test with numbers at start (should add col_ prefix in 
        # clean_column_name)
        result = spell_out_special_characters("123ABC")
        # Note: col_ prefix is added by clean_column_name
        self.assertEqual(result, "123ABC")

    def test_load_csv_files(self) -> None:
        """Test load_csv_files function."""

        # Create test CSV files
        csv_dir = Path(self.tmp_dir.name) / "csv_data"
        csv_dir.mkdir()

        # CSV 1: Normal data
        csv1 = pd.DataFrame({
            'ID': ['001', '002', '003'],
            'Value': [1.5, 2.5, 3.5],
            'Type': ['A', 'B', 'A']
        })
        csv1.to_csv(csv_dir / 'data1.csv', index=False)

        # CSV 2: Special characters in columns
        csv2 = pd.DataFrame({
            'ID': ['004', '005'],
            'Value': [4.5, 5.5],
            'Type': ['B', 'C'],
            'Area µm²': [100, 200]
        })
        csv2.to_csv(csv_dir / 'data2.csv', index=False)

        # Test 1: Basic loading with metadata
        config = pd.DataFrame({
            'file_name': ['data1.csv', 'data2.csv'],
            'experiment': ['Exp1', 'Exp2'],
            'batch': [1, 2]
        })

        result = load_csv_files(csv_dir, config)
        
        # Verify basic structure
        self.assertEqual(len(result), 5)  # 3 + 2 rows
        self.assertIn('file_name', result.columns)
        self.assertIn('experiment', result.columns)
        self.assertIn('batch', result.columns)
        self.assertIn('ID', result.columns)
        self.assertIn('Area_um2', result.columns)  # Cleaned name

        # Verify metadata mapping
        exp1_rows = result[result['file_name'] == 'data1.csv']
        self.assertTrue(all(exp1_rows['experiment'] == 'Exp1'))
        self.assertTrue(all(exp1_rows['batch'] == 1))

        # Test 2: String columns preservation
        result_str = load_csv_files(
            csv_dir, config, string_columns=['ID']
        )
        self.assertEqual(result_str['ID'].dtype, 'object')
        self.assertTrue(all(isinstance(x, str) for x in result_str['ID']))

        # Test 3: Empty string_columns list
        result_empty = load_csv_files(csv_dir, config, string_columns=[])
        self.assertIsInstance(result_empty, pd.DataFrame)

        # Test 4: Column name with spaces in config
        config_spaces = pd.DataFrame({
            'file_name': ['data1.csv'],
            'Sample Type': ['Control']  # Space in column name
        })
        with self.assertRaises(ValueError):
            # Should fail validation due to string_columns not being list
            load_csv_files(csv_dir, config_spaces, string_columns="ID")

        # Test 5: Missing file in config
        config_missing = pd.DataFrame({
            'file_name': ['missing.csv'],
            'experiment': ['Exp3']
        })
        with self.assertRaises(TypeError) as context:
            load_csv_files(csv_dir, config_missing)
        self.assertIn("not found", str(context.exception))

        # Test 6: Empty CSV file
        empty_csv = csv_dir / 'empty.csv'
        empty_csv.write_text('')
        config_empty = pd.DataFrame({
            'file_name': ['empty.csv'],
            'experiment': ['Exp4']
        })
        with self.assertRaises(TypeError) as context:
            load_csv_files(csv_dir, config_empty)
        self.assertIn("empty", str(context.exception))

        # Test 7: First file validation for string_columns
        config_single = pd.DataFrame({
            'file_name': ['data1.csv']
        })
        with self.assertRaises(ValueError):
            # Non-existent column should raise error
            load_csv_files(
                csv_dir, config_single, 
                string_columns=['NonExistentColumn']
            )

    @patch('builtins.print')
    def test_load_csv_files_console_output(self, mock_print) -> None:
        """Test console output from load_csv_files."""
        from spac.templates.template_utils import load_csv_files

        # Setup test data
        csv_dir = Path(self.tmp_dir.name) / "csv_test"
        csv_dir.mkdir()

        csv_data = pd.DataFrame({
            'ID': [1, 2],
            'CD4+': ['pos', 'neg']  # Special character
        })
        csv_data.to_csv(csv_dir / 'test.csv', index=False)

        config = pd.DataFrame({
            'file_name': ['test.csv'],
            'group': ['A']
        })

        # Run function
        load_csv_files(csv_dir, config)

        # Check console output
        print_calls = [str(call[0][0]) for call in mock_print.call_args_list
                      if call[0]]

        # Should print column name updates
        updates = [msg for msg in print_calls 
                  if 'Column Name Updated:' in msg]
        self.assertTrue(len(updates) > 0)
        # The function strips trailing underscores, so CD4+ becomes CD4_pos
        self.assertTrue(any('CD4+' in msg and 'CD4_pos' in msg 
                           for msg in updates))

        # Should print processing messages
        processing = [msg for msg in print_calls 
                     if 'Processing file:' in msg]
        self.assertTrue(len(processing) > 0)

        # Should print final info
        final_info = [msg for msg in print_calls 
                     if 'Final Dataframe Info' in msg]
        self.assertTrue(len(final_info) > 0)

if __name__ == "__main__":
    unittest.main()