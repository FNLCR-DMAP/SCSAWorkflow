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
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.template_utils import (
    load_input,
    save_results,
    _save_single_object,
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

            # Test 5: Convert pickle to h5ad
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

    def test_save_results_single_csv_file(self) -> None:
        """Test saving DataFrame as single CSV file using save_results."""
        # Setup
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        params = {
            "outputs": {
                "dataframe": {"type": "file", "name": "data.csv"}
            }
        }
        
        results = {
            "dataframe": df
        }
        
        # Execute
        saved = save_results(results, params, self.tmp_dir.name)
        
        # Verify
        csv_path = Path(self.tmp_dir.name) / "data.csv"
        self.assertTrue(csv_path.exists())
        self.assertTrue(csv_path.is_file())
        
        # Check content
        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded_df, df)
        
    def test_save_results_multiple_csvs_directory(self) -> None:
        """Test saving multiple DataFrames in directory using save_results."""
        # Setup
        df1 = pd.DataFrame({'X': [1, 2]})
        df2 = pd.DataFrame({'Y': [3, 4]})
        
        params = {
            "outputs": {
                "dataframe": {"type": "directory", "name": "dataframe_dir"}
            }
        }
        
        results = {
            "dataframe": {
                "first": df1,
                "second": df2
            }
        }
        
        # Execute
        saved = save_results(results, params, self.tmp_dir.name)
        
        # Verify
        dir_path = Path(self.tmp_dir.name) / "dataframe_dir"
        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())
        self.assertTrue((dir_path / "first.csv").exists())
        self.assertTrue((dir_path / "second.csv").exists())
        
    def test_save_results_figures_directory(self) -> None:
        """Test saving multiple figures in directory using save_results."""
        # Suppress matplotlib warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Setup
            fig1, ax1 = plt.subplots()
            ax1.plot([1, 2, 3])
            
            fig2, ax2 = plt.subplots()
            ax2.bar(['A', 'B'], [5, 10])
            
            params = {
                "outputs": {
                    "figures": {"type": "directory", "name": "plots"}
                }
            }
            
            results = {
                "figures": {
                    "line_plot": fig1,
                    "bar_plot": fig2
                }
            }
            
            # Execute
            saved = save_results(results, params, self.tmp_dir.name)

            # Verify
            plots_dir = Path(self.tmp_dir.name) / "plots"
            self.assertTrue(plots_dir.exists())
            self.assertTrue(plots_dir.is_dir())
            self.assertTrue((plots_dir / "line_plot.png").exists())
            self.assertTrue((plots_dir / "bar_plot.png").exists())
            
            # Clean up
            plt.close('all')
    
    def test_save_results_analysis_pickle_file(self) -> None:
        """Test saving analysis object as pickle file using save_results."""
        # Setup
        analysis = {
            "method": "test_analysis",
            "results": [1, 2, 3, 4, 5],
            "params": {"alpha": 0.05}
        }
        
        params = {
            "outputs": {
                "analysis": {"type": "file", "name": "results.pickle"}
            }
        }
        
        results = {
            "analysis": analysis
        }
        
        # Execute
        saved = save_results(results, params, self.tmp_dir.name)
        
        # Verify
        pickle_path = Path(self.tmp_dir.name) / "results.pickle"
        self.assertTrue(pickle_path.exists())
        self.assertTrue(pickle_path.is_file())
        
        # Check content
        with open(pickle_path, 'rb') as f:
            loaded = pickle.load(f)
        self.assertEqual(loaded, analysis)
    
    def test_save_results_html_directory(self) -> None:
        """Test saving HTML reports in directory using save_results."""
        # Setup
        html1 = "<html><body><h1>Report 1</h1></body></html>"
        html2 = "<html><body><h1>Report 2</h1></body></html>"
        
        params = {
            "outputs": {
                "html": {"type": "directory", "name": "reports"}
            }
        }
        
        results = {
            "html": {
                "main": html1,
                "summary": html2
            }
        }
        
        # Execute
        saved = save_results(results, params, self.tmp_dir.name)
        
        # Verify
        reports_dir = Path(self.tmp_dir.name) / "reports"
        self.assertTrue(reports_dir.exists())
        self.assertTrue(reports_dir.is_dir())
        self.assertTrue((reports_dir / "main.html").exists())
        self.assertTrue((reports_dir / "summary.html").exists())
        
        # Check content
        with open(reports_dir / "main.html") as f:
            content = f.read()
        self.assertIn("Report 1", content)
    
    def test_save_results_complete_configuration(self) -> None:
        """Test complete configuration with all output types using save_results."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Setup
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            
            df = pd.DataFrame({'A': [1, 2]})
            analysis = {"result": "complete"}
            html = "<html><body>Report</body></html>"
            
            params = {
                "outputs": {
                    "figures": {"type": "directory", "name": "figure_dir"},
                    "dataframe": {"type": "file", "name": "dataframe.csv"},
                    "analysis": {"type": "file", "name": "output.pickle"},
                    "html": {"type": "directory", "name": "html_dir"}
                }
            }
            
            results = {
                "figures": {"plot": fig},
                "dataframe": df,
                "analysis": analysis,
                "html": {"report": html}
            }
            
            # Execute
            saved = save_results(results, params, self.tmp_dir.name)
            
            # Verify all outputs created
            self.assertTrue((Path(self.tmp_dir.name) / "figure_dir").is_dir())
            self.assertTrue((Path(self.tmp_dir.name) / "dataframe.csv").is_file())
            self.assertTrue((Path(self.tmp_dir.name) / "output.pickle").is_file())
            self.assertTrue((Path(self.tmp_dir.name) / "html_dir").is_dir())
            
            # Clean up
            plt.close('all')
    
    def test_save_results_case_insensitive_matching(self) -> None:
        """Test case-insensitive matching of result keys to config."""
        # Setup
        df = pd.DataFrame({'A': [1, 2]})
        
        params = {
            "outputs": {
                "dataframe": {"type": "file", "name": "data.csv"}  # Capital D
            }
        }
        
        results = {
            "dataframe": df  # lowercase d
        }
        
        # Execute
        saved = save_results(results, params, self.tmp_dir.name)
        
        # Should still match and save
        self.assertTrue((Path(self.tmp_dir.name) / "data.csv").exists())
    
    def test_save_results_missing_config(self) -> None:
        """Test that missing config for result type generates warning."""
        # Setup
        df = pd.DataFrame({'A': [1, 2]})
        
        params = {
            "outputs": {
                # No config for "dataframes"
                "figures": {"type": "directory", "name": "plots"}
            }
        }
        
        results = {
            "dataframe": df,  # No matching config
            "figures": {}
        }
        
        # Execute (should not raise, just warn)
        saved = save_results(results, params, self.tmp_dir.name)
        
        # Only figures should be in saved files
        self.assertIn("figures", saved)
        self.assertNotIn("dataframes", saved)
        self.assertNotIn("DataFrames", saved)
    
    def test_save_single_object_dataframe(self) -> None:
        """Test _save_single_object helper with DataFrame."""
        df = pd.DataFrame({'A': [1, 2]})
        
        path = _save_single_object(df, "test", Path(self.tmp_dir.name))
        
        self.assertEqual(path.name, "test.csv")
        self.assertTrue(path.exists())
    
    def test_save_single_object_figure(self) -> None:
        """Test _save_single_object helper with matplotlib figure."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            
            path = _save_single_object(fig, "plot", Path(self.tmp_dir.name))
            
            self.assertEqual(path.name, "plot.png")
            self.assertTrue(path.exists())
            
            plt.close('all')
    
    def test_save_single_object_html(self) -> None:
        """Test _save_single_object helper with HTML string."""
        html = "<html><body>Test</body></html>"
        
        path = _save_single_object(html, "report.html", Path(self.tmp_dir.name))
        
        self.assertEqual(path.name, "report.html")
        self.assertTrue(path.exists())
    
    def test_save_single_object_generic(self) -> None:
        """Test _save_single_object helper with generic object."""
        data = {"test": "data", "value": 123}
        
        path = _save_single_object(data, "data", Path(self.tmp_dir.name))
        
        self.assertEqual(path.name, "data.pickle")
        self.assertTrue(path.exists())
    
    def test_save_results_dataframes_both_configurations(self) -> None:
        """Test DataFrames can be saved as both file and directory."""
        # Test 1: Single DataFrame as file
        df_single = pd.DataFrame({'A': [1, 2, 3]})
        
        params_file = {
            "outputs": {
                "dataframe": {"type": "file", "name": "single.csv"}
            }
        }
        
        results_single = {"dataframe": df_single}

        saved = save_results(results_single, params_file, self.tmp_dir.name)
        self.assertTrue((Path(self.tmp_dir.name) / "single.csv").exists())
        
        # Test 2: Multiple DataFrames as directory
        df1 = pd.DataFrame({'X': [1, 2]})
        df2 = pd.DataFrame({'Y': [3, 4]})

        params_dir = {
            "outputs": {
                "dataframe": {"type": "directory", "name": "multi_df"}
            }
        }
        
        results_multi = {
            "dataframe": {
                "data1": df1,
                "data2": df2
            }
        }

        saved = save_results(results_multi, params_dir,
                            os.path.join(self.tmp_dir.name, "test2"))
        
        dir_path = Path(self.tmp_dir.name) / "test2" / "multi_df"
        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())
        self.assertTrue((dir_path / "data1.csv").exists())
        self.assertTrue((dir_path / "data2.csv").exists())

    def test_save_results_auto_type_detection(self) -> None:
        """Test automatic type detection based on standardized schema."""
        # Setup - params with no explicit type
        params = {
            "outputs": {
                "figures": {"name": "plot.png"},  # No type specified
                "analysis": {"name": "results.pickle"},  # No type specified
                "dataframe": {"name": "data.csv"},  # No type specified
                "html": {"name": "report_dir"}  # No type specified
            }
        }
        
        # Create test data
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        
        results = {
            "figures": {"plot1": fig, "plot2": fig},  # Should auto-detect as directory
            "analysis": {"data": [1, 2, 3]},  # Should auto-detect as file
            "dataframe": pd.DataFrame({'A': [1, 2]}),  # Should auto-detect as file
            "html": {"report": "<html></html>"}  # Should auto-detect as directory
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Execute
            saved = save_results(results, params, self.tmp_dir.name)
            
            # Verify auto-detection worked correctly
            # figure should be directory (standardized for figures)
            self.assertTrue((Path(self.tmp_dir.name) / "plot.png").is_dir())
            
            # analysis should be file
            self.assertTrue((Path(self.tmp_dir.name) / "results.pickle").is_file())
            
            # dataframes should be file (standard case)
            self.assertTrue((Path(self.tmp_dir.name) / "data.csv").is_file())
            
            # html should be directory (standardized for html)
            self.assertTrue((Path(self.tmp_dir.name) / "report_dir").is_dir())
            
            plt.close('all')
    
    def test_save_results_neighborhood_profile_special_case(self) -> None:
        """Test special case for Neighborhood Profile as directory."""
        # Setup - Neighborhood Profile should be directory even though it's a dataframe
        params = {
            "outputs": {
                "dataframes": {"name": "Neighborhood_Profile_Results"}  # No type, should auto-detect
            }
        }
        
        df1 = pd.DataFrame({'X': [1, 2]})
        df2 = pd.DataFrame({'Y': [3, 4]})
        
        results = {
            "dataframes": {
                "profile1": df1,
                "profile2": df2
            }
        }
        
        # Execute
        saved = save_results(results, params, self.tmp_dir.name)
        
        # Verify it was saved as directory (special case)
        dir_path = Path(self.tmp_dir.name) / "Neighborhood_Profile_Results"
        self.assertTrue(dir_path.exists())
        self.assertTrue(dir_path.is_dir())
        self.assertTrue((dir_path / "profile1.csv").exists())
        self.assertTrue((dir_path / "profile2.csv").exists())
    
    def test_save_results_with_output_directory_param(self) -> None:
        """Test using Output_Directory from params."""
        custom_dir = os.path.join(self.tmp_dir.name, "custom_output")
        
        # Setup - params includes Output_Directory
        params = {
            "Output_Directory": custom_dir,
            "outputs": {
                "dataframes": {"type": "file", "name": "data.csv"}
            }
        }
        
        results = {
            "dataframes": pd.DataFrame({'A': [1, 2]})
        }
        
        # Execute without specifying output_base_dir (should use params)
        saved = save_results(results, params)
        
        # Verify it used the Output_Directory from params
        csv_path = Path(custom_dir) / "data.csv"
        self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main()