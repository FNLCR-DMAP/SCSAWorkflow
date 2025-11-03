"""
Unit tests for refactored template_utils functions.
Focus on testing the new save_results function and related utilities.
"""
import pytest
import tempfile
import shutil
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))
from template_utils_refactored import (
    save_results,
    save_outputs,  # Legacy function
    load_input,
    parse_params,
    parse_list_parameter,
    text_to_value,
)


class TestSaveResults:
    """Test the new generalized save_results function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_template_utils_"))
        
    def teardown_method(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        plt.close('all')
    
    def test_save_results_with_folders(self):
        """Test saving results to organized folders."""
        # Create test data
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        
        # Blueprint configuration
        outputs_config = {
            "DataFrames": "dataframe_folder",
            "figures": "figure_folder"
        }
        
        # Results to save
        results = {
            "dataframes": {"test_data": df},
            "figures": {"test_plot": fig}
        }
        
        # Save results
        saved_files = save_results(
            results=results,
            outputs_config=outputs_config,
            output_base_dir=self.test_dir
        )
        
        # Verify structure
        assert "DataFrames" in saved_files
        assert "figures" in saved_files
        assert len(saved_files["DataFrames"]) == 1
        assert len(saved_files["figures"]) == 1
        
        # Check directories were created
        assert (self.test_dir / "dataframe_folder").exists()
        assert (self.test_dir / "figure_folder").exists()
        
        # Check files exist
        assert (self.test_dir / "dataframe_folder" / "test_data.csv").exists()
        assert (self.test_dir / "figure_folder" / "test_plot.png").exists()
    
    def test_save_results_with_specific_file(self):
        """Test saving results to specific file paths."""
        # Create mock analysis object
        class MockAnnData:
            def __init__(self):
                self.X = np.array([[1, 2], [3, 4]])
        
        adata = MockAnnData()
        
        # Blueprint configuration with specific file
        outputs_config = {
            "analysis": "transform_output.pickle"
        }
        
        # Results to save
        results = {
            "analysis": adata
        }
        
        # Save results
        saved_files = save_results(
            results=results,
            outputs_config=outputs_config,
            output_base_dir=self.test_dir
        )
        
        # Verify
        assert "analysis" in saved_files
        assert len(saved_files["analysis"]) == 1
        assert (self.test_dir / "transform_output.pickle").exists()
        
        # Verify content can be loaded
        with open(self.test_dir / "transform_output.pickle", 'rb') as f:
            loaded = pickle.load(f)
            assert np.array_equal(loaded.X, adata.X)
    
    def test_save_results_with_lists(self):
        """Test saving multiple items as lists."""
        # Create multiple figures
        figs = []
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot([i, i+1, i+2])
            ax.set_title(f"Plot {i}")
            figs.append(fig)
        
        # Configuration
        outputs_config = {
            "figures": "figure_folder"
        }
        
        # Results as list
        results = {
            "figures": figs
        }
        
        # Save results
        saved_files = save_results(
            results=results,
            outputs_config=outputs_config,
            output_base_dir=self.test_dir
        )
        
        # Verify
        assert "figures" in saved_files
        assert len(saved_files["figures"]) == 3
        
        # Check files exist with auto-generated names
        for i in range(3):
            assert (self.test_dir / "figure_folder" / f"figures_{i}.png").exists()
    
    def test_save_results_with_mixed_types(self):
        """Test saving different data types together."""
        # Create various data types
        df1 = pd.DataFrame({'X': [1, 2], 'Y': [3, 4]})
        df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        fig, ax = plt.subplots()
        ax.bar(['A', 'B'], [10, 20])
        
        # Configuration for mixed outputs
        outputs_config = {
            "analysis": "results.pickle",
            "DataFrames": "dataframe_folder",
            "figures": "figure_folder"
        }
        
        # Results
        results = {
            "analysis": {"summary": "test_analysis"},
            "dataframes": {
                "data1": df1,
                "data2": df2
            },
            "figures": {"bar_chart": fig}
        }
        
        # Save results
        saved_files = save_results(
            results=results,
            outputs_config=outputs_config,
            output_base_dir=self.test_dir
        )
        
        # Verify all types saved
        assert "analysis" in saved_files
        assert "DataFrames" in saved_files
        assert "figures" in saved_files
        
        # Check file counts
        assert len(saved_files["analysis"]) == 1
        assert len(saved_files["DataFrames"]) == 2
        assert len(saved_files["figures"]) == 1
        
        # Verify files exist
        assert (self.test_dir / "results.pickle").exists()
        assert (self.test_dir / "dataframe_folder" / "data1.csv").exists()
        assert (self.test_dir / "dataframe_folder" / "data2.csv").exists()
        assert (self.test_dir / "figure_folder" / "bar_chart.png").exists()
    
    def test_save_results_partial_config(self):
        """Test behavior when results contain types not in config."""
        # Create data
        df = pd.DataFrame({'A': [1, 2, 3]})
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        
        # Config only specifies dataframes
        outputs_config = {
            "DataFrames": "dataframe_folder"
        }
        
        # Results include both dataframes and figures
        results = {
            "dataframes": {"data": df},
            "figures": {"plot": fig}  # Not in config
        }
        
        # Save results - should handle unmatched types gracefully
        saved_files = save_results(
            results=results,
            outputs_config=outputs_config,
            output_base_dir=self.test_dir
        )
        
        # Should save dataframes as configured
        assert "DataFrames" in saved_files
        assert (self.test_dir / "dataframe_folder" / "data.csv").exists()
        
        # Figures should be saved to base directory with warning
        assert "figures" in saved_files


class TestLegacySaveOutputs:
    """Test backward compatibility with original save_outputs."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_legacy_"))
        
    def teardown_method(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        plt.close('all')
    
    def test_legacy_save_outputs(self):
        """Test the legacy save_outputs function still works."""
        # Create test data
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        # Use legacy function
        outputs = {
            "results.csv": df
        }
        
        saved_files = save_outputs(
            outputs=outputs,
            output_dir=self.test_dir
        )
        
        # Verify
        assert "results.csv" in saved_files
        assert (self.test_dir / "results.csv").exists()
        
        # Check content
        loaded_df = pd.read_csv(self.test_dir / "results.csv")
        pd.testing.assert_frame_equal(loaded_df, df)


class TestParseParams:
    """Test parameter parsing functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_params_"))
    
    def teardown_method(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_parse_params_from_dict(self):
        """Test parsing parameters from dictionary."""
        params = {
            "param1": "value1",
            "param2": 123
        }
        
        result = parse_params(params)
        assert result == params
    
    def test_parse_params_from_json_file(self):
        """Test parsing parameters from JSON file."""
        params = {
            "param1": "value1",
            "param2": 123,
            "outputs": {
                "DataFrames": "dataframe_folder"
            }
        }
        
        # Write JSON file
        json_path = self.test_dir / "params.json"
        with open(json_path, 'w') as f:
            json.dump(params, f)
        
        # Parse from file
        result = parse_params(json_path)
        assert result == params
        assert "outputs" in result
    
    def test_parse_params_from_json_string(self):
        """Test parsing parameters from JSON string."""
        params = {"key": "value"}
        json_string = json.dumps(params)
        
        result = parse_params(json_string)
        assert result == params


class TestParseListParameter:
    """Test the parse_list_parameter utility."""
    
    def test_parse_list_from_list(self):
        """Test parsing when input is already a list."""
        input_list = ["item1", "item2", "item3"]
        result = parse_list_parameter(input_list)
        assert result == input_list
    
    def test_parse_list_from_newline_string(self):
        """Test parsing newline-separated string."""
        input_str = "item1\nitem2\nitem3"
        result = parse_list_parameter(input_str)
        assert result == ["item1", "item2", "item3"]
    
    def test_parse_list_from_comma_string(self):
        """Test parsing comma-separated string."""
        input_str = "item1,item2,item3"
        result = parse_list_parameter(input_str)
        assert result == ["item1", "item2", "item3"]
    
    def test_parse_list_from_mixed_string(self):
        """Test parsing string with extra whitespace."""
        input_str = " item1 \n item2 \n item3 "
        result = parse_list_parameter(input_str)
        assert result == ["item1", "item2", "item3"]
    
    def test_parse_list_empty_input(self):
        """Test parsing empty or None input."""
        assert parse_list_parameter(None) == []
        assert parse_list_parameter("") == []
        assert parse_list_parameter("None") == []
    
    def test_parse_list_single_value(self):
        """Test parsing single value."""
        assert parse_list_parameter("single") == ["single"]


class TestTextToValue:
    """Test the text_to_value conversion utility."""
    
    def test_text_to_none(self):
        """Test converting 'None' text to None."""
        assert text_to_value("None") is None
        assert text_to_value("none") is None
        assert text_to_value("") is None
    
    def test_text_to_float(self):
        """Test converting text to float."""
        assert text_to_value("3.14", to_float=True) == 3.14
        assert text_to_value("42", to_float=True) == 42.0
    
    def test_text_to_int(self):
        """Test converting text to integer."""
        assert text_to_value("42", to_int=True) == 42
        assert text_to_value("-10", to_int=True) == -10
    
    def test_text_to_value_error(self):
        """Test error handling for invalid conversions."""
        with pytest.raises(ValueError):
            text_to_value("not_a_number", to_float=True)
        
        with pytest.raises(ValueError):
            text_to_value("3.14", to_int=True)
    
    def test_text_to_custom_value(self):
        """Test converting to custom value."""
        assert text_to_value("None", value_to_convert_to="custom") == "custom"
        assert text_to_value("", value_to_convert_to=42) == 42


class TestLoadInput:
    """Test the load_input function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_load_"))
    
    def teardown_method(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_load_pickle_file(self):
        """Test loading pickle file."""
        # Create test data
        test_data = {"key": "value", "number": 123}
        
        # Save as pickle
        pickle_path = self.test_dir / "test.pickle"
        with open(pickle_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Load and verify
        loaded = load_input(pickle_path)
        assert loaded == test_data
    
    def test_load_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_input(self.test_dir / "nonexistent.pickle")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
