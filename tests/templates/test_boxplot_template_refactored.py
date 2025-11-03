"""
Unit tests for refactored boxplot_template.
Focus on testing the new save_results integration and outputs configuration.
"""
import pytest
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))
from boxplot_template_refactored import run_from_json


class TestBoxplotTemplate:
    """Test the refactored boxplot template with new save_results."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_boxplot_"))
        
        # Create mock AnnData object
        self.mock_adata = Mock()
        self.mock_adata.X = np.random.rand(100, 10)
        self.mock_adata.n_obs = 100
        self.mock_adata.n_vars = 10
        self.mock_adata.var_names = [f"Gene_{i}" for i in range(10)]
        self.mock_adata.obs = pd.DataFrame({
            "cell_type": np.random.choice(["A", "B", "C"], 100),
            "treatment": np.random.choice(["Control", "Treated"], 100)
        })
        self.mock_adata.layers = {}
        
        # Create test parameters
        self.test_params = {
            "Upstream_Analysis": str(self.test_dir / "input.pickle"),
            "Primary_Annotation": "cell_type",
            "Secondary_Annotation": "None",
            "Table_to_Visualize": "Original",
            "Feature_s_to_Plot": ["Gene_0", "Gene_1"],
            "Value_Axis_Log_Scale": False,
            "Figure_Title": "Test Boxplot",
            "Horizontal_Plot": False,
            "Figure_Width": 10,
            "Figure_Height": 6,
            "Figure_DPI": 100,
            "Font_Size": 12,
            "Keep_Outliers": True,
            "output_dir": str(self.test_dir)
        }
        
        # Save mock input file
        import pickle
        with open(self.test_dir / "input.pickle", 'wb') as f:
            pickle.dump(self.mock_adata, f)
    
    def teardown_method(self):
        """Clean up test directory and close plots."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        plt.close('all')
    
    @patch('boxplot_template_refactored.boxplot')
    def test_run_with_default_outputs_config(self, mock_boxplot_func):
        """Test running boxplot with default outputs configuration."""
        # Setup mock boxplot function
        fig, ax = plt.subplots()
        df = pd.DataFrame({
            'Gene_0': np.random.rand(100),
            'Gene_1': np.random.rand(100)
        })
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Run with default config
        result = run_from_json(
            self.test_params,
            save_results_flag=True,
            show_plot=False
        )
        
        # Verify outputs structure
        assert isinstance(result, dict)
        assert "DataFrames" in result or "dataframes" in result
        assert "figures" in result
        
        # Check that directories were created
        assert (self.test_dir / "dataframe_folder").exists()
        assert (self.test_dir / "figure_folder").exists()
        
        # Verify files were saved
        dataframe_files = list((self.test_dir / "dataframe_folder").glob("*.csv"))
        figure_files = list((self.test_dir / "figure_folder").glob("*.png"))
        
        assert len(dataframe_files) > 0
        assert len(figure_files) > 0
    
    @patch('boxplot_template_refactored.boxplot')
    def test_run_with_custom_outputs_config(self, mock_boxplot_func):
        """Test running boxplot with custom outputs configuration."""
        # Setup mock boxplot function
        fig, ax = plt.subplots()
        df = pd.DataFrame({
            'Gene_0': np.random.rand(100),
            'Gene_1': np.random.rand(100)
        })
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Custom outputs configuration
        custom_outputs_config = {
            "DataFrames": "custom_data_dir",
            "figures": "custom_fig_dir"
        }
        
        # Run with custom config
        result = run_from_json(
            self.test_params,
            save_results_flag=True,
            show_plot=False,
            outputs_config=custom_outputs_config
        )
        
        # Check that custom directories were created
        assert (self.test_dir / "custom_data_dir").exists()
        assert (self.test_dir / "custom_fig_dir").exists()
        
        # Verify files in custom directories
        dataframe_files = list((self.test_dir / "custom_data_dir").glob("*.csv"))
        figure_files = list((self.test_dir / "custom_fig_dir").glob("*.png"))
        
        assert len(dataframe_files) > 0
        assert len(figure_files) > 0
    
    @patch('boxplot_template_refactored.boxplot')
    def test_run_without_saving(self, mock_boxplot_func):
        """Test running boxplot without saving results."""
        # Setup mock boxplot function
        fig, ax = plt.subplots()
        df = pd.DataFrame({
            'Gene_0': np.random.rand(100),
            'Gene_1': np.random.rand(100)
        })
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Run without saving
        result = run_from_json(
            self.test_params,
            save_results_flag=False,
            show_plot=False
        )
        
        # Should return tuple of (figure, dataframe)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        # Verify types
        returned_fig, returned_df = result
        assert hasattr(returned_fig, 'savefig')  # Check it's a figure
        assert isinstance(returned_df, pd.DataFrame)
        
        # No directories should be created
        assert not (self.test_dir / "dataframe_folder").exists()
        assert not (self.test_dir / "figure_folder").exists()
    
    @patch('boxplot_template_refactored.boxplot')
    def test_all_features_selection(self, mock_boxplot_func):
        """Test selecting all features for plotting."""
        # Setup mock boxplot function
        fig, ax = plt.subplots()
        df = pd.DataFrame({f'Gene_{i}': np.random.rand(100) for i in range(10)})
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Modify params to use "All" features
        params = self.test_params.copy()
        params["Feature_s_to_Plot"] = ["All"]
        
        # Run
        result = run_from_json(
            params,
            save_results_flag=True,
            show_plot=False
        )
        
        # Verify boxplot was called with all features
        mock_boxplot_func.assert_called_once()
        call_args = mock_boxplot_func.call_args
        
        # Check features argument contains all genes
        features_arg = call_args[1]['features']
        assert len(features_arg) == 10
        assert all(f"Gene_{i}" in features_arg for i in range(10))
    
    @patch('boxplot_template_refactored.boxplot')
    def test_parameter_processing(self, mock_boxplot_func):
        """Test correct processing of template parameters."""
        # Setup mock boxplot function
        fig, ax = plt.subplots()
        df = pd.DataFrame({'data': [1, 2, 3]})
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Test various parameter combinations
        test_cases = [
            # (params_update, expected_args)
            ({"Secondary_Annotation": "treatment"}, {"second_annotation": "treatment"}),
            ({"Table_to_Visualize": "normalized"}, {"layer": "normalized"}),
            ({"Horizontal_Plot": True}, {"orient": "h"}),
            ({"Value_Axis_Log_Scale": True}, {"log_scale": True}),
            ({"Keep_Outliers": False}, {"showfliers": False}),
        ]
        
        for params_update, expected_args in test_cases:
            # Update parameters
            params = self.test_params.copy()
            params.update(params_update)
            
            # Reset mock
            mock_boxplot_func.reset_mock()
            
            # Run
            run_from_json(
                params,
                save_results_flag=False,
                show_plot=False
            )
            
            # Verify correct arguments were passed
            call_args = mock_boxplot_func.call_args[1]
            for key, value in expected_args.items():
                assert call_args[key] == value, f"Failed for {key}: expected {value}, got {call_args.get(key)}"
    
    @patch('boxplot_template_refactored.boxplot')
    def test_outputs_config_from_params(self, mock_boxplot_func):
        """Test that outputs config can be read from params."""
        # Setup mock boxplot function
        fig, ax = plt.subplots()
        df = pd.DataFrame({'data': [1, 2, 3]})
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Add outputs config to params
        params = self.test_params.copy()
        params["outputs"] = {
            "DataFrames": "param_data_folder",
            "figures": "param_fig_folder"
        }
        
        # Run without explicit outputs_config
        result = run_from_json(
            params,
            save_results_flag=True,
            show_plot=False,
            outputs_config=None  # Should use from params
        )
        
        # Check that directories from params were created
        assert (self.test_dir / "param_data_folder").exists()
        assert (self.test_dir / "param_fig_folder").exists()
    
    @patch('boxplot_template_refactored.boxplot')
    def test_multiple_output_files(self, mock_boxplot_func):
        """Test that multiple dataframes are saved correctly."""
        # Setup mock boxplot function
        fig, ax = plt.subplots()
        df = pd.DataFrame({
            'Gene_0': np.random.rand(100),
            'Gene_1': np.random.rand(100)
        })
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Run
        result = run_from_json(
            self.test_params,
            save_results_flag=True,
            show_plot=False
        )
        
        # Check that both summary and plot data are saved
        dataframe_files = list((self.test_dir / "dataframe_folder").glob("*.csv"))
        dataframe_names = [f.stem for f in dataframe_files]
        
        # Should have summary_statistics and plot_data
        assert any("summary" in name for name in dataframe_names)
        assert any("plot_data" in name for name in dataframe_names)
    
    def test_json_file_input(self):
        """Test running with JSON file path instead of dict."""
        # Save params to JSON file
        json_path = self.test_dir / "params.json"
        with open(json_path, 'w') as f:
            json.dump(self.test_params, f)
        
        # Mock the boxplot function
        with patch('boxplot_template_refactored.boxplot') as mock_boxplot_func:
            fig, ax = plt.subplots()
            df = pd.DataFrame({'data': [1, 2, 3]})
            mock_boxplot_func.return_value = (fig, ax, df)
            
            # Run with JSON file path
            result = run_from_json(
                str(json_path),
                save_results_flag=True,
                show_plot=False
            )
            
            # Verify it worked
            assert isinstance(result, dict)
            assert len(result) > 0
    
    def test_error_handling_missing_input(self):
        """Test error handling when input file is missing."""
        params = self.test_params.copy()
        params["Upstream_Analysis"] = str(self.test_dir / "nonexistent.pickle")
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            run_from_json(
                params,
                save_results_flag=True,
                show_plot=False
            )


class TestBoxplotCLI:
    """Test command-line interface functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_cli_"))
        
        # Create mock input file
        mock_adata = Mock()
        mock_adata.X = np.random.rand(10, 5)
        mock_adata.var_names = [f"Gene_{i}" for i in range(5)]
        mock_adata.obs = pd.DataFrame({"type": ["A", "B"] * 5})
        mock_adata.layers = {}
        
        import pickle
        with open(self.test_dir / "input.pickle", 'wb') as f:
            pickle.dump(mock_adata, f)
        
        # Create test parameters
        self.params = {
            "Upstream_Analysis": str(self.test_dir / "input.pickle"),
            "Primary_Annotation": "type",
            "Feature_s_to_Plot": ["Gene_0"],
            "output_dir": str(self.test_dir)
        }
        
        # Create blueprint
        self.blueprint = {
            "outputs": {
                "DataFrames": "blueprint_data",
                "figures": "blueprint_figs"
            }
        }
    
    def teardown_method(self):
        """Clean up test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        plt.close('all')
    
    @patch('boxplot_template_refactored.boxplot')
    def test_cli_with_blueprint(self, mock_boxplot_func):
        """Test CLI with blueprint file for outputs config."""
        # Setup mock
        fig, ax = plt.subplots()
        df = pd.DataFrame({'data': [1, 2, 3]})
        mock_boxplot_func.return_value = (fig, ax, df)
        
        # Save files
        params_path = self.test_dir / "params.json"
        blueprint_path = self.test_dir / "blueprint.json"
        
        with open(params_path, 'w') as f:
            json.dump(self.params, f)
        with open(blueprint_path, 'w') as f:
            json.dump(self.blueprint, f)
        
        # Simulate CLI execution
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["boxplot_template.py", str(params_path), str(blueprint_path)]
            
            # Load outputs config from blueprint as CLI would
            with open(blueprint_path, 'r') as f:
                blueprint = json.load(f)
                outputs_config = blueprint.get("outputs", None)
            
            # Run
            result = run_from_json(
                str(params_path),
                outputs_config=outputs_config
            )
            
            # Verify blueprint directories were used
            assert (self.test_dir / "blueprint_data").exists()
            assert (self.test_dir / "blueprint_figs").exists()
            
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
