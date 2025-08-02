#!/usr/bin/env python3
"""Unit tests for the Manual Phenotyping template."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.manual_phenotyping_template import run_from_json


def create_mock_data_and_phenotypes(tmp_dir: Path) -> tuple:
    """Create minimal data and phenotypes files for testing."""
    # Create mock expression data
    n_cells = 20  # Simple scenario with 20 cells for testing
    data = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_cells)],
        'CD3D_expression': np.random.choice([0, 1], n_cells),
        'CD4_expression': np.random.choice([0, 1], n_cells),
        'CD8A_expression': np.random.choice([0, 1], n_cells),
        'FOXP3_expression': np.random.choice([0, 1], n_cells),
        'CD68_expression': np.random.choice([0, 1], n_cells),
        'CD20_expression': np.random.choice([0, 1], n_cells),
        'CD21_expression': np.random.choice([0, 1], n_cells),
        'CD56_expression': np.random.choice([0, 1], n_cells),
    })
    data_path = tmp_dir / 'input_data.csv'
    data.to_csv(data_path, index=False)
    
    # Create phenotypes definition
    phenotypes = pd.DataFrame({
        'phenotype_code': [
            'CD3D+CD4+FOXP3+',
            'CD3D+CD4+',
            'CD3D+CD8A+',
            'CD68+',
            'CD20+'
        ],
        'phenotype_name': [
            'Regulatory T Cell',
            'Helper T Cell',
            'Cytotoxic T Cell',
            'Macrophage',
            'B Cell'
        ]
    })
    phenotypes_path = tmp_dir / 'phenotypes.csv'
    phenotypes.to_csv(phenotypes_path, index=False)
    
    return data_path, phenotypes_path


class TestManualPhenotypingTemplate(unittest.TestCase):
    """Unit tests for the Manual Phenotyping template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        
        # Create mock files
        self.data_path, self.phenotypes_path = \
            create_mock_data_and_phenotypes(self.tmp_path)
        
        # Minimal parameters
        self.params = {
            "Upstream_Dataset": str(self.data_path),
            "Phenotypes_Code": str(self.phenotypes_path),
            "Classification_Column_Prefix": "",
            "Classification_Column_Suffix": "_expression",
            "Allow_Multiple_Phenotypes": True,
            "Manual_Annotation_Name": "manual_phenotype",
            "Output_File": "phenotyped_data.csv"
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.manual_phenotyping_template.'
           'assign_manual_phenotypes')
    def test_complete_io_workflow(self, mock_assign) -> None:
        """Single I/O test covering all input/output scenarios."""
        # Mock the assign_manual_phenotypes function
        def mock_assign_func(df, pheno, **kwargs):
            # Add phenotype column with the correct annotation name
            annotation_name = kwargs.get('annotation', 'manual_phenotype')
            df[annotation_name] = np.random.choice(
                ['T Cell', 'B Cell', 'Macrophage', 'no_label'],
                len(df)
            )
            return {'status': 'success'}
        
        mock_assign.side_effect = mock_assign_func
        
        # Change to temp directory for output
        original_cwd = os.getcwd()
        os.chdir(self.tmp_path)
        
        try:
            # Test 1: Run with save_results=True
            saved_files = run_from_json(self.params)
            self.assertIn("phenotyped_data.csv", saved_files)
            output_path = Path(saved_files["phenotyped_data.csv"])
            self.assertTrue(output_path.exists())
            
            # Verify content
            result_df = pd.read_csv(output_path)
            self.assertEqual(len(result_df), 20)  # Same number of cells
            self.assertIn('manual_phenotype', result_df.columns)
            self.assertIn('cell_id', result_df.columns)
            
            # Test 2: Run with save_results=False (in-memory)
            df_result = run_from_json(self.params, save_results=False)
            self.assertIsInstance(df_result, pd.DataFrame)
            self.assertEqual(len(df_result), 20)
            self.assertIn('manual_phenotype', df_result.columns)
            
            # Test 3: JSON file input
            json_path = self.tmp_path / "params.json"
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            saved_from_json = run_from_json(json_path)
            self.assertIn("phenotyped_data.csv", saved_from_json)
            
            # Test 4: Direct DataFrame input (chained workflow)
            input_df = pd.read_csv(self.data_path)
            params_df = self.params.copy()
            params_df["Upstream_Dataset"] = input_df  # Pass DataFrame
            
            df_from_df = run_from_json(params_df, save_results=False)
            self.assertIsInstance(df_from_df, pd.DataFrame)
            self.assertEqual(len(df_from_df), 20)
            self.assertIn('manual_phenotype', df_from_df.columns)
            
            # Test 5: Custom parameters
            params_custom = self.params.copy()
            params_custom["Allow_Multiple_Phenotypes"] = False
            params_custom["Manual_Annotation_Name"] = "cell_type"
            params_custom["Output_File"] = "custom_output.csv"
            
            saved_custom = run_from_json(params_custom)
            self.assertIn("custom_output.csv", saved_custom)
            
            # Verify custom annotation name
            custom_df = pd.read_csv(saved_custom["custom_output.csv"])
            self.assertIn('cell_type', custom_df.columns)
            
            # Verify mock was called with correct parameters
            call_args = mock_assign.call_args_list[-1]  # Last call
            self.assertEqual(call_args[1]['multiple'], False)
            self.assertEqual(call_args[1]['annotation'], 'cell_type')
            
        finally:
            os.chdir(original_cwd)

    @patch('spac.templates.manual_phenotyping_template.'
           'assign_manual_phenotypes')
    def test_error_validation(self, mock_assign) -> None:
        """Test exact error messages for various failure scenarios."""
        # Test 1: Missing phenotypes file
        params_missing = self.params.copy()
        params_missing["Phenotypes_Code"] = str(
            self.tmp_path / "missing.csv"
        )
        
        with self.assertRaises(FileNotFoundError) as context:
            run_from_json(params_missing)
   
        # Test 2: Missing input data file
        params_no_input = self.params.copy()
        params_no_input["Upstream_Dataset"] = str(
            self.tmp_path / "nonexistent.csv"
        )
        
        with self.assertRaises(FileNotFoundError):
            run_from_json(params_no_input)
        
        # Test 3: Invalid CSV file for phenotypes
        # SPAC function error
        # Mock to simulate SPAC function error
        mock_assign.side_effect = ValueError("Invalid phenotype code format")
        
        with self.assertRaises(ValueError) as context:
            run_from_json(self.params)
        
        expected_msg = "Invalid phenotype code format"
        self.assertEqual(str(context.exception), expected_msg)

    @patch('spac.templates.manual_phenotyping_template.'
           'assign_manual_phenotypes')
    @patch('builtins.print')
    def test_console_output(self, mock_print, mock_assign) -> None:
        """Test that expected messages are printed to console."""
        # Mock the assign function
        def mock_assign_func(df, pheno, **kwargs):
            df['manual_phenotype'] = ['T Cell'] * len(df)
            return {'status': 'success'}
        
        mock_assign.side_effect = mock_assign_func
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(self.tmp_path)
        
        try:
            run_from_json(self.params)
            
            # Check for expected print statements
            print_calls = [str(call[0][0]) for call in 
                          mock_print.call_args_list if call[0]]
            
            # Should print phenotypes DataFrame
            phenotypes_printed = any('phenotype_code' in str(call) 
                                   for call in print_calls)
            self.assertTrue(phenotypes_printed)
            
            # Should print completion message
            completion_msgs = [
                msg for msg in print_calls
                if 'Manual Phenotyping completed successfully' in msg
            ]
            self.assertTrue(len(completion_msgs) > 0)
            
            # Should print file save message
            save_msgs = [
                msg for msg in print_calls
                if 'Manual Phenotyping completed →' in msg
            ]
            self.assertTrue(len(save_msgs) > 0)
            
        finally:
            os.chdir(original_cwd)

    @patch('spac.templates.manual_phenotyping_template.'
           'assign_manual_phenotypes')
    def test_phenotype_distribution_output(self, mock_assign) -> None:
        """Test that phenotype distribution is correctly calculated/printed."""
        # Create specific phenotype assignments
        def mock_assign_func(df, pheno, **kwargs):
            # Assign specific phenotypes for testing
            phenotypes = ['T Cell'] * 10 + ['B Cell'] * 5 + ['no_label'] * 5
            df[kwargs.get('annotation', 'manual_phenotype')] = phenotypes[:len(df)]
            return {'status': 'success'}
        
        mock_assign.side_effect = mock_assign_func
        
        with patch('builtins.print') as mock_print:
            df_result = run_from_json(self.params, save_results=False)
            
            # Check distribution in result
            counts = df_result['manual_phenotype'].value_counts()
            self.assertEqual(counts['T Cell'], 10)
            self.assertEqual(counts['B Cell'], 5)
            self.assertEqual(counts['no_label'], 5)
            
            # Check that distribution was printed
            print_calls = [str(call[0][0]) for call in 
                          mock_print.call_args_list if call[0]]
            distribution_printed = any(
                'Phenotype distribution' in str(call) 
                for call in print_calls
            )
            self.assertTrue(distribution_printed)


if __name__ == "__main__":
    unittest.main()