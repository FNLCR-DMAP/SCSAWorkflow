# tests/templates/test_add_pin_color_rule_template.py
"""Unit tests for the Append Pin Color Rule template."""

import json
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch, MagicMock

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.add_pin_color_rule_template import run_from_json


def mock_adata(n_cells: int = 10) -> ad.AnnData:
    """Return a minimal synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "cell_type": (["TypeA", "TypeB"] * ((n_cells + 1) // 2))[:n_cells],
        "batch": (["Batch1", "Batch2"] * ((n_cells + 1) // 2))[:n_cells]
    })
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    # Initialize uns dict for color rules
    adata.uns = {}
    return adata


class TestAddPinColorRuleTemplate(unittest.TestCase):
    """Unit tests for the Append Pin Color Rule template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(
            self.tmp_dir.name, "input.pickle"
        )
        self.out_file = "color_mapped_analysis"

        # Save minimal mock data
        with open(self.in_file, 'wb') as f:
            pickle.dump(mock_adata(), f)

        # Minimal parameters from NIDAP template
        self.params = {
            "Upstream_Analysis": self.in_file,
            "Label_Color_Map": ["TypeA:red", "TypeB:blue"],
            "Color_Map_Name": "_spac_colors",
            "Overwrite_Previous_Color_Map": True,
            "Output_File": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_complete_io_workflow(self) -> None:
        """Single I/O test covering input/output scenarios."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Test 1: Run with default parameters
            result = run_from_json(self.params)
            self.assertIsInstance(result, dict)
            # Verify file was saved
            self.assertTrue(len(result) > 0)
            pickle_files = [f for f in result.values() if '.pickle' in str(f)]
            self.assertTrue(len(pickle_files) > 0)
            
            # Test 2: Run without saving
            result_no_save = run_from_json(self.params, save_results=False)
            # Check appropriate return type based on template
            self.assertIsInstance(result_no_save, ad.AnnData)
            # Verify color map was added
            self.assertIn("_spac_colors", result_no_save.uns)
            self.assertIn("_spac_colors_summary", result_no_save.uns)
            
            # Test 3: JSON file input
            json_path = os.path.join(self.tmp_dir.name, "params.json")
            with open(json_path, "w") as f:
                json.dump(self.params, f)
            
            result_json = run_from_json(json_path)
            self.assertIsInstance(result_json, dict)

    def test_error_validation(self) -> None:
        """Test exact error message for invalid parameters."""
        # Test invalid color format
        params_bad = self.params.copy()
        params_bad["Label_Color_Map"] = ["TypeA-red"]  # Missing colon
        
        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)
        
        # Check error message contains expected text
        error_msg = str(context.exception)
        self.assertIn("Missing ':' separator", error_msg)

    @patch('spac.templates.add_pin_color_rule_template.add_pin_color_rules')
    def test_function_calls(self, mock_add_rules) -> None:
        """Test that main function is called with correct parameters."""
        # Mock the main function to simulate adding summary to adata.uns
        def side_effect_add_rules(adata, **kwargs):
            color_map_name = kwargs.get('color_map_name', '_spac_colors')
            adata.uns[f'{color_map_name}_summary'] = "Mock summary"
            return None
        
        mock_add_rules.side_effect = side_effect_add_rules
        
        # Test with different parameters
        params_alt = self.params.copy()
        params_alt["Color_Map_Name"] = "custom_colors"
        params_alt["Overwrite_Previous_Color_Map"] = False
        params_alt["Label_Color_Map"] = [
            "CD4+ T cells:cyan",
            "CD8+ T cells:royalblue",
            "B cells:yellowgreen"
        ]
        
        run_from_json(params_alt, save_results=False)
        
        # Verify function was called correctly
        mock_add_rules.assert_called_once()
        call_args = mock_add_rules.call_args
        
        # Check specific parameter conversions
        self.assertEqual(call_args[1]['color_map_name'], "custom_colors")
        self.assertEqual(call_args[1]['overwrite'], False)
        
        # Verify color dict was parsed correctly
        expected_dict = {
            "CD4+ T cells": "cyan",
            "CD8+ T cells": "royalblue",
            "B cells": "yellowgreen"
        }
        self.assertEqual(call_args[1]['label_color_dict'], expected_dict)


if __name__ == "__main__":
    unittest.main()