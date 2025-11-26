# tests/templates/test_relational_heatmap_template_new.py
"""
Unit tests for the refactored Relational Heatmap template.
Tests the new implementation using pio.to_image() for PNG export.
"""

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.relational_heatmap_template import run_from_json


def create_test_adata(n_cells: int = 100) -> ad.AnnData:
    """Create a realistic test AnnData object."""
    rng = np.random.default_rng(42)
    
    # Create observations with two categorical annotations
    obs = pd.DataFrame({
        "phenograph_k60_r1": rng.choice(
            ["cluster1", "cluster2", "cluster3"],
            n_cells
        ),
        "renamed_phenotypes": rng.choice(
            ["phenotype_A", "phenotype_B", "phenotype_C"],
            n_cells
        )
    })
    
    # Create expression matrix
    x_mat = rng.normal(size=(n_cells, 20))
    
    # Create AnnData object
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(20)]
    
    return adata


class TestRelationalHeatmapTemplateRefactored(unittest.TestCase):
    """Test suite for the refactored relational heatmap template."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        
        # Create test data file
        self.input_file = self.tmp_path / "input.pickle"
        test_adata = create_test_adata(n_cells=100)
        with open(self.input_file, 'wb') as f:
            pickle.dump(test_adata, f)
        
        # Define test parameters
        self.params = {
            "Upstream_Analysis": str(self.input_file),
            "Source_Annotation_Name": "phenograph_k60_r1",
            "Target_Annotation_Name": "renamed_phenotypes",
            "Colormap": "darkmint",
            "Figure_Width_inch": 8,
            "Figure_Height_inch": 10,
            "Figure_DPI": 150,  # Lower DPI for faster tests
            "Font_Size": 8,
            "Output_File": "relational_heatmap",
            "Output_Directory": str(self.tmp_path),
            "outputs": {
                "figures": {"type": "directory", "name": "figures_dir"},
                "html": {"type": "directory", "name": "html_dir"},
                "dataframe": {"type": "file", "name": "dataframe.csv"}
            }
        }

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.tmp_dir.cleanup()

    def test_complete_workflow_with_file_outputs(self):
        """Test complete workflow with actual file outputs."""
        print("\n=== Testing Complete Workflow ===")
        
        # Run the template
        result = run_from_json(self.params, save_to_disk=True)
        
        # Check that result is a dictionary
        self.assertIsInstance(result, dict)
        print(f"✓ Result is a dictionary with keys: {list(result.keys())}")
        
        # Check that all expected outputs are present
        self.assertIn("figures", result)
        self.assertIn("html", result)
        self.assertIn("dataframe", result)
        print("✓ All expected output types present")
        
        # Check figures directory
        figures_list = result["figures"]
        self.assertIsInstance(figures_list, list)
        self.assertGreater(len(figures_list), 0)
        
        # Verify PNG file exists and has content
        png_file = Path(figures_list[0])
        self.assertTrue(png_file.exists())
        self.assertTrue(png_file.suffix == '.png')
        png_size = png_file.stat().st_size
        self.assertGreater(png_size, 1000)  # Should be at least 1KB
        print(f"✓ PNG file created: {png_file.name} ({png_size} bytes)")
        
        # Check HTML directory
        html_list = result["html"]
        self.assertIsInstance(html_list, list)
        self.assertGreater(len(html_list), 0)
        
        # Verify HTML file exists and has content
        html_file = Path(html_list[0])
        self.assertTrue(html_file.exists())
        self.assertTrue(html_file.suffix == '.html')
        
        # Check HTML content
        with open(html_file, 'r') as f:
            html_content = f.read()
        self.assertIn('plotly', html_content.lower())
        self.assertGreater(len(html_content), 1000)
        print(f"✓ HTML file created: {html_file.name} "
              f"({len(html_content)} chars)")
        
        # Check dataframe file
        df_file = Path(result["dataframe"])
        self.assertTrue(df_file.exists())
        self.assertTrue(df_file.suffix == '.csv')
        
        # Verify CSV content
        df = pd.read_csv(df_file)
        self.assertGreater(len(df), 0)
        self.assertGreater(len(df.columns), 0)
        print(f"✓ CSV file created: {df_file.name} "
              f"({df.shape[0]} rows, {df.shape[1]} cols)")
        
        print("✓ Complete workflow test passed!")

    def test_no_save_returns_figure_and_data(self):
        """Test that save_to_disk=False returns figure and dataframe."""
        print("\n=== Testing No-Save Mode ===")
        
        # Run without saving
        result = run_from_json(self.params, save_to_disk=False)
        
        # Check result is a tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        print("✓ Result is a tuple of length 2")
        
        # Unpack result
        fig, df = result
        
        # Check figure is a Plotly figure
        self.assertTrue(hasattr(fig, 'data'))
        self.assertTrue(hasattr(fig, 'layout'))
        print(f"✓ Returned Plotly figure: {type(fig).__name__}")
        
        # Check dataframe
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        print(f"✓ Returned DataFrame: {df.shape[0]} rows, "
              f"{df.shape[1]} cols")
        
        print("✓ No-save mode test passed!")

    def test_json_file_input(self):
        """Test reading parameters from JSON file."""
        print("\n=== Testing JSON File Input ===")
        
        # Write params to JSON file
        json_file = self.tmp_path / "params.json"
        with open(json_file, 'w') as f:
            json.dump(self.params, f)
        print(f"✓ Created JSON file: {json_file}")
        
        # Run with JSON file path
        result = run_from_json(str(json_file), save_to_disk=True)
        
        # Verify outputs
        self.assertIsInstance(result, dict)
        self.assertIn("figures", result)
        print("✓ Successfully loaded params from JSON file")
        
        print("✓ JSON file input test passed!")

    def test_png_image_quality(self):
        """Test that PNG image is properly generated with correct size."""
        print("\n=== Testing PNG Image Quality ===")
        
        # Run with specific dimensions
        self.params["Figure_DPI"] = 300
        self.params["Figure_Width_inch"] = 10
        self.params["Figure_Height_inch"] = 12
        
        result = run_from_json(self.params, save_to_disk=True)
        
        # Get PNG file
        png_file = Path(result["figures"][0])
        png_size = png_file.stat().st_size
        
        # Higher DPI should produce larger file
        self.assertGreater(png_size, 10000)  # At least 10KB
        print(f"✓ High-quality PNG: {png_size} bytes")
        
        # Try to verify it's a valid PNG by checking magic bytes
        with open(png_file, 'rb') as f:
            header = f.read(8)
        self.assertEqual(header[:4], b'\x89PNG')
        print("✓ Valid PNG file header")
        
        print("✓ PNG quality test passed!")

    def test_different_colormaps(self):
        """Test with different colormap options."""
        print("\n=== Testing Different Colormaps ===")
        
        for colormap in ['viridis', 'plasma', 'darkmint']:
            print(f"  Testing colormap: {colormap}")
            self.params["Colormap"] = colormap
            self.params["Output_Directory"] = str(
                self.tmp_path / f"output_{colormap}"
            )
            
            result = run_from_json(self.params, save_to_disk=True)
            
            # Verify outputs created
            self.assertIsInstance(result, dict)
            png_file = Path(result["figures"][0])
            self.assertTrue(png_file.exists())
            print(f"    ✓ {colormap}: {png_file.stat().st_size} bytes")
        
        print("✓ Colormap test passed!")


def run_specific_test(test_name: str = None):
    """Run a specific test or all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_name:
        # Run specific test
        suite.addTest(
            TestRelationalHeatmapTemplateRefactored(test_name)
        )
    else:
        # Run all tests
        suite.addTests(
            loader.loadTestsFromTestCase(
                TestRelationalHeatmapTemplateRefactored
            )
        )
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Relational Heatmap Template - Refactored Tests")
    print("Testing new implementation with pio.to_image()")
    print("=" * 70)
    
    # Run all tests
    result = run_specific_test()
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
    print("=" * 70)
    
    sys.exit(0 if result.wasSuccessful() else 1)
