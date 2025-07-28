# tests/templates/test_setup_analysis_template.py
"""Unit tests for the Setup Analysis template."""

import json
import os
import sys
import tempfile
import unittest
import pickle
from pathlib import Path
import pandas as pd
import anndata as ad
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.setup_analysis_template import run_from_json


def create_test_dataframe(n_cells: int = 10) -> pd.DataFrame:
    """Create minimal test dataframe for setup analysis."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'CellID': range(1, n_cells + 1),
        'X_centroid': rng.uniform(0, 100, n_cells),
        'Y_centroid': rng.uniform(0, 100, n_cells),
        'CD25': rng.normal(10, 2, n_cells),
        'CD3D': rng.normal(15, 3, n_cells),
        'CD45': rng.normal(20, 4, n_cells),
        'CD4': rng.normal(12, 2.5, n_cells),
        'CD8A': rng.normal(8, 2, n_cells),
        'broad_cell_type': rng.choice(
            ['T cells', 'B cells'], n_cells
        ),
        'detailed_cell_type': rng.choice(
            ['CD4 T cells', 'CD8 T cells', 'B cells'], n_cells
        )
    })
    return df


class TestSetupAnalysisTemplate(unittest.TestCase):
    """Unit tests for the Setup Analysis template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()

        # Create test data
        self.test_df = create_test_dataframe()
        self.csv_file = os.path.join(
            self.tmp_dir.name, "test_data.csv"
        )
        self.test_df.to_csv(self.csv_file, index=False)

        # Minimal parameters
        self.params = {
            "Upstream_Dataset": self.csv_file,
            "Features_to_Analyze": ["CD25", "CD3D", "CD45", "CD4", "CD8A"],
            "Feature_Regex": [],
            "X_Coordinate_Column": "X_centroid",
            "Y_Coordinate_Column": "Y_centroid",
            "Annotation_s_": ["broad_cell_type", "detailed_cell_type"],
            "Output_File": "analysis_output.pickle"
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    @patch('spac.templates.setup_analysis_template.ingest_cells')
    def test_run_with_save(self, mock_ingest) -> None:
        """Test setup analysis with file saving."""
        # Mock the ingest_cells function
        mock_adata = ad.AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame({'cell_type': ['A'] * 10})
        )
        mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
        mock_ingest.return_value = mock_adata

        saved_files = run_from_json(self.params)

        # Check that ingest_cells was called with correct parameters
        mock_ingest.assert_called_once()
        call_args = mock_ingest.call_args

        # Verify the call arguments
        self.assertIsInstance(call_args[1]['dataframe'], pd.DataFrame)
        self.assertEqual(
            call_args[1]['x_col'], "X_centroid"
        )
        self.assertEqual(
            call_args[1]['y_col'], "Y_centroid"
        )
        self.assertEqual(
            call_args[1]['annotation'],
            ["broad_cell_type", "detailed_cell_type"]
        )

        # Check regex includes feature names
        regex_list = call_args[1]['regex_str']
        for feature in self.params["Features_to_Analyze"]:
            self.assertIn(f"^{feature}$", regex_list)

        # Check that output file was created
        self.assertIn("analysis_output.pickle", saved_files)

    @patch('spac.templates.setup_analysis_template.ingest_cells')
    def test_run_without_save(self, mock_ingest) -> None:
        """Test setup analysis without file saving."""
        # Mock the ingest_cells function
        mock_adata = ad.AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame({'cell_type': ['A'] * 10})
        )
        mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
        mock_ingest.return_value = mock_adata

        result = run_from_json(self.params, save_results=False)

        # Check that we got an AnnData object back
        self.assertIsInstance(result, ad.AnnData)
        # For AnnData, we check that it's the same object reference
        self.assertIs(result, mock_adata)

    def test_annotation_none_handling(self) -> None:
        """Test handling of 'None' annotation."""
        params_none = self.params.copy()
        params_none["Annotation_s_"] = ["None"]

        with patch(
            'spac.templates.setup_analysis_template.ingest_cells'
        ) as mock_ingest:
            mock_adata = ad.AnnData(X=np.random.rand(10, 5))
            mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
            mock_ingest.return_value = mock_adata

            run_from_json(params_none, save_results=False)

            # Check that annotation was set to None
            call_args = mock_ingest.call_args
            self.assertIsNone(call_args[1]['annotation'])

    def test_annotation_validation_error_message(self) -> None:
        """Test exact error message for invalid annotation."""
        params_bad = self.params.copy()
        params_bad["Annotation_s_"] = ["broad_cell_type", "None", "other"]

        with self.assertRaises(ValueError) as context:
            run_from_json(params_bad)

        expected_msg = 'String "None" found in the annotation list'
        actual_msg = str(context.exception)
        self.assertEqual(expected_msg, actual_msg)

    def test_coordinate_none_handling(self) -> None:
        """Test handling of 'None' coordinates."""
        params_no_coords = self.params.copy()
        params_no_coords["X_Coordinate_Column"] = "None"
        params_no_coords["Y_Coordinate_Column"] = "None"

        with patch(
            'spac.templates.setup_analysis_template.ingest_cells'
        ) as mock_ingest:
            mock_adata = ad.AnnData(X=np.random.rand(10, 5))
            mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
            mock_ingest.return_value = mock_adata

            run_from_json(params_no_coords, save_results=False)

            # Check that coordinates were set to None
            call_args = mock_ingest.call_args
            self.assertIsNone(call_args[1]['x_col'])
            self.assertIsNone(call_args[1]['y_col'])

    def test_json_file_input(self) -> None:
        """Test with JSON file input."""
        json_path = os.path.join(self.tmp_dir.name, "params.json")
        with open(json_path, "w") as f:
            json.dump(self.params, f)

        with patch(
            'spac.templates.setup_analysis_template.ingest_cells'
        ) as mock_ingest:
            mock_adata = ad.AnnData(X=np.random.rand(10, 5))
            mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
            mock_ingest.return_value = mock_adata

            saved_files = run_from_json(json_path)

            # Check that files were saved
            self.assertTrue(len(saved_files) > 0)

    def test_feature_regex_combination(self) -> None:
        """Test combination of feature names and regex."""
        params_regex = self.params.copy()
        params_regex["Feature_Regex"] = [".*_expression$", "DAPI.*"]

        with patch(
            'spac.templates.setup_analysis_template.ingest_cells'
        ) as mock_ingest:
            mock_adata = ad.AnnData(X=np.random.rand(10, 5))
            mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
            mock_ingest.return_value = mock_adata

            run_from_json(params_regex, save_results=False)

            # Check that regex includes both custom patterns and features
            call_args = mock_ingest.call_args
            regex_list = call_args[1]['regex_str']

            # Should include custom regex
            self.assertIn(".*_expression$", regex_list)
            self.assertIn("DAPI.*", regex_list)

            # Should include feature patterns
            for feature in params_regex["Features_to_Analyze"]:
                self.assertIn(f"^{feature}$", regex_list)

    def test_dataframe_input(self) -> None:
        """Test with DataFrame as upstream dataset."""
        params_df = self.params.copy()
        params_df["Upstream_Dataset"] = self.test_df

        with patch(
            'spac.templates.setup_analysis_template.ingest_cells'
        ) as mock_ingest:
            mock_adata = ad.AnnData(X=np.random.rand(10, 5))
            mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
            mock_ingest.return_value = mock_adata

            run_from_json(params_df, save_results=False)

            # Check that DataFrame was passed directly
            call_args = mock_ingest.call_args
            pd.testing.assert_frame_equal(
                call_args[1]['dataframe'], self.test_df
            )

    @patch('builtins.print')
    def test_console_output(self, mock_print) -> None:
        """Test console output messages."""
        with patch(
            'spac.templates.setup_analysis_template.ingest_cells'
        ) as mock_ingest:
            mock_adata = ad.AnnData(X=np.random.rand(10, 5))
            mock_adata.var_names = ['CD25', 'CD3D', 'CD45', 'CD4', 'CD8A']
            mock_ingest.return_value = mock_adata

            run_from_json(self.params, save_results=False)

            # Verify output messages
            print_calls = [
                str(call[0][0]) for call in mock_print.call_args_list
                if call[0]
            ]

            # Should print "Analysis Setup:"
            setup_msgs = [
                msg for msg in print_calls
                if 'Analysis Setup:' in msg
            ]
            self.assertTrue(len(setup_msgs) > 0)

            # Should print "Schema:"
            schema_msgs = [
                msg for msg in print_calls
                if 'Schema:' in msg
            ]
            self.assertTrue(len(schema_msgs) > 0)

    def test_single_feature_and_annotation(self) -> None:
        """Test with single feature and annotation as strings."""
        params_single = self.params.copy()
        params_single["Features_to_Analyze"] = "CD25"
        params_single["Annotation_s_"] = "broad_cell_type"

        with patch(
            'spac.templates.setup_analysis_template.ingest_cells'
        ) as mock_ingest:
            mock_adata = ad.AnnData(X=np.random.rand(10, 1))
            mock_adata.var_names = ['CD25']
            mock_ingest.return_value = mock_adata

            run_from_json(params_single, save_results=False)

            # Check that single values were converted to lists
            call_args = mock_ingest.call_args
            self.assertIn("^CD25$", call_args[1]['regex_str'])
            self.assertEqual(
                call_args[1]['annotation'], ["broad_cell_type"]
            )


if __name__ == "__main__":
    unittest.main()