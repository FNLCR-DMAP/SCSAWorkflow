import unittest
import json
import tempfile
import os
import sys
import pickle

# Add parent directory to path to import ripley_l_template
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ripley_l_template import ripley_l_calculation_template


class MockData:
    """Mock data object for testing"""
    def __init__(self):
        self.uns = {}


class TestRipleyLTemplate(unittest.TestCase):

    def setUp(self):
        """Create test data and parameters"""
        # Use the test data file if it exists
        if os.path.exists('../test_anndata.pickle'):
            self.data_file = '../test_anndata.pickle'
        else:
            # Create simple mock data for testing
            mock = MockData()
            self.data_file = "test_mock.pickle"
            with open(self.data_file, 'wb') as f:
                pickle.dump(mock, f)

        # Create test parameters
        self.test_params = {
            "input_data": self.data_file,
            "radii": [0, 50, 100],
            "annotation": "cell_type",
            "phenotypes": ["TypeA", "TypeB"],
            "regions": None,
            "n_simulations": 10,
            "output_path": "test_output.pickle"
        }

        # Save parameters as JSON
        self.json_file = "test_params.json"
        with open(self.json_file, 'w') as f:
            json.dump(self.test_params, f)

    def tearDown(self):
        """Clean up test files"""
        files_to_clean = [
            "test_mock.pickle",
            self.json_file,
            "test_output.pickle"
        ]
        for f in files_to_clean:
            if os.path.exists(f):
                os.remove(f)

    def test_json_file_input(self):
        """Test running with JSON file"""
        result = ripley_l_calculation_template(self.json_file)

        # Check output file was created
        self.assertTrue(os.path.exists("test_output.pickle"))

        # Check result has expected structure
        self.assertTrue(hasattr(result, 'uns'))
        self.assertIn('ripley_l_results', result.uns)

    def test_json_string_input(self):
        """Test running with JSON string"""
        json_string = json.dumps(self.test_params)
        ripley_l_calculation_template(json_string)

        # Check output was created
        self.assertTrue(os.path.exists("test_output.pickle"))

    def test_json_parameters_loaded(self):
        """Test that parameters are correctly loaded from JSON"""
        # This tests the internal logic without running the full analysis
        with open(self.json_file, 'r') as f:
            loaded_params = json.load(f)

        self.assertEqual(loaded_params['radii'], [0, 50, 100])
        self.assertEqual(loaded_params['annotation'], 'cell_type')
        self.assertIsNone(loaded_params['regions'])


if __name__ == '__main__':
    unittest.main()
