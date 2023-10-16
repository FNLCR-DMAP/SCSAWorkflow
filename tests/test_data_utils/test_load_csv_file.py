import os
import unittest
import pandas as pd
from unittest.mock import patch
from spac.data_utils import load_csv_files


class TestLoadCSVFiles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.valid_file = "valid.csv"
        cls.invalid_file = "invalid.csv"
        cls.empty_file = "empty.csv"
        cls.unreadable_file = "unreadable.csv"
        cls.mismatch_file = "mismatch.csv"

        # Create valid csv file
        data = {'column1': [1, 2], 'column2': [3, 4]}
        df = pd.DataFrame(data)
        df.to_csv(cls.valid_file, index=False)

        # Create empty csv file
        with open(cls.empty_file, "w") as f:
            pass

        # Create csv file with mismatched columns
        data = {'different_column': [1, 2], 'column2': [3, 4]}
        df = pd.DataFrame(data)
        df.to_csv(cls.mismatch_file, index=False)

        # Create unreadable csv file
        with open(cls.unreadable_file, "w") as f:
            f.write("column1,column2\n1,2\n")
        os.chmod(cls.unreadable_file, 0o200)  # Remove read permission

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.valid_file)
        os.remove(cls.empty_file)
        os.remove(cls.mismatch_file)
        os.chmod(cls.unreadable_file, 0o600)  # Restore read permission
        os.remove(cls.unreadable_file)

    def test_load_single_csv_file(self):
        result = load_csv_files(self.valid_file)
        self.assertIsInstance(result, pd.DataFrame)

    def test_load_multiple_csv_files(self):
        result = load_csv_files([self.valid_file, self.valid_file])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)

    def test_invalid_file_type(self):
        with self.assertRaises(TypeError):
            load_csv_files(42)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_csv_files(self.invalid_file)

    def test_empty_file(self):
        with self.assertRaises(TypeError):
            load_csv_files(self.empty_file)

    def test_unreadable_file(self):
        with patch("os.access") as mock_access:
            mock_access.return_value = False  # Simulate unreadable file
            with self.assertRaises(PermissionError):
                load_csv_files(self.unreadable_file)


if __name__ == "__main__":
    unittest.main()
