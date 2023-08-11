import os
import unittest
import pandas as pd
from spac.data_utils import combine_dfs


class TestCombineDFs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create sample dataframes
        cls.df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        cls.df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

        # Create annotations dataframe
        cls.annotations = pd.DataFrame(
            {"annotation1": [10, 20]},
            index=["file1.csv", "file2.csv"]
            )

        # Save sample dataframes to temporary CSV files
        cls.file1_path = "file1.csv"
        cls.file2_path = "file2.csv"
        cls.df1.to_csv(cls.file1_path, index=False)
        cls.df2.to_csv(cls.file2_path, index=False)

        cls.dataframes = [
            [cls.file1_path, cls.df1],
            [cls.file2_path, cls.df2]
        ]

    def test_combine_dfs(self):
        combined_df = combine_dfs(self.dataframes, self.annotations)
        expected_df = pd.DataFrame(
            {"A": [1, 2, 5, 6],
             "B": [3, 4, 7, 8],
             "annotation1": [10, 10, 20, 20]}
             )
        pd.testing.assert_frame_equal(combined_df, expected_df)

    def test_combine_dfs_wrong_annotations_type(self):
        with self.assertRaises(TypeError):
            combine_dfs(self.dataframes, "wrong_type")

    def test_combine_dfs_missing_annotation(self):
        annotations_missing_data = self.annotations.drop("file1.csv")
        with self.assertRaises(ValueError):
            combine_dfs(self.dataframes, annotations_missing_data)

    @classmethod
    def tearDownClass(cls):
        # Remove temporary CSV files
        os.remove(cls.file1_path)
        os.remove(cls.file2_path)


if __name__ == "__main__":
    unittest.main()
