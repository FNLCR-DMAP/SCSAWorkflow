import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../src")

from spac.data_utils import combine_csvs


class TestCombineCSVs(unittest.TestCase):

    def setUp(self):
        self.csv1 = 'test1.csv'
        self.csv2 = 'test2.csv'
        self.df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        self.df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        self.df1.to_csv(self.csv1, index=False)
        self.df2.to_csv(self.csv2, index=False)
        self.observations = pd.DataFrame({'Obs1': [10, 20], 'Obs2': [30, 40]}, index=[self.csv1, self.csv2])

    def tearDown(self):
        os.remove(self.csv1)
        os.remove(self.csv2)

    def test_combine_csvs(self):
        combined_df = combine_csvs([self.csv1, self.csv2], self.observations, nidap=False)
        expected_df = pd.DataFrame({'A': [1, 2, 5, 6], 'B': [3, 4, 7, 8], 'Obs1': [10, 10, 20, 20], 'Obs2': [30, 30, 40, 40]})
        # Reset index of expected_df
        expected_df.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(combined_df, expected_df, check_like=True)

    def test_wrong_file_names_type(self):
        with self.assertRaises(TypeError):
            combine_csvs(123, self.observations, nidap=False)

    def test_wrong_observations_type(self):
        with self.assertRaises(TypeError):
            combine_csvs([self.csv1, self.csv2], "not_a_dataframe", nidap=False)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            combine_csvs(['non_existent_file.csv'], self.observations, nidap=False)

    def test_invalid_columns(self):
        self.df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
        self.df2.to_csv(self.csv2, index=False)
        with self.assertRaises(ValueError):
            combine_csvs([self.csv1, self.csv2], self.observations, nidap=False)

    def test_missing_observations(self):
        missing_obs = pd.DataFrame({'Obs1': [10]}, index=[self.csv1])
        with self.assertRaises(ValueError):
            combine_csvs([self.csv1, self.csv2], missing_obs, nidap=False)


if __name__ == '__main__':
    unittest.main()