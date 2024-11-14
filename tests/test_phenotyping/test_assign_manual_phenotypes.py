import unittest
import pandas as pd
from spac.phenotyping import assign_manual_phenotypes


class TestAssignManualPhenotypes(unittest.TestCase):
    def setUp(self):
        """Set up for testing select_values with both DataFrame and AnnData."""
        # DataFrame setup with values 'A', 'B', 'C'
        self.binary_df = pd.DataFrame({
            'cd4_phenotype': [0, 1, 0, 1],
            'cd8_phenotype': [0, 0, 1, 1]
        })

        self.string_df = pd.DataFrame({
            'cd4_phenotype': ["cd4-", "cd4+", "cd4-", "cd4+"],
            'cd8_phenotype': ["cd8-", "cd8-", "cd8+", "cd8+"]
        })

        name = "phenotype_name"
        code = "phenotype_code"
        cd4 = {name: "cd4_cells", code: "cd4+cd8-"}
        cd8 = {name: "cd8_cells", code: "cd8+"}
        cd4_cd8 = {name: "cd4_cd8", code: "cd4+cd8+"}

        all_phenotypes = [cd4, cd8, cd4_cd8]

        self.phenotypes_df = pd.DataFrame(all_phenotypes)

    def test_binary_labels(self):
        """
        Generate correct phenotypes using data frames with binary
        labels
        """
        assign_manual_phenotypes(
            self.binary_df,
            self.phenotypes_df,
            annotation="manual",
            prefix='',
            suffix='_phenotype',
            multiple=False
        )

        # Check that string_df has the correct manual phenotypes

        self.assertEqual(self.binary_df['manual'].tolist(),
                         ['no_label', 'cd4_cells', 'cd8_cells', 'no_label'])

    def test_string_labels(self):
        """
        Generate correct phenotypes using data frames with string
        labels
        """
        assign_manual_phenotypes(
            self.string_df,
            self.phenotypes_df,
            annotation="manual",
            prefix='',
            suffix='_phenotype',
            multiple=False
        )

        # Check that string_df has the correct manual phenotypes

        self.assertEqual(self.string_df['manual'].tolist(),
                         ['no_label', 'cd4_cells', 'cd8_cells', 'no_label'])

    def test_multiple(self):
        """
        Generate correct behavior when a cell has multiple phenotypes.
        """
        assign_manual_phenotypes(
            self.string_df,
            self.phenotypes_df,
            annotation="manual",
            prefix='',
            suffix='_phenotype',
            multiple=True
        )

        # Check that string_df has the correct manual phenotypes
        self.assertEqual(
            self.string_df['manual'].tolist(),
            ['no_label', 'cd4_cells', 'cd8_cells', 'cd8_cells, cd4_cd8']
        )

    def test_returned_dic(self):
        """
        Generate correct behavior when a cell has multiple phenotypes.
        """
        returned_dic = assign_manual_phenotypes(
            self.string_df,
            self.phenotypes_df,
            annotation="manual",
            prefix='',
            suffix='_phenotype',
            multiple=True
        )

        # Check that returned dic has the keys
        # "assigned_phenotype_counts" and "multiple_phenotypes_summary"
        # and phenotypes_counts

        self.assertEqual(
            list(returned_dic.keys()),
            [
                "assigned_phenotype_counts",
                "multiple_phenotypes_summary",
                "phenotypes_counts"
            ]
        )

        # Test assigned phenotype counts is correct
        expected_assigned_phenotype_counts = \
            pd.Series({0.0: 1, 1.0: 2, 2.0: 1})
        self.assertTrue(
            returned_dic["assigned_phenotype_counts"].equals(
                expected_assigned_phenotype_counts)
            )

        expected_mulitple_phenotypes_summary = pd.DataFrame({
            'manual': ['cd8_cells, cd4_cd8'],
            'count': [1]
        })
        pd.testing.assert_frame_equal(
            returned_dic["multiple_phenotypes_summary"],
            expected_mulitple_phenotypes_summary
        )

        expected_phenotypes_counts = {
            'cd4_cells': 1,
            'cd8_cells': 2,
            'cd4_cd8': 1
        }

        self.assertDictEqual(
            returned_dic["phenotypes_counts"],
            expected_phenotypes_counts
        )

    def test_code_with_no_matching_column(self):
        """
        Test exction is the code for the phenotype is not in the
        phenotype dataframe
        """
        name = "phenotype_name"
        code = "phenotype_code"
        # cd5 is not a column in the dataframe
        cd4 = {name: "cell_type", code: "cd5+"}
        phenotypes_df = pd.DataFrame([cd4])

        with self.assertRaises(ValueError) as cm:
            assign_manual_phenotypes(
                self.string_df,
                phenotypes_df,
                annotation="manual",
                prefix='',
                suffix='_phenotype',
                multiple=True
            )

        expect_string = (
            """The feature "cd5_phenotype" does not exist in the """
            """input table. Existing columns are """
            """"['cd4_phenotype', 'cd8_phenotype']" """
        )
        self.assertEqual(str(cm.exception), expect_string.strip())

    def test_suffix(self):
        """
        Test an error message where suffix is not correct
        """
        name = "phenotype_name"
        code = "phenotype_code"
        # cd5 is not a column in the dataframe
        cd4 = {name: "cell_type", code: "cd4+"}
        phenotypes_df = pd.DataFrame([cd4])

        with self.assertRaises(ValueError) as cm:
            assign_manual_phenotypes(
                self.string_df,
                phenotypes_df,
                annotation="manual",
                prefix='',
                suffix='_wrong_suffix',
                multiple=True
            )

        expect_string = (
            """The feature "cd4_wrong_suffix" does not exist in the """
            """input table. Existing columns are """
            """"['cd4_phenotype', 'cd8_phenotype']" """
        )
        self.assertEqual(str(cm.exception), expect_string.strip())


    def test_prefix(self):
        """
        Test an error message where prefix is not correct
        """
        name = "phenotype_name"
        code = "phenotype_code"
        # cd5 is not a column in the dataframe
        cd4 = {name: "cell_type", code: "cd4+"}
        phenotypes_df = pd.DataFrame([cd4])

        with self.assertRaises(ValueError) as cm:
            assign_manual_phenotypes(
                self.string_df,
                phenotypes_df,
                annotation="manual",
                prefix='wrong_prefix_',
                suffix='_phenotype',
                multiple=True
            )

        expect_string = (
            """The feature "wrong_prefix_cd4_phenotype" does not exist in the """
            """input table. Existing columns are """
            """"['cd4_phenotype', 'cd8_phenotype']" """
        )
        self.assertEqual(str(cm.exception), expect_string.strip())





    def test_code_with_no_positive_or_negative_sign(self):
        """
        The phenotype code does not have a '+' or '-'
        """
        name = "phenotype_name"
        code = "phenotype_code"
        # cd8 is missing a '+' or '-'
        cd4 = {name: "cell_type", code: "cd4+cd8"}
        phenotypes_df = pd.DataFrame([cd4])

        with self.assertRaises(ValueError) as cm:
            assign_manual_phenotypes(
                self.string_df,
                phenotypes_df,
                annotation="manual",
                prefix='',
                suffix='_phenotype',
                multiple=True
            )

        expect_string = (
            """The passed phenotype code "cd4+cd8" """
            """should end with "+" or "-" """
        )
        self.assertEqual(str(cm.exception), expect_string.strip())


if __name__ == '__main__':
    unittest.main()
