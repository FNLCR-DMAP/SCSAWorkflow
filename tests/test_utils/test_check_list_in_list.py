import unittest
from spac.utils import check_list_in_list


class TestCheckListInList(unittest.TestCase):

    def test_check_list_exist(self):
        # Test when all items in the input list
        # exist in the target list (need_exist=True)
        target_list = ['apple', 'banana', 'orange', 'grape']
        input_list = ['apple', 'banana']
        # No error should be raised since all elements
        # of input_list exist in target_list.
        self.assertIsNone(
                check_list_in_list(
                    input_list,
                    'input_list',
                    'fruit',
                    target_list,
                    need_exist=True
                )
            )

    def test_check_list_not_exist(self):
        # Test when none of the items in the input list
        # exist in the target list (need_exist=True)
        target_list = ['apple', 'banana', 'orange', 'grape']
        input_list = ['mango', 'kiwi']
        # Error should be raised with informative message
        # for each element in the input_list.
        with self.assertRaises(ValueError) as context:
            check_list_in_list(
                    input_list,
                    'input_list',
                    'fruit',
                    target_list,
                    need_exist=True
                )
        self.assertIn(
                "The fruit 'mango' does not exist in the provided dataset.",
                str(context.exception)
            )

    def test_check_list_not_exist_warn_true(self):
        # Test when none of the items in the input list
        # exist in the target list (need_exist=True)
        # and warning = True
        target_list = ['apple', 'banana', 'orange', 'grape']
        input_list = ['mango', 'kiwi']
        # Warning should be raised with informative message
        # for each element in the input_list.
        with self.assertWarns(UserWarning) as context:
            check_list_in_list(
                    input_list,
                    'input_list',
                    'fruit',
                    target_list,
                    need_exist=True,
                    warning=True
                )
        self.assertIn(
                "The fruit 'mango' does not exist in the provided dataset.",
                str(context.warning)
            )

    def test_check_list_exist_fail(self):
        # Test when at least one item in the input list does not exist
        # in the target list (need_exist=False)
        target_list = ['apple', 'banana', 'orange', 'grape']
        input_list = ['apple', 'mango']
        # Error should be raised with informative message
        # for each element in the input_list.
        with self.assertRaises(ValueError) as context:
            check_list_in_list(
                input_list,
                'input_list',
                'fruit',
                target_list,
                need_exist=False
            )
        self.assertIn(
                "The fruit 'apple' exist in the provided dataset.",
                str(context.exception)
            )

    def test_check_list_exist_fail_warning_true(self):
        # Test when at least one item in the input list does not exist
        # in the target list (need_exist=False)
        # and warning = True
        target_list = ['apple', 'banana', 'orange', 'grape']
        input_list = ['apple', 'mango']
        # Warning should be raised with informative message
        # for each element in the input_list.
        with self.assertWarns(UserWarning) as context:
            check_list_in_list(
                input_list,
                'input_list',
                'fruit',
                target_list,
                need_exist=False,
                warning=True
            )
        self.assertIn(
                "The fruit 'apple' exist in the provided dataset.",
                str(context.warning)
            )


    def test_single_string_input(self):
        # Test when a single string is passed as input
        # should be converted to a list)
        target_list = ['apple', 'banana', 'orange', 'grape']
        input_string = 'apple'
        # No error should be raised since the input_string is
        # converted to a list containing only 'apple'.
        self.assertIsNone(
            check_list_in_list(
                input_string,
                'input_string',
                'fruit',
                target_list,
                need_exist=True
                )
            )

    def test_none_input(self):
        # Test when input is None (no validation should be performed)
        target_list = ['apple', 'banana', 'orange', 'grape']
        # No error should be raised since no validation is performed.
        self.assertIsNone(
            check_list_in_list(
                None,
                'input_list',
                'fruit',
                target_list,
                need_exist=True
                )
            )

    def test_invalid_input_type(self):
        # Test when an invalid input type is provided (neither string nor list)
        target_list = ['apple', 'banana', 'orange', 'grape']
        invalid_input = 123
        # Error should be raised with a specific message
        # about the invalid input type.
        with self.assertRaises(ValueError) as context:
            check_list_in_list(
                invalid_input,
                'input_list',
                'fruit',
                target_list,
                need_exist=True
            )
        self.assertIn(
            "The 'input_list' parameter should be a string or",
            str(context.exception)
            )


if __name__ == '__main__':
    unittest.main()
