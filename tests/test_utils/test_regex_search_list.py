import unittest
from spac.data_utils import regex_search_list


class RegexSearchListTests(unittest.TestCase):
    def test_single_pattern_single_match(self):
        regex_pattern = "AB.*"
        list_to_search = ["ABC", "BC", "AC"]
        expected_result = ["ABC"]
        result = regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(result, expected_result)

    def test_single_pattern_multiple_matches(self):
        regex_pattern = "A.*"
        list_to_search = ["ABC", "BC", "ACD", "CA", "BA", "CAB"]
        expected_result = ["ABC", "ACD"]
        result = regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(result, expected_result)

    def test_multiple_patterns_single_match_each(self):
        regex_pattern = ["A", "^B.*"]
        list_to_search = ["ABC", "BC", "AC", "AB"]
        expected_result = ["BC"]
        result = regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(result, expected_result)

    def test_multiple_patterns_multiple_matches_each(self):
        regex_pattern = ["A", "^B.*"]
        list_to_search = ["ABC", "BC", "ACD", "CA", "BA", "CAB"]
        expected_result = ['BC', 'BA']
        result = regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(result, expected_result)

    def test_empty_list_to_search(self):
        regex_pattern = "A"
        list_to_search = []
        expected_result = []
        result = regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(result, expected_result)

    def test_empty_regex_pattern_list(self):
        regex_pattern = []
        list_to_search = ["ABC", "BC", "AC"]
        expected_result = []
        result = regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(result, expected_result)

    def test_non_list_to_search_raises_error(self):
        regex_pattern = "A"
        list_to_search = "ABC"
        with self.assertRaises(TypeError):
            regex_search_list(regex_pattern, list_to_search)

    def test_non_list_non_string_regex_pattern_raises_error(self):
        regex_pattern = 123
        list_to_search = ["ABC", "BC", "AC"]
        with self.assertRaises(TypeError):
            regex_search_list(regex_pattern, list_to_search)

    def test_non_string_items_in_regex_pattern_list_raises_error(self):
        regex_pattern = ["A", 123]
        list_to_search = ["ABC", "BC", "AC"]
        with self.assertRaises(TypeError):
            regex_search_list(regex_pattern, list_to_search)

    def test_exact_matches(self):
        regex_pattern = "^ABCD$"
        list_to_search = ["ABCD", "ABCD1", "ABCD2", "ABCD3", "ABCD4"]
        expected_result = ["ABCD"]
        result = regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(result, expected_result)

    def test_invalid_search_pattern(self):
        regex_pattern = "*ABCD$"
        list_to_search = ["ABCD", "ABCD1", "ABCD2", "ABCD3", "ABCD4"]
        expected_error_msg = (
            "Error occurred when searching with regex:\n"
            "nothing to repeat at position 0\nPlease review "
            "your regex pattern: *ABCD$\nIf using * at the "
            "start, always have a . before the asterisk."
                )
        with self.assertRaises(ValueError) as context:
            regex_search_list(regex_pattern, list_to_search)
        self.assertEqual(str(context.exception), expected_error_msg)


if __name__ == '__main__':
    unittest.main()
