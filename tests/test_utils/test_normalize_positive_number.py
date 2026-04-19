import unittest

from spac.utils import normalize_positive_number


class TestNormalizePositiveNumber(unittest.TestCase):
    def test_float_conversion(self):
        self.assertEqual(
            normalize_positive_number("11.5", convert_to="float"),
            11.5,
        )

    def test_int_conversion(self):
        self.assertEqual(
            normalize_positive_number("3", convert_to="int"),
            3,
        )

    def test_default_like_values_return_none(self):
        self.assertIsNone(normalize_positive_number("auto", convert_to="int"))
        self.assertIsNone(normalize_positive_number("None", convert_to="float"))
        self.assertIsNone(normalize_positive_number(None, convert_to="float"))

    def test_invalid_or_non_positive_values_return_none(self):
        self.assertIsNone(normalize_positive_number("bad", convert_to="int"))
        self.assertIsNone(normalize_positive_number("-1", convert_to="float"))
        self.assertIsNone(normalize_positive_number(0, convert_to="float"))

    def test_sanitized_inputs_are_logged(self):
        with self.assertLogs("spac.utils", level="INFO") as logs:
            self.assertIsNone(
                normalize_positive_number(
                    "auto",
                    var_name="facet_ncol",
                    convert_to="int",
                )
            )

        self.assertTrue(
            any("facet_ncol='auto'" in message for message in logs.output)
        )

    def test_invalid_inputs_are_logged_as_warning(self):
        with self.assertLogs("spac.utils", level="WARNING") as logs:
            self.assertIsNone(
                normalize_positive_number(
                    "bad",
                    var_name="facet_fig_width",
                    convert_to="float",
                )
            )

        self.assertTrue(
            any("facet_fig_width='bad'" in message for message in logs.output)
        )


if __name__ == "__main__":
    unittest.main()
