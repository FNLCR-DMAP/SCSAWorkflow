import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")
import unittest
import re # Import re if used directly in tests, though _css_rgb_or_hex_to_hex encapsulates its re usage
import matplotlib.colors as mcolors
from spac.visualization import _css_rgb_or_hex_to_hex

class TestCssRgbOrHexToHex(unittest.TestCase):
    """
    Test suite for the _css_rgb_or_hex_to_hex function,
    focusing on major features and error handling.
    """

    def test_valid_hex_colors(self):
        """
        Test valid hex color conversions:
        - 6-digit (lowercase, uppercase, mixed) to lowercase.
        - 3-digit to 6-digit lowercase.
        - 8-digit (with alpha) to 8-digit or 6-digit lowercase.
        """
        self.assertEqual(_css_rgb_or_hex_to_hex('#ff0000'), '#ff0000')
        self.assertEqual(_css_rgb_or_hex_to_hex('#FF00AA'), '#ff00aa')
        self.assertEqual(_css_rgb_or_hex_to_hex('#Ff00aA'), '#ff00aa')
        # 3-digit hex
        self.assertEqual(_css_rgb_or_hex_to_hex('#f0a'), '#ff00aa')
        # 8-digit hex with alpha
        self.assertEqual(
            _css_rgb_or_hex_to_hex('#ff00aa80', keep_alpha=True),
            '#ff00aa80'
        )
        self.assertEqual(
            _css_rgb_or_hex_to_hex('#FF00AA80', keep_alpha=False),
            '#ff00aa'
        )

    # ---------- named colours -------------------------------------------

    def test_named_colour_passthrough(self):
        self.assertEqual(_css_rgb_or_hex_to_hex('gold'), 'gold')
        self.assertEqual(_css_rgb_or_hex_to_hex(' GOLd '), 'gold')

    # ---------- hexadecimal forms ---------------------------------------

    def test_short_and_long_hex(self):
        self.assertEqual(_css_rgb_or_hex_to_hex('#ABC'), '#aabbcc')
        self.assertEqual(_css_rgb_or_hex_to_hex('#a1b2c3'), '#a1b2c3')

    def test_keep_alpha(self):
        # 8-digit input keeps alpha → unchanged
        self.assertEqual(
            _css_rgb_or_hex_to_hex('#ff000080', keep_alpha=True),
            '#ff000080',
        )
        # when keep_alpha is False alpha is stripped
        self.assertEqual(
            _css_rgb_or_hex_to_hex('#ff000080', keep_alpha=False),
            '#ff0000',
        )

    # ---------- rgb()/rgba() strings ------------------------------------

    def test_rgb_to_hex(self):
        self.assertEqual(
            _css_rgb_or_hex_to_hex('rgb(255,0,0)'),
            '#ff0000',
        )

    def test_rgba_to_hex_with_alpha(self):
        self.assertEqual(
            _css_rgb_or_hex_to_hex('rgba(255,0,0,0.5)', keep_alpha=True),
            '#ff000080',                        # 0.5 → 0x80
        )
        # keep_alpha False strips alpha
        self.assertEqual(
            _css_rgb_or_hex_to_hex('rgba(255,0,0,0.5)', keep_alpha=False),
            '#ff0000',
        )

    # ---------- error handling ------------------------------------------

    def test_out_of_range_raises_value_error(self):
        """RGB components >255 should trigger a descriptive ValueError."""
        with self.assertRaisesRegex(
            ValueError,
            r'RGB components in ".+" must be between 0 and 255'
        ):
            _css_rgb_or_hex_to_hex('rgb(300,0,0)')


if __name__ == '__main__':
    unittest.main()