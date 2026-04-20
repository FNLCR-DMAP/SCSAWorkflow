import unittest

from spac.visualization import _derive_facet_geometry


class TestDeriveFacetGeometry(unittest.TestCase):
    def test_minimal_single_group_defaults(self):
        """Single-group input should keep one column and default geometry."""
        facet_layout = _derive_facet_geometry(
            n_groups=1,
            default_height=3.2,
            default_aspect=1.25,
        )

        self.assertEqual(facet_layout["facet_ncol"], 1)
        self.assertEqual(facet_layout["facet_height"], 3.2)
        self.assertEqual(facet_layout["facet_aspect"], 1.25)

    def test_auto_layout_uses_single_column_below_threshold(self):
        """Check that auto layout selects 1 column when n_groups is at or below threshold."""
        facet_layout = _derive_facet_geometry(
            n_groups=5,
            facet_ncol=None,
            vertical_threshold=5,
        )

        self.assertEqual(facet_layout["facet_ncol"], 1)

    def test_auto_layout_uses_sqrt_rule_above_threshold(self):
        """Auto layout should use sqrt rule when n_groups is above threshold."""
        facet_layout = _derive_facet_geometry(
            n_groups=5,
            facet_ncol=None,
            vertical_threshold=3,
        )

        self.assertEqual(facet_layout["facet_ncol"], 3)

    def test_explicit_column_count_and_figure_size_hints_drive_geometry(self):
        """Explicit facet_ncol should be used directly to compute geometry."""
        facet_layout = _derive_facet_geometry(
            n_groups=5,
            facet_ncol=2,
            vertical_threshold=5,
            facet_fig_width=11,
            facet_fig_height=4,
        )

        self.assertEqual(facet_layout["facet_ncol"], 2)
        self.assertAlmostEqual(facet_layout["facet_height"], 1.6)
        self.assertAlmostEqual(facet_layout["facet_aspect"], 2.0)

    def test_single_figure_size_hint_falls_back_to_defaults(self):
        """A one-sided size hint should not partially derive facet geometry."""
        facet_layout = _derive_facet_geometry(
            n_groups=4,
            facet_fig_width=11,
            vertical_threshold=3,
            default_height=3.2,
            default_aspect=1.25,
        )

        self.assertEqual(facet_layout["facet_ncol"], 2)
        self.assertEqual(facet_layout["facet_height"], 3.2)
        self.assertEqual(facet_layout["facet_aspect"], 1.25)

    def test_none_inputs_fall_back_to_auto_and_default_geometry(self):
        """Missing pre-normalized hints should use auto layout and defaults."""
        facet_layout = _derive_facet_geometry(
            n_groups=3,
            facet_ncol=None,
            facet_fig_width=None,
            facet_fig_height=None,
            vertical_threshold=3,
            default_height=3.2,
            default_aspect=1.25,
        )

        self.assertEqual(facet_layout["facet_ncol"], 1)
        self.assertEqual(facet_layout["facet_height"], 3.2)
        self.assertEqual(facet_layout["facet_aspect"], 1.25)

    def test_explicit_column_count_is_clamped_to_group_count(self):
        """Explicit facet_ncol should be clamped to n_groups if it exceeds it."""
        facet_layout = _derive_facet_geometry(
            n_groups=2,
            facet_ncol=10,
            vertical_threshold=3,
        )

        self.assertEqual(facet_layout["facet_ncol"], 2)


if __name__ == "__main__":
    unittest.main()
