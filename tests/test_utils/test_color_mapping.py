import unittest
from spac.utils import color_mapping


class TestColorMapping(unittest.TestCase):

    def test_continuous_colormap(self):
        labels = ['A', 'B', 'C', 'D', 'E']
        colors = color_mapping(labels, 'viridis', 1.0)
        self.assertEqual(len(colors), len(labels))

    def test_one_label(self):
        labels = ['A']
        colors = color_mapping(labels, 'viridis', 1.0)
        self.assertEqual(len(colors), len(labels))

    def test_discrete_colormap(self):
        labels = ['A', 'B', 'C', 'D', 'E']
        colors = color_mapping(labels, 'Set3', 1.0)
        self.assertEqual(len(colors), len(labels))

    def test_opacity(self):
        labels = ['A', 'B', 'C', 'D', 'E']
        colors = color_mapping(labels, 'viridis', 0.5)
        self.assertTrue(
            all(color.endswith('0.5)') for color in colors)
        )

    def test_invalid_colormap(self):
        labels = ['A', 'B', 'C', 'D', 'E']
        with self.assertRaisesRegex(
            ValueError, "Invalid color map name: invalid_colormap"
        ):
            color_mapping(labels, 'invalid_colormap', 1.0)

    def test_invalid_opacity(self):
        labels = ['A', 'B', 'C', 'D', 'E']
        with self.assertRaisesRegex(
            ValueError, "Opacity must be between 0 and 1"
        ):
            color_mapping(labels, 'viridis', -1.0)

        with self.assertRaisesRegex(
            ValueError, "Opacity must be between 0 and 1"
        ):
            color_mapping(labels, 'viridis', 2.0)


if __name__ == '__main__':
    unittest.main()
