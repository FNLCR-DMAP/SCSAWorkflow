import io
import unittest
from unittest.mock import patch
import pandas as pd
from spac.utils import get_defined_color_map


class DummyAnnData:
    def __init__(self, uns, obs=None):
        self.uns = uns
        self.obs = obs or {}


class TestGetDefinedColorMap(unittest.TestCase):

    def test_valid_color_map(self):
        """
        Test that a valid defined color map key returns the
        corresponding dictionary and prints the correct message.
        """
        dummy = DummyAnnData(uns={'my_map': {'a': 'red', 'b': 'blue'}})
        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            result = get_defined_color_map(dummy, 'my_map')
            printed = fake_out.getvalue().strip()
        expected_print = (
            'Selected color mapping "my_map":\n'
            + str(dummy.uns['my_map'])
        )
        self.assertEqual(result, {'a': 'red', 'b': 'blue'})
        self.assertEqual(printed, expected_print)

    def test_non_string_map_key(self):
        """
        Test that passing a non-string as defined_color_map raises a
        TypeError with the correct message.
        """
        dummy = DummyAnnData(uns={'my_map': {'a': 'red'}})
        with self.assertRaisesRegex(
            TypeError,
            (r'The "defined_color_map" should be a string, '
             r'getting <class \'int\'>\.')
        ):
            get_defined_color_map(dummy, 123)

    def test_empty_uns_keys(self):
        """
        Test that an empty uns raises a ValueError with the correct
        message.
        """
        dummy = DummyAnnData(uns={})
        with self.assertRaisesRegex(
            ValueError,
            (r'No existing color map found\. Please make sure the '
             r'Append Pin Color Rules template has been run prior to '
             r'the current visualization node\.')
        ):
            get_defined_color_map(dummy, 'my_map')

    def test_map_key_not_found(self):
        """
        Test that a missing color map key raises a ValueError with the
        correct message.
        """
        dummy = DummyAnnData(uns={'other_map': {'a': 'red'}})
        with self.assertRaisesRegex(
            ValueError,
            (r'The given color map name: my_map is not found '
             r'in current analysis\. Available items are: \[\'other_map\'\]')
        ):
            get_defined_color_map(dummy, 'my_map')

    def test_generate_color_map(self):
        """
        Test that if defined_color_map is not provided and annotations is,
        a color mapping is generated using unique labels from a pandas Series.
        """
        # Simulate anndata.obs using a pandas Series.
        obs = {'my_ann': pd.Series(['a', 'b', 'a'])}
        dummy = DummyAnnData(uns={'dummy': {}}, obs=obs)
        # Call the actual function to generate the color map.
        result = get_defined_color_map(dummy,
                                       defined_color_map=None,
                                       annotations='my_ann',
                                       colorscale='viridis')
        # Check that the result contains the expected keys.
        self.assertIn('a', result)
        self.assertIn('b', result)
        # Check that the colors are correctly generated.
        self.assertTrue(all(isinstance(color, str) for color in result.values()))

    def test_missing_annotations(self):
        """
        Test that if defined_color_map is None and annotations is not provided,
        a ValueError is raised.
        """
        dummy = DummyAnnData(uns={'dummy': {}})
        with self.assertRaisesRegex(
            ValueError,
            r'Either a defined color map must be provided, or an '
            r'annotation column must be specified\.'
        ):
            get_defined_color_map(dummy, defined_color_map=None)


if __name__ == '__main__':
    unittest.main()
