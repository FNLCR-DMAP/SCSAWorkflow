import unittest
import anndata
import numpy as np
import pandas as pd
from spac.data_utils import add_pin_color_rules


class TestAddPinColorRules(unittest.TestCase):
    def setUp(self):
        # Create a simple AnnData object for testing
        obs = pd.DataFrame(
            {
                "metadata1": ["label1", "label2", "label_all"],
                "metadata2": ["label4", "label5", "label_all"]
            },
            index=["cell1", "cell2", "cell3"],
        )
        var = pd.DataFrame(
            {
                "col1": ["value1", "label_all"],
                "col2": ["value2", "label_all"]
            },
            index=["gene1", "gene2"],
        )
        X = np.array([[1, 2], [3, 4], [5, 6]])
        self.adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Create a simple label color dictionary for testing
        self.label_color_dict = {
            "label1": "red",
            "label2": "green",
            "label3": "blue",
            "label4": "yellow",
            "label5": "purple",
            "label6": "orange",
            "label7": "pink",
            "label8": "brown"
        }

    def test_add_pin_color_rules(self):
        # Test that the function raises a ValueError when overwrite is False
        # and the color_map_name already exists in adata.uns
        self.adata.uns["test_color_map"] = {}
        with self.assertRaises(ValueError):
            add_pin_color_rules(
                self.adata,
                self.label_color_dict,
                "test_color_map",
                overwrite=False
            )

    def test_add_pin_color_rules_overwrite(self):
        # Test that the function overwrites
        # existing color map when overwrite is True
        self.adata.uns["test_color_map"] = {"old_label": "old_color"}
        add_pin_color_rules(
            self.adata,
            self.label_color_dict,
            "test_color_map",
            overwrite=True
        )
        self.assertEqual(
            self.adata.uns["test_color_map"],
            self.label_color_dict
        )

    def test_add_pin_color_rules_new_map(self):
        # Test that the function adds a new color map when color_map_name
        # does not exist in adata.uns
        add_pin_color_rules(
            self.adata,
            self.label_color_dict,
            "new_color_map",
            overwrite=False
        )
        self.assertEqual(
            self.adata.uns["new_color_map"],
            self.label_color_dict
        )

        # Test that adata.uns contains "new_color_map_summary"
        print(self.adata.uns)
        self.assertIn("new_color_map_summary", self.adata.uns)

    def test_add_pin_color_rules_empty_dict(self):
        # Test that the function handles an empty
        # label color dictionary correctly
        add_pin_color_rules(
            self.adata,
            {},
            "empty_color_map",
            overwrite=False
        )
        self.assertEqual(
            self.adata.uns["empty_color_map"],
            {}
        )

    def test_add_pin_color_rules_no_labels(self):
        # Test that the function handles a label color
        # dictionary with labels not present in adata correctly
        add_pin_color_rules(
            self.adata,
            {"nonexistent_label": "color"},
            "nonexistent_color_map",
            overwrite=False
        )
        self.assertEqual(
            self.adata.uns["nonexistent_color_map"],
            {"nonexistent_label": "color"}
        )

    def test_add_pin_color_rules_label_in_all_fields(self):
        label_color_dict = {"label_all": "red"}
        color_map_name = "test_color_map"
        label_matches, _ = add_pin_color_rules(
            self.adata, label_color_dict, color_map_name, overwrite=True
        )
        expected_label_matches = {
            "obs": {"metadata1": ["label_all"], "metadata2": ["label_all"]},
            "var": {"col1": ["label_all"], "col2": ["label_all"]},
            "X": {"column_names": []},
        }
        self.assertEqual(label_matches, expected_label_matches)


if __name__ == "__main__":
    unittest.main()
