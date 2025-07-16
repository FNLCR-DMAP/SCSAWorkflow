# tests/templates/test_ripley_l_template.py
"""Unit‑tests for the Ripley‑L template."""

import json
import os
import pickle
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd

sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) + "/../../src"
)

from spac.templates.ripley_l_template import run_from_json


def mock_adata(n_cells: int = 40) -> ad.AnnData:
    """Return a tiny synthetic AnnData for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "renamed_phenotypes": np.where(
                rng.random(n_cells) > 0.5, "B cells", "CD8 T cells"
            )
        }
    )
    x_mat = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=x_mat, obs=obs)
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 300.0
    return adata


class TestRipleyLTemplate(unittest.TestCase):
    """Light‑weight sanity checks for the Ripley‑L template."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.in_file = os.path.join(self.tmp_dir.name, "input.pickle")
        self.out_file = os.path.join(self.tmp_dir.name, "output.pickle")

        with open(self.in_file, "wb") as handle:
            pickle.dump(mock_adata(), handle)

        self.params = {
            "input_data": self.in_file,
            "radii": [0, 50, 100],
            "annotation": "renamed_phenotypes",
            "phenotypes": ["B cells", "CD8 T cells"],
            "output_path": self.out_file,
        }

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_run_from_dict(self) -> None:
        """run_from_json accepts dict parameters."""
        adata = run_from_json(self.params)
        self.assertIn("ripley_l_results", adata.uns)

    def test_run_from_json_file(self) -> None:
        """run_from_json accepts a JSON file path."""
        json_path = os.path.join(self.tmp_dir.name, "params.json")
        with open(json_path, "w") as handle:
            json.dump(self.params, handle)
        adata = run_from_json(json_path)
        self.assertTrue(os.path.exists(self.out_file))
        self.assertIn("ripley_l_results", adata.uns)


if __name__ == "__main__":
    unittest.main()
