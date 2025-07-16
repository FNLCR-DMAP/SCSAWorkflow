import json, pickle, unittest
from pathlib import Path
import tempfile

from spac.templates.ripley_l_template import run_from_json
from tests.templates._fixtures import mock_adata


class TestRipleyLTemplate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.in_file = Path(self.tmpdir.name) / "input.pickle"
        self.out_file = Path(self.tmpdir.name) / "output.pickle"

        with self.in_file.open("wb") as fh:
            pickle.dump(mock_adata(), fh)

        self.params = {
            "input_data": str(self.in_file),
            "radii": [0, 50, 100],
            "annotation": "renamed_phenotypes",
            "phenotypes": ["B cells", "CD8 T cells"],
            "output_path": str(self.out_file),
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_from_dict(self):
        adata = run_from_json(self.params)
        self.assertIn("ripley_l_results", adata.uns)

    def test_run_from_json_file(self):
        json_file = Path(self.tmpdir.name) / "p.json"
        json_file.write_text(json.dumps(self.params))
        adata = run_from_json(json_file)
        self.assertTrue(self.out_file.exists())
        self.assertIn("ripley_l_results", adata.uns)

