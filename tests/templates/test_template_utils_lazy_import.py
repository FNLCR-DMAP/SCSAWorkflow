import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _repo_env(repo_root: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return env


def _block_anndata_script(body: str) -> str:
    return textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class BlockAnndata(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "anndata" or fullname.startswith("anndata."):
                    raise ModuleNotFoundError("No module named 'anndata'")
                return None

        sys.meta_path.insert(0, BlockAnndata())

        {body}
        """
    )


def test_template_utils_import_does_not_require_anndata():
    repo_root = Path(__file__).resolve().parents[2]

    script = _block_anndata_script(
        """
        from spac.templates.template_utils import parse_params

        assert parse_params({"ok": True})["ok"] is True
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=_repo_env(repo_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_anndata_pickle_without_anndata_has_clear_error(tmp_path):
    import pickle

    import anndata as ad
    import numpy as np

    repo_root = Path(__file__).resolve().parents[2]
    pickle_path = tmp_path / "adata.pickle"
    with pickle_path.open("wb") as fh:
        pickle.dump(ad.AnnData(X=np.ones((2, 2), dtype=np.float32)), fh)

    script = _block_anndata_script(
        f"""
        from spac.templates.template_utils import load_input

        try:
            load_input({str(pickle_path)!r})
        except ImportError as e:
            assert "anndata package required to read AnnData pickle files" in str(e)
        else:
            raise AssertionError("expected ImportError")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=_repo_env(repo_root),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
