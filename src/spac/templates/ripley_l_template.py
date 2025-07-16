"""
Platform‑agnostic Ripley‑L template.

Usage
-----
>>> from spac.templates.ripley_l_template import run_from_json
>>> run_from_json("examples/ripley_l_params.json")
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Union

import anndata as ad
from spac.utils import check_table
from spac.spatial_analysis import ripley_l

from .template_utils import load_pickle, save_pickle


# public API
def run_from_json(src: Union[str, Path, Dict[str, Any]]) -> ad.AnnData:
    """
    Execute Ripley‑L with parameters supplied via *src*.

    Parameters
    ----------
    src
        • path to a params JSON file
        • raw JSON string
        • already‑parsed dict
    Returns
    -------
    AnnData
        Same AnnData with ``uns['ripley_l_results']`` attached.
    """
    params = _parse_params(src)
    _validate(params)

    adata = load_pickle(params["input_data"])
    check_table(adata)

    # analysis
    result_df = ripley_l(
        adata,
        annotation=params["annotation"],
        phenotypes=params["phenotypes"],
        distances=params["radii"],
        regions=params["regions"],
        n_simulations=params["n_simulations"],
        area=params["area"],
        seed=params["seed"],
        spatial_key=params["spatial_key"],
        edge_correction=params["edge_correction"],
    )
    adata.uns["ripley_l_results"] = result_df

    save_pickle(adata, params["output_path"])
    return adata


# helpers
def _parse_params(src: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(src, dict):
        return src
    if isinstance(src, (str, Path)):
        text = Path(src).read_text() if str(src).endswith(".json") else src
        return json.loads(text)
    raise TypeError("src must be dict, JSON string, or path to JSON")


def _validate(p: Dict[str, Any]) -> None:
    """Fill defaults and enforce types."""
    required = ["input_data", "radii", "annotation", "phenotypes"]
    missing = [k for k in required if k not in p]
    if missing:
        raise ValueError(f"missing required keys: {missing}")

    if not (isinstance(p["radii"], list) and all(isinstance(r, (int, float)) for r in p["radii"])):
        raise TypeError("'radii' must be list[float]")

    if not (isinstance(p["phenotypes"], list) and len(p["phenotypes"]) == 2):
        raise ValueError("'phenotypes' must contain exactly two strings")

    # defaults
    p.setdefault("regions", None)
    p.setdefault("n_simulations", 100)
    p.setdefault("area", None)
    p.setdefault("seed", 42)
    p.setdefault("spatial_key", "spatial")
    p.setdefault("edge_correction", True)
    p.setdefault("output_path", "transform_output.pickle")


# CLI hook
if __name__ == "__main__":
    import sys, pprint
    if len(sys.argv) != 2:
        print("usage: python ripley_l_template.py params.json")
        sys.exit(1)
    ad_out = run_from_json(sys.argv[1])
    pprint.pp(ad_out.uns["ripley_l_results"])

