"""
/opt/spac-gpu/merge_labels.py — Stage 3 of the GPU orchestrator.

Runs inside the 'preprocess' conda env of nciccbr/spac-gpu (where anndata
is available). Reads:
  - adata.pickle            (from stage 1)
  - cluster_result_<res>.npy (from stage 2, one per resolution)
  - cleaned_params.json

Writes labels into adata.obs. Single-resolution mode writes one column
named ``Output_Annotation_Name``. Profiling mode writes one column per
resolution, named ``<Output_Annotation_Name>-<k>-<res>`` — matching the
Foundry/Biowulf convention so downstream consumers don't have to change.

Final output uses ``spac.templates.template_utils.save_results`` to write
``output.pickle`` in the Galaxy work dir.

Usage (from run_gpu.sh):
    python merge_labels.py <cleaned_params.json> <work_dir>
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

from spac.templates.template_utils import parse_params, save_results


def _resolutions(params):
    profiling = bool(params.get("Profiling", False))
    res_list = params.get("Resolution_List") or []
    res_param = params.get("Resolution_Parameter", 1.0)
    if profiling:
        if not res_list:
            raise ValueError("Profiling=True but Resolution_List is empty.")
        return True, [float(r) for r in res_list]
    return False, [float(res_param)]


def main(params_path: str, work_dir: str) -> None:
    params = parse_params(params_path)

    # Switch into work_dir so save_results' relative output paths land there.
    orig_dir = os.getcwd()
    os.chdir(work_dir)

    try:
        with open("adata.pickle", "rb") as f:
            adata = pickle.load(f)

        k = int(params.get("K_Nearest_Neighbors", 30))
        seed = int(params.get("Seed", 42))
        n_iterations = int(params.get("Number_of_Iterations", 100))
        output_name = params.get("Output_Annotation_Name", "phenograph")
        layer = params.get("Table_to_Process", "Original")
        profiling, resolutions = _resolutions(params)

        print(
            f"  Merging labels: profiling={profiling}, k={k}, "
            f"resolutions={resolutions}",
            flush=True,
        )
        print(f"  AnnData before merge: {adata}", flush=True)

        for res in resolutions:
            npy_path = f"cluster_result_{res}.npy"
            if not os.path.exists(npy_path):
                raise FileNotFoundError(
                    f"Expected {npy_path} from stage 2, not found."
                )
            labels = np.load(npy_path)

            if len(labels) != adata.n_obs:
                raise ValueError(
                    f"Label count {len(labels)} from {npy_path} does not "
                    f"match adata.n_obs {adata.n_obs}."
                )

            col_name = (
                f"{output_name}-{k}-{res}" if profiling else output_name
            )
            # Store as string categorical — consistent with the CPU path's
            # AnnData convention.
            adata.obs[col_name] = labels.astype(str)
            adata.obs[col_name] = adata.obs[col_name].astype("category")

            n_clusters = int(adata.obs[col_name].nunique())
            print(
                f"\n  Column '{col_name}': {n_clusters} clusters",
                flush=True,
            )
            print("  Top cluster sizes:", flush=True)
            print(
                adata.obs[col_name].value_counts().head(10).to_string(),
                flush=True,
            )

        # Provenance
        adata.uns.setdefault("phenograph_clustering_gpu", {}).update(
            {
                "k": k,
                "seed": seed,
                "n_iterations": n_iterations,
                "resolutions": resolutions,
                "profiling": profiling,
                "layer": None if layer == "Original" else layer,
                "output_annotation": output_name,
                "backend": "grapheno",
                "rapids_version": "22.08",
            }
        )

        print(f"\n  AnnData after merge: {adata}", flush=True)

        # Ensure outputs dict exists with the standard shape save_results
        # expects, in case the caller didn't inject it.
        if "outputs" not in params:
            params["outputs"] = {
                "analysis": {"type": "file", "name": "output.pickle"}
            }

        saved = save_results(
            results={"analysis": adata},
            params=params,
            output_base_dir=".",
        )
        print(f"\n  Saved analysis: {saved['analysis']}", flush=True)
        print("  Stage 3 complete.", flush=True)
    finally:
        os.chdir(orig_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: merge_labels.py <cleaned_params.json> <work_dir>",
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
