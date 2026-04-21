"""
/opt/spac-gpu/run_grapheno.py — Stage 2 of the GPU orchestrator.

Runs inside the 'rapids' conda env of nciccbr/spac-gpu. Reads the feature
matrix written by stage 1 (``features.npy``), runs grapheno clustering at
one or more resolutions, and writes cluster labels as .npy files for stage 3.

In profiling mode (multiple resolutions at same k), grapheno's parquet
caching reuses KNN and Jaccard across resolutions — only Leiden re-runs.

Usage (from run_gpu.sh):
    python run_grapheno.py <cleaned_params.json> <work_dir>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from spac.transform.clustering.phenograph.gpu.grapheno import cluster_from_array


def _resolutions(params):
    profiling = bool(params.get("Profiling", False))
    res_list = params.get("Resolution_List") or []
    res_param = params.get("Resolution_Parameter", 1.0)
    if profiling:
        if not res_list:
            raise ValueError(
                "Profiling=True but Resolution_List is empty. Add resolutions "
                "in the Galaxy UI or disable Profiling."
            )
        return True, [float(r) for r in res_list]
    return False, [float(res_param)]


def main(params_path: str, work_dir: str) -> None:
    with open(params_path) as f:
        params = json.load(f)

    with open(os.path.join(work_dir, "features_meta.json")) as f:
        meta = json.load(f)
    expected_n_obs = int(meta["n_obs"])

    data = np.load(os.path.join(work_dir, "features.npy"))
    if data.shape[0] != expected_n_obs:
        raise RuntimeError(
            f"features.npy has {data.shape[0]} rows but features_meta.json "
            f"says {expected_n_obs}. Stage 1 output is inconsistent."
        )

    k = int(params.get("K_Nearest_Neighbors", 30))
    seed = int(params.get("Seed", 42))
    n_iterations = int(params.get("Number_of_Iterations", 100))
    profiling, resolutions = _resolutions(params)

    print(
        f"  data: {data.shape[0]:,} cells x {data.shape[1]} features",
        flush=True,
    )
    print(
        f"  k={k}, seed={seed}, n_iterations={n_iterations}",
        flush=True,
    )
    print(
        f"  mode: {'profiling' if profiling else 'single'}, "
        f"resolutions={resolutions}",
        flush=True,
    )

    # Run grapheno in the work dir so its parquet caches (used for profiling
    # across resolutions) land next to our other intermediates and get
    # cleaned up by Galaxy at job end.
    orig_dir = os.getcwd()
    os.chdir(work_dir)

    try:
        for i, res in enumerate(resolutions, 1):
            print(
                f"\n  --- Resolution {res} ({i}/{len(resolutions)}) ---",
                flush=True,
            )
            t0 = time.time()

            labels, modularity, elapsed = cluster_from_array(
                data=data,
                n_neighbors=k,
                resolution=res,
                random_state=seed,
                n_iterations=n_iterations,
                min_size=10,
                work_dir=".",
            )

            if len(labels) != expected_n_obs:
                raise RuntimeError(
                    f"Grapheno returned {len(labels)} labels at res={res} "
                    f"but adata has {expected_n_obs} obs. Refusing to write "
                    f"mismatched labels."
                )

            out_path = f"cluster_result_{res}.npy"
            np.save(out_path, labels)

            n_clusters = int(np.unique(labels).size)
            n_valid = int((labels >= 0).sum())
            n_small = int((labels < 0).sum())
            wall = time.time() - t0

            print(
                f"  res={res}: {n_clusters} clusters "
                f"({n_valid:,} assigned, {n_small:,} < min_size), "
                f"modularity={modularity:.4f}, time={wall:.1f}s, "
                f"wrote {out_path}",
                flush=True,
            )
    finally:
        os.chdir(orig_dir)

    print("\n  Stage 2 complete.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: run_grapheno.py <cleaned_params.json> <work_dir>",
            file=sys.stderr,
        )
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
