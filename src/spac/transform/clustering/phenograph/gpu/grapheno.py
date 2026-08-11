"""
spac.transform.clustering.phenograph.gpu.grapheno

GPU PhenoGraph clustering — ported from the grapheno_dmap package used in
Foundry/Biowulf production.

Attribution
-----------
Original algorithm: Erik Burlingame (Apache 2.0)
    https://gitlab.com/eburling/grapheno
DMAP adaptations: Rui He (NCI/DMAP), March 2024 — added resolution and
    random_state forwarding.
SCSAWorkflow port: Fang Liu (NCI/DMAP), April 2026 — packaged into SPAC,
    added numpy-array and AnnData entry points, added cugraph 22.08/24.x
    API compatibility.

Algorithm
---------
    Dask KNN (cuml)  ->  Jaccard (cugraph)  ->  Leiden (cugraph)  ->  sort_by_size

The three core functions (``compute_and_cache_knn_edgelist``,
``compute_and_cache_jac_edgelist``, and ``cluster``) are preserved
byte-compatible with the Biowulf production source to keep the validated
5.5M-cell output identical. Two additions:

1. ``cluster_from_array(data, ...)`` — numpy-array entry point. Wraps the
   CSV-based ``cluster()`` by writing a temp CSV to ``work_dir`` and reading
   the labels back.
2. ``phenograph_gpu(adata, ...)`` — AnnData-level entry point that mirrors
   the CPU signature in ``spac.transform.clustering.phenograph.cpu``.

Requires
--------
RAPIDS (cuml, cugraph, cudf, cupy, dask_cudf, dask_cuda). Do NOT import
this module in a CPU-only environment; ``spac.transform.clustering.phenograph``
__init__ handles the ImportError gracefully.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np

# Heavy RAPIDS imports — only succeed inside the GPU container.
import cudf
import cugraph
import cupy as cp
from cuml.neighbors import NearestNeighbors as NN
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

# Optional import: dask_cudf and distributed KNN. Not all RAPIDS versions
# expose the Dask NN at the same path.
try:
    import dask_cudf
    from cuml.dask.neighbors import NearestNeighbors as DaskNN

    HAS_DASK_NN = True
except ImportError:
    dask_cudf = None
    DaskNN = None
    HAS_DASK_NN = False


# ---------------------------------------------------------------------------
# KNN edgelist (ported verbatim from grapheno_dmap.cluster, minus tqdm)
# ---------------------------------------------------------------------------

def compute_and_cache_knn_edgelist(
    input_csv_path: str,
    knn_edgelist_path: str,
    features: List[str],
    n_neighbors: int,
    client=None,
) -> None:
    print(
        f"Computing and caching {n_neighbors}NN edgelist: {knn_edgelist_path}"
    )

    if client and HAS_DASK_NN:
        chunksize = cugraph.dask.get_chunksize(input_csv_path)
        X = dask_cudf.read_csv(input_csv_path, chunksize=chunksize)
        X = X.loc[:, features].astype("float32")
        model = DaskNN(n_neighbors=n_neighbors + 1, client=client)
    else:
        X = cudf.read_csv(input_csv_path)
        X = X.loc[:, features].astype("float32")
        model = NN(n_neighbors=n_neighbors + 1)

    model.fit(X)

    n_vertices = X.shape[0].compute() if client and HAS_DASK_NN else X.shape[0]

    # exclude self index
    knn_edgelist = model.kneighbors(X, return_distance=False).loc[:, 1:]
    if client and HAS_DASK_NN:
        knn_edgelist = knn_edgelist.compute().reset_index(drop=True)
    knn_edgelist = knn_edgelist.melt(var_name="knn", value_name="dst")
    knn_edgelist = knn_edgelist.reset_index().rename(columns={"index": "src"})
    knn_edgelist = knn_edgelist.loc[:, ["src", "dst"]]
    knn_edgelist["src"] = knn_edgelist["src"] % n_vertices  # avoids transpose
    knn_edgelist.to_parquet(knn_edgelist_path)


# ---------------------------------------------------------------------------
# Jaccard edgelist
# ---------------------------------------------------------------------------

def compute_and_cache_jac_edgelist(
    knn_edgelist_path: str,
    jac_edgelist_path: str,
    distributed: bool = False,
) -> None:
    print(f"Computing and caching jaccard edgelist: {jac_edgelist_path}")
    knn_graph = _load_knn_graph(knn_edgelist_path, distributed)
    jac_graph = cugraph.jaccard(knn_graph)
    jac_graph.to_parquet(jac_edgelist_path)


def _load_knn_graph(knn_edgelist_path: str, distributed: bool = False):
    """
    Load KNN edgelist parquet back into a cugraph.Graph.

    RAPIDS 22.08 uses source='src', destination='dst'. RAPIDS 24.x may
    use different column names. We detect and remap here.
    """
    G = cugraph.Graph()
    if distributed and HAS_DASK_NN:
        edgelist = dask_cudf.read_parquet(
            knn_edgelist_path, split_row_groups=True
        )
    else:
        edgelist = cudf.read_parquet(knn_edgelist_path)

    cols = list(edgelist.columns)
    # Biowulf-produced parquet uses 'src'/'dst'; normalize if needed.
    src_col = "src" if "src" in cols else cols[0]
    dst_col = "dst" if "dst" in cols else cols[1]

    if distributed and HAS_DASK_NN:
        G.from_dask_cudf_edgelist(edgelist, source=src_col, destination=dst_col)
    else:
        G.from_cudf_edgelist(edgelist, source=src_col, destination=dst_col)
    return G


def _load_jac_graph(jac_edgelist_path: str, distributed: bool = False):
    """
    Load Jaccard edgelist parquet back into a cugraph.Graph with weights.

    cugraph 22.08 jaccard output columns: ``source``, ``destination``,
    ``jaccard_coeff``. cugraph 24.x may rename these to ``first``, ``second``.
    The ``edge_attr`` key should be stable as ``jaccard_coeff``.
    """
    G = cugraph.Graph()
    if distributed and HAS_DASK_NN:
        edgelist = dask_cudf.read_parquet(
            jac_edgelist_path, split_row_groups=True
        )
        G.from_dask_cudf_edgelist(edgelist, edge_attr="jaccard_coeff")
    else:
        edgelist = cudf.read_parquet(jac_edgelist_path)
        G.from_cudf_edgelist(edgelist, edge_attr="jaccard_coeff")
    return G


# ---------------------------------------------------------------------------
# sort_by_size (verbatim from grapheno_dmap)
# ---------------------------------------------------------------------------

def sort_by_size(clusters, min_size: int):
    """
    Relabel clustering in order of descending cluster size.

    New labels are consecutive integers beginning at 0. Clusters smaller
    than ``min_size`` are assigned to -1.

    Copied from https://github.com/jacoblevine/PhenoGraph.
    """
    relabeled = cp.zeros(clusters.shape, dtype=int)
    _, counts = cp.unique(clusters, return_counts=True)
    o = cp.argsort(counts)[::-1]
    for i, c in enumerate(o):
        if counts[c] > min_size:
            relabeled[clusters == c] = i
        else:
            relabeled[clusters == c] = -1
    return relabeled


# ---------------------------------------------------------------------------
# Leiden wrapper — version-agnostic
# ---------------------------------------------------------------------------

def _run_leiden(jac_graph, resolution: float, random_state: int, max_iter: int):
    """
    Call cugraph.leiden with forward-compatible kwargs.

    RAPIDS 22.08's cugraph.leiden signature:
        leiden(G, resolution=1.0)
    RAPIDS 24.x's cugraph.leiden signature:
        leiden(G, resolution=1.0, random_state=None, max_iter=100)

    We prefer the newer signature; fall back to the old one via try/except.
    """
    try:
        return cugraph.leiden(
            jac_graph,
            resolution=resolution,
            random_state=random_state,
            max_iter=max_iter,
        )
    except TypeError:
        print(
            "  [WARN] Falling back to cugraph.leiden without random_state/"
            "max_iter (RAPIDS 22.08 signature)"
        )
        return cugraph.leiden(jac_graph, resolution=resolution)


# ---------------------------------------------------------------------------
# Core cluster() — ported from grapheno_dmap.cluster
# ---------------------------------------------------------------------------

def cluster(
    input_csv_path: str,
    features: List[str],
    n_neighbors: int = 30,
    resolution: float = 1.0,
    random_state: int = 42,
    n_iterations: int = 100,
    distributed_knn: bool = True,
    distributed_graphs: bool = False,
    min_size: int = 10,
):
    """
    Run grapheno clustering on a CSV input, writing a parquet output with
    a ``cluster`` column alongside the original features.

    Reads the input CSV, runs KNN -> Jaccard -> Leiden -> sort_by_size, and
    returns a cuDF DataFrame with the original features plus a ``cluster``
    column. KNN and Jaccard edgelists are cached as parquet files in the
    current working directory so subsequent calls at the same ``n_neighbors``
    can reuse them (the profiling optimization).

    Matches the original grapheno_dmap.cluster signature plus an added
    ``n_iterations`` for Leiden (threaded through for cugraph 24.x).
    """
    tic = time.time()

    stem = os.path.basename(input_csv_path).rsplit(".", 1)[0]
    knn_edgelist_path = f"{stem}_{n_neighbors}NN_edgelist.parquet"
    jac_edgelist_path = f"{stem}_{n_neighbors}NN_edgelist_jaccard.parquet"

    subtic = time.time()

    if os.path.exists(jac_edgelist_path):
        print(f"Loading cached jaccard edgelist: {jac_edgelist_path}")
        jac_graph = _load_jac_graph(jac_edgelist_path, distributed_graphs)
        print(f"Jaccard graph loaded in {time.time() - subtic:.2f}s")

    elif os.path.exists(knn_edgelist_path):
        print(f"Loading cached kNN edgelist: {knn_edgelist_path}")
        compute_and_cache_jac_edgelist(
            knn_edgelist_path, jac_edgelist_path, distributed_graphs
        )
        jac_graph = _load_jac_graph(jac_edgelist_path, distributed_graphs)
        print(
            f"Jaccard graph computed, cached, and reloaded in "
            f"{time.time() - subtic:.2f}s"
        )

    else:
        # Note: LocalCUDACluster is started even when distributed_knn=False,
        # matching the original grapheno_dmap behavior. On AWS Docker this is
        # safe; on Biowulf Singularity with RAPIDS 24.x this triggers the UCX
        # segfault — which is why production stays on RAPIDS 22.08.
        if distributed_knn and HAS_DASK_NN:
            with LocalCUDACluster() as lc, Client(lc) as client:
                compute_and_cache_knn_edgelist(
                    input_csv_path,
                    knn_edgelist_path,
                    features,
                    n_neighbors,
                    client,
                )
        else:
            compute_and_cache_knn_edgelist(
                input_csv_path, knn_edgelist_path, features, n_neighbors, None
            )

        print(
            f"{n_neighbors}NN edgelist computed in "
            f"{time.time() - subtic:.2f}s"
        )

        subtic = time.time()
        compute_and_cache_jac_edgelist(
            knn_edgelist_path, jac_edgelist_path, distributed_graphs
        )
        jac_graph = _load_jac_graph(jac_edgelist_path, distributed_graphs)
        print(
            f"Jaccard graph computed, cached, and reloaded in "
            f"{time.time() - subtic:.2f}s"
        )

    subtic = time.time()
    print("Computing Leiden clustering over Jaccard graph...")
    clusters_df, modularity = _run_leiden(
        jac_graph,
        resolution=resolution,
        random_state=random_state,
        max_iter=n_iterations,
    )
    print(f"Leiden clustering completed in {time.time() - subtic:.2f}s")
    print(f"Clusters detected: {len(clusters_df.partition.unique())}")
    print(f"Clusters modularity: {modularity}")

    clusters_arr = clusters_df.sort_values(by="vertex").partition.values
    clusters_arr = sort_by_size(clusters_arr, min_size)

    out_parquet = f"{input_csv_path.rsplit('.', 1)[0]}_{n_neighbors}NN_leiden.parquet"
    print(f"Writing output dataframe: {out_parquet}")

    df = cudf.read_csv(input_csv_path)
    df["cluster"] = clusters_arr
    df.to_parquet(out_parquet)
    df = cudf.read_parquet(out_parquet)
    print(f"Grapheno completed in {time.time() - tic:.2f}s")

    return df, float(modularity), time.time() - tic


# ---------------------------------------------------------------------------
# NumPy-array entry point
# ---------------------------------------------------------------------------

def cluster_from_array(
    data: np.ndarray,
    n_neighbors: int = 30,
    resolution: float = 1.0,
    random_state: int = 42,
    n_iterations: int = 100,
    min_size: int = 10,
    work_dir: str = ".",
) -> Tuple[np.ndarray, float, float]:
    """
    NumPy-array wrapper around ``cluster()``.

    Writes ``data`` as ``cluster_input.csv`` in ``work_dir``, calls
    ``cluster()``, and reads back only the ``cluster`` column. Temp CSV and
    parquet caches remain in ``work_dir`` — clean up externally if needed.

    Returns
    -------
    labels : np.ndarray
        (n_cells,) int cluster labels, -1 for clusters smaller than ``min_size``.
    modularity : float
    elapsed : float
        Total wall-clock time.
    """
    import pandas as pd  # kept local to avoid pandas import cost at module load

    os.makedirs(work_dir, exist_ok=True)
    feature_names = [f"feature_{i}" for i in range(data.shape[1])]
    csv_path = os.path.join(work_dir, "cluster_input.csv")
    pd.DataFrame(data, columns=feature_names).to_csv(csv_path, index=False)

    orig_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        df, modularity, elapsed = cluster(
            input_csv_path=os.path.basename(csv_path),
            features=feature_names,
            n_neighbors=n_neighbors,
            resolution=resolution,
            random_state=random_state,
            n_iterations=n_iterations,
            distributed_knn=True,
            distributed_graphs=False,
            min_size=min_size,
        )
        # df is a cudf.DataFrame; pull cluster column as numpy
        labels = df["cluster"].to_numpy()
    finally:
        os.chdir(orig_dir)

    if len(labels) != data.shape[0]:
        raise RuntimeError(
            f"Grapheno returned {len(labels)} labels but input had "
            f"{data.shape[0]} rows."
        )

    return labels, modularity, elapsed


# ---------------------------------------------------------------------------
# AnnData-level entry point (mirrors CPU signature)
# ---------------------------------------------------------------------------

def phenograph_gpu(
    adata,
    features: Optional[List[str]] = None,
    layer: Optional[str] = None,
    k: int = 30,
    seed: int = 42,
    resolution_parameter: float = 1.0,
    n_iterations: int = 100,
    output_annotation: str = "phenograph",
    min_cluster_size: int = 10,
    work_dir: Optional[str] = None,
):
    """
    GPU PhenoGraph clustering mirroring the CPU signature.

    Parameters match ``spac.transform.clustering.phenograph.cpu.phenograph_cpu``
    so templates can swap between them with minimal branching.

    Parameters
    ----------
    adata : AnnData
    features : list of str or None
    layer : str or None
    k : int
    seed : int
    resolution_parameter : float
    n_iterations : int
    output_annotation : str
    min_cluster_size : int
    work_dir : str or None
        Scratch dir for grapheno's CSV and parquet caches. Defaults to cwd.

    Returns
    -------
    adata : AnnData
        The same object, with ``adata.obs[output_annotation]`` (categorical
        string labels) and ``adata.uns['phenograph_clustering_gpu']`` populated.
    """
    from .preprocess import prepare_features  # local import to avoid cycle

    data, feature_names = prepare_features(adata, layer=layer, features=features)

    print(
        f"GPU PhenoGraph (grapheno): {data.shape[0]:,} cells x "
        f"{data.shape[1]} features, k={k}, resolution={resolution_parameter}"
    )

    labels, modularity, elapsed = cluster_from_array(
        data=data,
        n_neighbors=k,
        resolution=resolution_parameter,
        random_state=seed,
        n_iterations=n_iterations,
        min_size=min_cluster_size,
        work_dir=work_dir or os.getcwd(),
    )

    # Convert to pandas categorical string labels (consistent with CPU)
    adata.obs[output_annotation] = labels.astype(str)
    adata.obs[output_annotation] = adata.obs[output_annotation].astype("category")

    adata.uns.setdefault("phenograph_clustering_gpu", {}).update(
        {
            "k": int(k),
            "seed": int(seed),
            "resolution_parameter": float(resolution_parameter),
            "n_iterations": int(n_iterations),
            "min_cluster_size": int(min_cluster_size),
            "layer": layer,
            "features": list(feature_names),
            "modularity": float(modularity),
            "elapsed_seconds": float(elapsed),
            "backend": "grapheno",
        }
    )

    n_clusters = int(adata.obs[output_annotation].nunique())
    print(
        f"  {n_clusters} clusters, modularity={modularity:.4f}, time={elapsed:.1f}s"
    )
    return adata


__all__ = [
    "cluster",
    "cluster_from_array",
    "phenograph_gpu",
    "sort_by_size",
]
