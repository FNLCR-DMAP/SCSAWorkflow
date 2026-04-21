"""
spac.transform.clustering.phenograph.cpu

Standalone CPU PhenoGraph clustering (parallel sibling to gpu/grapheno.py).

This implementation calls the ``phenograph`` Python package directly and
does NOT wrap ``spac.transformations.phenograph_clustering``. Keeping the
two paths independent allows the legacy ``phenograph_clustering.xml``
Galaxy tool to remain untouched while the new ``phenograph_clustering_cpu.xml``
tool evolves freely.
"""
from __future__ import annotations

import time
from typing import List, Optional

from .preprocess import prepare_features


def phenograph_cpu(
    adata,
    features: Optional[List[str]] = None,
    layer: Optional[str] = None,
    k: int = 30,
    seed: int = 42,
    resolution_parameter: float = 1.0,
    n_iterations: int = 100,
    output_annotation: str = "phenograph",
    min_cluster_size: int = 10,
):
    """
    CPU PhenoGraph clustering: KNN + Jaccard + Leiden, all on CPU via
    the ``phenograph`` Python package.

    Parameters
    ----------
    adata : AnnData
        Input data. Modified in-place; cluster labels are written to
        ``adata.obs[output_annotation]``.
    features : list of str or None
        Subset of ``adata.var.index`` to use. If ``None``, uses all variables.
    layer : str or None
        Layer name. Pass ``None`` or ``"Original"`` to use ``adata.X``.
    k : int
        Number of nearest neighbors.
    seed : int
        Random seed forwarded to ``phenograph.cluster``.
    resolution_parameter : float
        Leiden resolution parameter.
    n_iterations : int
        Leiden iterations. Negative means run until no improvement.
    output_annotation : str
        Column name in ``adata.obs`` to write cluster labels.
    min_cluster_size : int
        Minimum cluster size; smaller clusters are labeled -1.

    Returns
    -------
    adata : AnnData
        The same object, with ``adata.obs[output_annotation]`` and
        ``adata.uns['phenograph_clustering_cpu']`` populated.
    """
    import phenograph

    data, feature_names = prepare_features(adata, layer=layer, features=features)

    print(
        f"CPU PhenoGraph: {data.shape[0]:,} cells x {data.shape[1]} features, "
        f"k={k}, resolution={resolution_parameter}"
    )

    t0 = time.time()
    communities, graph, Q = phenograph.cluster(
        data,
        k=k,
        seed=seed,
        resolution_parameter=resolution_parameter,
        n_iterations=n_iterations,
        min_cluster_size=min_cluster_size,
        clustering_algo="leiden",
    )
    elapsed = time.time() - t0

    adata.obs[output_annotation] = communities.astype("category")

    # Provenance — makes it easy to distinguish old vs. new CPU path downstream.
    adata.uns.setdefault("phenograph_clustering_cpu", {}).update(
        {
            "k": int(k),
            "seed": int(seed),
            "resolution_parameter": float(resolution_parameter),
            "n_iterations": int(n_iterations),
            "min_cluster_size": int(min_cluster_size),
            "layer": layer,
            "features": list(feature_names),
            "modularity": float(Q),
            "elapsed_seconds": float(elapsed),
            "backend": "phenograph",
        }
    )

    n_clusters = int(adata.obs[output_annotation].nunique())
    print(
        f"  {n_clusters} clusters, modularity={Q:.4f}, time={elapsed:.1f}s"
    )
    return adata
