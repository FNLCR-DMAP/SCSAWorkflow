"""
spac.transform.clustering.phenograph — PhenoGraph clustering (CPU and GPU).

Public entry points
-------------------
prepare_features(adata, layer, features)
    Shared preprocessing: extract a dense float32 feature matrix from AnnData.
    Used by BOTH cpu and gpu paths to guarantee identical input.

phenograph_cpu(adata, features, layer, k, seed, ...)
    CPU PhenoGraph via the `phenograph` Python package.
    INDEPENDENT of spac.transformations.phenograph_clustering.

phenograph_gpu(adata, features, layer, k, seed, ...)
    GPU PhenoGraph via grapheno (RAPIDS cuml + cugraph).
    Only importable inside the GPU container; returns None in CPU-only envs.

Notes
-----
This module is a deliberate parallel to the Biowulf ``spac_phenograph/CPU/``
and ``spac_phenograph/GPU/`` folder split. Each path is a standalone
implementation; neither wraps the other. The legacy
``spac.transformations.phenograph_clustering`` function is untouched and
continues to serve the legacy ``phenograph_clustering.xml`` Galaxy tool.
"""

from .preprocess import prepare_features
from .cpu import phenograph_cpu

# GPU import is deferred. Import failures in CPU-only environments must not
# break ``from spac.transform.clustering.phenograph import phenograph_cpu``.
try:
    from .gpu.grapheno import phenograph_gpu
except ImportError:
    phenograph_gpu = None

__all__ = ["prepare_features", "phenograph_cpu", "phenograph_gpu"]
