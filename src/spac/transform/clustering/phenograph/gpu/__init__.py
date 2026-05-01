"""
spac.transform.clustering.phenograph.gpu

GPU phenograph implementations. Currently exposes ``grapheno`` (the
RAPIDS-based port of grapheno_dmap).

If a ``phenograph_cuml`` backend is added later, it lives in this package
alongside ``grapheno`` and this __init__ decides which to expose as the
default ``phenograph_gpu`` entry point.
"""
