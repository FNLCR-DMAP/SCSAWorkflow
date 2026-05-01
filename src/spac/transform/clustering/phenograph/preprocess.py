"""
Shared preprocessing for CPU and GPU phenograph paths.

Both paths use the same ``prepare_features`` to extract a dense float32
feature matrix from AnnData, guaranteeing identical inputs. This is the
single piece of code shared between the two implementations, in line with
George's April 17 guidance ("all the preprocessing you do can happen in
one place, independent if you are GPU or CPU").
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def prepare_features(
    adata,
    layer: Optional[str] = None,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract a dense feature matrix from AnnData for clustering.

    Parameters
    ----------
    adata : AnnData
    layer : str or None
        Layer name. Pass ``None`` or ``"Original"`` to use ``adata.X``.
    features : list of str or None
        Subset of ``adata.var.index`` to use. If ``None``, uses all variables.

    Returns
    -------
    data : np.ndarray
        (n_cells, n_features) float32, C-contiguous.
    feature_names : list of str
        Ordered feature names matching the columns of ``data``.

    Raises
    ------
    KeyError
        If ``layer`` is not ``None``/``"Original"`` and is not in
        ``adata.layers``.
    """
    if layer in (None, "", "Original"):
        matrix = adata.X
    else:
        if layer not in adata.layers:
            available = list(adata.layers.keys())
            raise KeyError(
                f"Layer '{layer}' not in adata.layers. Available: {available}"
            )
        matrix = adata.layers[layer]

    # Densify sparse matrices
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    feature_names = list(features) if features else list(adata.var.index)

    # Subset columns if a feature list was provided
    if features:
        col_idx = [adata.var.index.get_loc(f) for f in features]
        matrix = matrix[:, col_idx]

    data = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32))
    return data, feature_names
