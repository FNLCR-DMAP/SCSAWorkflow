import numpy as np
import pandas as pd
import anndata as ad


def mock_adata(n_cells: int = 40) -> ad.AnnData:
    """Tiny AnnData with coords + phenotype labels for fast tests."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {"renamed_phenotypes": np.where(rng.random(n_cells) > 0.5,
                                        "B cells", "CD8 T cells")}
    )
    X = rng.normal(size=(n_cells, 3))
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["spatial"] = rng.random((n_cells, 2)) * 300
    return adata

