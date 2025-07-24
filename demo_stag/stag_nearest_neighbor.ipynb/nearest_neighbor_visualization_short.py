# ------------------------------------------------------------------
# 0. Imports & compatibility shims
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from spac.spatial_analysis import calculate_nearest_neighbor
from spac.visualization   import visualize_nearest_neighbor

# ------------------------------------------------------------------
# 1. Build a synthetic AnnData
# ------------------------------------------------------------------
np.random.seed(42)
n_cells, n_genes = 500, 20

# --- spatial coordinates: points on three concentric rings --------
radii = np.random.choice([20, 40, 60], n_cells, p=[0.3, 0.4, 0.3])
theta = np.random.uniform(0, 2 * np.pi, n_cells)
x = radii * np.cos(theta) + np.random.normal(0, 2, n_cells)
y = radii * np.sin(theta) + np.random.normal(0, 2, n_cells)

# --- cell‑type labels (three clusters) ----------------------------
clusters = np.where(radii < 30, "Cluster A",
           np.where(radii < 50, "Cluster B", "Cluster C"))

# --- expression matrix (random counts) ----------------------------
expr = np.random.poisson(lam=5, size=(n_cells, n_genes)).astype(np.float32)

# --- build AnnData -------------------------------------------------
adata = ad.AnnData(
    X=expr,
    obs=pd.DataFrame({"renamed_clusters": clusters}),
    var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
)
adata.obsm["spatial"] = np.column_stack([x, y])

print("Synthetic AnnData created:", adata)

# ------------------------------------------------------------------
# 2. Nearest‑neighbor calculation
# ------------------------------------------------------------------
calculate_nearest_neighbor(
    adata=adata,
    annotation="renamed_clusters",
    spatial_associated_table="spatial",
    imageid=None,
    label="spatial_distance",
)
print("NN distances stored at adata.obsm['spatial_distance']:",
      adata.obsm["spatial_distance"].shape)

# ------------------------------------------------------------------
# 3. Visualization
# ------------------------------------------------------------------
result = visualize_nearest_neighbor(
    adata=adata,
    annotation="renamed_clusters",
    spatial_distance="spatial_distance",
    distance_from="Cluster A",
    distance_to=None,              # None → distances to every other cluster
    plot_type="boxen",
    method="numeric",
    log=True,
)
result["fig"].suptitle("Synthetic demo – distances from Cluster A", weight="bold")
plt.show()
