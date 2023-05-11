import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../src")

from spac.transformations import rename_clustering
import anndata
import pandas as pd
import numpy as np
import unittest

class TestRenameClustering(unittest.TestCase):
    def test_rename_clustering_basic(self):  
        # Create an AnnData object with 3 cells and dummy gene expression data
        X = np.random.rand(3, 4)
        obs = pd.DataFrame(index=["cell_1", "cell_2", "cell_3"])
        obs["original_phenotype"] = ["A", "B", "A"]

        adata = anndata.AnnData(X=X, obs=obs)

        # Map two of these cells to a group called "group_0" and one cell to a group called "group_1"
        new_phenotypes = {
            "A": "group_0",
            "B": "group_1",
        }

        # Use the rename_clustering function
        adata = rename_clustering(adata, "original_phenotype", new_phenotypes, new_column_name="renamed_clusters")

        # Verify that the mapping has taken place correctly
        self.assertTrue(all(adata.obs["renamed_clusters"] == ["group_0", "group_1", "group_0"]))  

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


