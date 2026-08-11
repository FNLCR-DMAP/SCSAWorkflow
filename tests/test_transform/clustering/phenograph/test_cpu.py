import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from anndata import AnnData

sys.path.append(str(Path(__file__).resolve().parents[4] / "src"))

from spac.transform.clustering.phenograph.cpu import phenograph_cpu


class TestPhenographCPU(unittest.TestCase):
    def test_numpy_communities_are_saved_as_aligned_categorical(self):
        obs_index = ["cell-c", "cell-a", "cell-b"]
        adata = AnnData(
            X=np.array(
                [
                    [1.0, 0.1],
                    [0.9, 0.2],
                    [0.1, 1.1],
                ],
                dtype=np.float32,
            ),
            obs=pd.DataFrame(index=obs_index),
            var=pd.DataFrame(index=["marker_a", "marker_b"]),
        )

        def fake_cluster(data, **kwargs):
            return np.array([2, 2, 1]), None, 0.75

        fake_phenograph = types.SimpleNamespace(cluster=fake_cluster)

        with patch.dict(sys.modules, {"phenograph": fake_phenograph}):
            result = phenograph_cpu(
                adata,
                k=2,
                seed=42,
                output_annotation="phenograph",
            )

        self.assertIs(result, adata)
        self.assertEqual(list(adata.obs.index), obs_index)
        self.assertIn("phenograph", adata.obs)
        self.assertIsInstance(
            adata.obs["phenograph"].dtype,
            pd.CategoricalDtype,
        )
        self.assertEqual(list(adata.obs["phenograph"]), [2, 2, 1])
        self.assertEqual(
            adata.uns["phenograph_clustering_cpu"]["modularity"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
