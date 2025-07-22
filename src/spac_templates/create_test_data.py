import pickle
import numpy as np

# Create simple test data that mimics what SPAC expects
test_data = {
    "obs": {"cell_id": list(range(100))},
    "var": {"gene_names": ["gene1", "gene2"]},
    "X": np.random.rand(100, 2),
    "obsm": {"spatial": np.random.rand(100, 2) * 1000},
    "uns": {}
}

# Save as pickle
with open("test_data.pickle", "wb") as f:
    pickle.dump(test_data, f)

print("Created test_data.pickle")
