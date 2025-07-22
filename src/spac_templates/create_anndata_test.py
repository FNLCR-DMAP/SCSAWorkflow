# create_anndata_test.py
import numpy as np
import pandas as pd
import pickle

try:
    import anndata
    import scanpy as sc

    # Create proper AnnData object
    n_cells = 100
    n_genes = 2

    # Expression matrix
    X = np.random.rand(n_cells, n_genes)

    # Cell metadata
    obs = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_cells)],
        'cell_type': np.random.choice(['TypeA', 'TypeB'], n_cells),
        'renamed_phenotypes': np.random.choice(['B cells', 'CD8 T cells'], n_cells)
    })

    # Gene metadata
    var = pd.DataFrame({
        'gene_names': [f'gene_{i}' for i in range(n_genes)]
    })

    # Create AnnData object
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Add spatial coordinates
    adata.obsm['spatial'] = np.random.rand(n_cells, 2) * 1000

    # Save
    with open('test_anndata.pickle', 'wb') as f:
        pickle.dump(adata, f)

    print("Created test_anndata.pickle with proper AnnData object")

except ImportError:
    print("AnnData not installed - creating mock object")
    # Create mock that has the right structure
    class MockAnnData:
        def __init__(self):
            self.uns = {}
            self.obs = pd.DataFrame({
                'cell_type': ['TypeA', 'TypeB'] * 50
            })
            self.obsm = {'spatial': np.random.rand(100, 2) * 1000}

    mock_data = MockAnnData()
    with open('test_anndata.pickle', 'wb') as f:
        pickle.dump(mock_data, f)

    print("Created test_anndata.pickle with mock object")
