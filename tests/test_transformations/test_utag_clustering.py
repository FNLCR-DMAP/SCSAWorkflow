import unittest
import numpy as np
import pandas as pd
from anndata import AnnData
from spac.transformations import run_utag_clustering

# change setting the initiation of the data as separate functions 

class TestRunUtagClustering(unittest.TestCase):
    np.random.seed(42)
    # make a dataset for clustering with 2 clusters
    def create_syn_data(self):
        syn_dataset = np.array([
            np.concatenate(
                (
                    np.random.normal(100, 1, 500),
                    np.random.normal(10, 1, 500)
                )
            ),
            np.concatenate(
                (
                    np.random.normal(10, 1, 500),
                    np.random.normal(100, 1, 500)
                )
            ),
        ]).reshape(-1, 2)

        syn_data = AnnData(
            syn_dataset,
            var=pd.DataFrame(index=['gene1', 'gene2'])
        )
        syn_data.layers['counts'] = syn_dataset
        syn_data.obsm['derived_features'] = syn_dataset

        # Add spatial coordinates    
        syn_data.obsm['spatial'] = np.array([
            np.concatenate(
                (
                    np.random.normal(100, 1, 500),
                    np.random.normal(10, 1, 500)
                )
            ),
            np.concatenate(
                (
                    np.random.normal(10, 1, 500),
                    np.random.normal(100, 1, 500)
                )
            ),
        ]).reshape(-1, 2)

        return syn_data

    # make a dataset with non normal distribution of genes, so that clustering 
    # done with PCAs and with features will produce different clusters 
    def create_adata_complex(self, n_cells_complex=500):
        # Creates a complex AnnData object with spatial gene expression patterns.
        # Step 1: Generate spatial coordinates in a circular pattern
            # - theta represents angular position (0 to 2π radians)
            # - r represents radial distance from center (0 to 10 units)
        theta = np.random.uniform(0, 2*np.pi, n_cells_complex)
        r = np.random.uniform(0, 10, n_cells_complex)
        # Step 2: Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
        x_coord = r * np.cos(theta)
        y_coord = r * np.sin(theta)
        # Step 3: Create radial distance-dependent genes
            # - gene1: Expression increases with distance from center (exponential gradient)
            # - gene2: Expression decreases with distance from center (negative exponential gradient)
            # - Both include random noise to simulate biological variability
        gene1 = np.exp(r/5) + np.random.normal(0, 0.5, n_cells_complex) # Higher expression at periphery
        gene2 = -np.exp(r/5) + np.random.normal(0, 0.5, n_cells_complex) # Higher expression at center
        # Step 4: Create angular position-dependent genes
            # - gene3: Expression follows sinusoidal pattern based on angular position
            # - gene4: Expression follows cosine pattern based on angular position
            # - These create 3 peaks/valleys around the circle (frequency=3)
            # - The cosine pattern in gene4 is shifted 30° (π/6 radians) compared to gene3's sine pattern
            # - Adds random noise with lower standard deviation (0.3)
        gene3 = np.sin(3*theta) + np.random.normal(0, 0.3, n_cells_complex)
        gene4 = np.cos(3*theta) + np.random.normal(0, 0.3, n_cells_complex)
        # Step 5: Identify quadrants based on Cartesian coordinates
            # - Quadrant 1: x>0, y>0 (top right)
            # - Quadrant 2: x<0, y>0 (top left)
            # - Quadrant 3: x<0, y<0 (bottom left)
            # - Quadrant 4: x>0, y<0 (bottom right)
        quadrant = np.where((x_coord > 0) & (y_coord > 0), 1,
                        np.where((x_coord < 0) & (y_coord > 0), 2,
                                np.where((x_coord < 0) & (y_coord < 0), 3, 4)))
        # Step 6: Create genes with quadrant-specific expression patterns
            # - gene5: Highly expressed in top half (quadrants 1,2)
            # - gene6: Highly expressed in right half (quadrants 1,4)
            # - Both include random noise to simulate biological variability                        
        gene5 = np.where(np.isin(quadrant, [1, 2]), 3, 0) + np.random.normal(0, 0.3, n_cells_complex)
        gene6 = np.where(np.isin(quadrant, [1, 4]), 3, 0) + np.random.normal(0, 0.3, n_cells_complex)
        # Step 7: Create control genes with random expression (no spatial pattern)
            # - gene7, gene8: Random normal distribution with no spatial dependency
            # - These simulate genes that are not spatially regulated
        gene7 = np.random.normal(0, 1, n_cells_complex)
        gene8 = np.random.normal(0, 1, n_cells_complex)
        # Combine all genes
        expression_matrix = np.column_stack([gene1, gene2, gene3, gene4, gene5, gene6, gene7, gene8])

        # Create AnnData object
        adata_complex = AnnData(
            X=expression_matrix,
            obs=pd.DataFrame(
                {
                    'spatial_x': x_coord,
                    'spatial_y': y_coord,
                    'quadrant': quadrant
                },
                index=[f'cell_{i}' for i in range(n_cells_complex)]
            ),
            var=pd.DataFrame(
                {
                    'gene_type': ['radial', 'radial', 'angular', 'angular', 
                                'quadrant', 'quadrant', 'random', 'random']
                },
                index=[f'gene{i}' for i in range(8)]
            )
        )

        # Add raw counts layer
        adata_complex.layers['counts'] = expression_matrix.copy()
        # Add spatial coordinates
        adata_complex.obsm["spatial"] = np.random.rand(n_cells_complex, n_cells_complex)
        return adata_complex

    def setUp(self):
        # This method is run before each test.
        # It sets up a test case with an AnnData object, a list of features,
        # and a layer name.
        np.random.seed(42)
        # make a dataset for clustering with 2 clusters
        self.syn_data = self.create_syn_data()
        # maka adata for testing clustering based on features vs PCA
        self.adata_complex = self.create_adata_complex()
        self.features = ['gene1', 'gene2', 'gene3']
        self.layer = 'counts'

    def test_same_cluster_assignments_with_same_seed(self):
        # Run run_utag_clustering with a specific seed
        # and store the cluster assignments
        run_utag_clustering(adata=self.adata_complex,
                            features=None,
                            k=15,
                            resolution=0.5,
                            max_dist=20,
                            n_pcs=None,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=None,
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel = False)
        first_run_clusters = self.adata_complex.obs['UTAG'].copy()

        # Reset the UTAG annotation and run again with the same seed
        del self.adata_complex.obs['UTAG']
        run_utag_clustering(adata=self.adata_complex,
                            features=None,
                            k=15,
                            resolution=0.5,
                            max_dist=20,
                            n_pcs=None,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=None,
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel = False)

        # Check if the cluster assignments are the same
        self.assertTrue(
            (first_run_clusters == self.adata_complex.obs['UTAG']).all()
        )

    def test_typical_case(self):
        # This test checks if the function correctly adds 'UTAG' to the
        # AnnData object's obs attribute and if it correctly sets
        # 'utag_features' in the AnnData object's uns attribute.
        run_utag_clustering(adata=self.adata_complex,
                            features=self.features,
                            k=15,
                            resolution=0.5,
                            max_dist=20,
                            n_pcs=2,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=self.layer,
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel=False)
        self.assertIn('UTAG', self.adata_complex.obs)
        self.assertEqual(self.adata_complex.uns['utag_features'], self.features)

    def test_output_annotation(self):
        # This test checks if the function correctly adds the "output_annotation" 
        # to the AnnData object's obs attribute 
        output_annotation_name = 'my_output_annotation'
        run_utag_clustering(adata=self.adata_complex,
                            features=self.features,
                            k=15,
                            resolution=0.5,
                            max_dist=20,
                            n_pcs=2,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=self.layer, 
                            output_annotation=output_annotation_name,
                            associated_table=None,
                            parallel=False)
        self.assertIn(output_annotation_name, self.adata_complex.obs)

    def test_invalid_k(self):
        # This test checks if the function raises a ValueError when the
        # k argument is not a positive integer.
        with self.assertRaises(ValueError):
            run_utag_clustering(adata=self.adata_complex,
                                features=self.features,
                                k='invalid',
                                resolution=0.5,
                                max_dist=20,
                                n_pcs=2,
                                random_state=42,
                                n_jobs=1,
                                n_iterations=5,
                                slide_key=None,
                                layer=self.layer, 
                                output_annotation="UTAG",
                                associated_table=None,
                                parallel=False)
            
    def test_clustering_accuracy(self):
        run_utag_clustering(adata=self.syn_data,
                            features=None,
                            k=15,
                            resolution=0.5,
                            max_dist=20,
                            n_pcs=None,
                            random_state=42,
                            n_jobs=1,
                            n_iterations=5,
                            slide_key=None,
                            layer=None, 
                            output_annotation="UTAG",
                            associated_table=None,
                            parallel=False)

        self.assertIn('UTAG', self.syn_data.obs)
        self.assertEqual(
            len(np.unique(self.syn_data.obs['UTAG'])),
            2)
    
    def test_features_vs_pca_utag_clustering(self):
        run_utag_clustering(
            self.adata_complex,
            features=None,
            k=15,
            resolution=0.5,
            max_dist=2,
            n_pcs=None,
            random_state=42,
            n_jobs=1,
            n_iterations=5,
            slide_key=None,
            layer=None,
            output_annotation="UTAG",
            associated_table=None,
            parallel = False
            )
        run1_clusters = list(self.adata_complex.obs["UTAG"].copy())

        # Reset the UTAG annotation and run again with the same seed
        del self.adata_complex.obs['UTAG']
        run_utag_clustering(
            self.adata_complex,
            features=None,
            k=15,
            resolution=0.5,
            max_dist=2,
            n_pcs=2,
            random_state=42,
            n_jobs=1,
            n_iterations=5,
            slide_key=None,
            layer=None,
            output_annotation="UTAG",
            associated_table=None,
            parallel = False
            )
        run2_clusters = list(self.adata_complex.obs["UTAG"].copy())

        # Check if the cluster assignments are different
        self.assertFalse(run1_clusters == run2_clusters)

if __name__ == '__main__':
    unittest.main()