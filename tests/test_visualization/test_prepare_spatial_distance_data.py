import unittest
import anndata
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from spac.visualization import _prepare_spatial_distance_data


class TestPrepareSpatialDistanceData(unittest.TestCase):

    def setUp(self):
        # Create a minimal deterministic AnnData object
        # Cells: Cell1(A), Cell2(B), Cell3(A)
        data = np.array([[1.0], [2.0], [3.0]])
        obs = pd.DataFrame(
            {'cell_type': ['A', 'B', 'A']},
            index=['Cell1', 'Cell2', 'Cell3']
        )

        # Create a minimal "spatial_distance" DataFrame
        # Columns represent phenotypes: A, B
        # Distances:
        # Cell1: A=0.0, B=1.0
        # Cell2: A=1.0, B=0.0
        # Cell3: A=0.5, B=0.5
        spatial_dist_df = pd.DataFrame(
            data=[[0.0, 1.0],
                  [1.0, 0.0],
                  [0.5, 0.5]],
            index=['Cell1', 'Cell2', 'Cell3'],
            columns=['A', 'B']
        )

        self.adata = anndata.AnnData(X=data, obs=obs)
        self.adata.obsm['spatial_distance'] = spatial_dist_df

    def _convert_expected_to_categorical(self, df, stratify_by=None):
        """Convert specific columns to categorical for comparison."""
        df['group'] = df['group'].astype('category')
        df['phenotype'] = df['phenotype'].astype('category')
        if stratify_by and stratify_by in df.columns:
            df[stratify_by] = df[stratify_by].astype('category')
        return df

    def test_io_correctness_minimal(self):
        """
        Test minimal input scenario: no distance_to, no stratify_by, no log.
        Ensures output DataFrame is tidy and correct.
        """
        df_result = _prepare_spatial_distance_data(
            adata=self.adata,
            annotation='cell_type',
            distance_from='A',
            spatial_distance='spatial_distance',
            log=False
        )

        # Expected DataFrame after filtering (distance_from='A')
        # Cell1: A=0.0, B=1.0
        # Cell3: A=0.5, B=0.5
        # Long form:
        # cellid | group | distance | phenotype
        # Cell1  | A     | 0.0      | A
        # Cell1  | B     | 1.0      | A
        # Cell3  | A     | 0.5      | A
        # Cell3  | B     | 0.5      | A
        expected = pd.DataFrame({
            'cellid': ['Cell1', 'Cell1', 'Cell3', 'Cell3'],
            'group': ['A', 'B', 'A', 'B'],
            'distance': [0.0, 1.0, 0.5, 0.5],
            'phenotype': ['A', 'A', 'A', 'A']
        })
        expected = self._convert_expected_to_categorical(expected)

        df_res_srt = df_result.sort_values(['cellid', 'group']).reset_index(
            drop=True
        )
        exp_srt = expected.sort_values(['cellid', 'group']).reset_index(
            drop=True
        )
        assert_frame_equal(df_res_srt, exp_srt, check_dtype=False)

    def test_with_distance_to_subset(self):
        """
        Test using distance_to as a subset of the available pehnotypes.
        Choose distance_to = 'B' only.
        """
        df_result = _prepare_spatial_distance_data(
            adata=self.adata,
            annotation='cell_type',
            distance_from='A',
            distance_to='B',
            spatial_distance='spatial_distance',
            log=False
        )

        # Phenotype A cells: Cell1, Cell3
        # Distances to B only:
        # Cell1: B=1.0
        # Cell3: B=0.5
        expected = pd.DataFrame({
            'cellid': ['Cell1', 'Cell3'],
            'group': ['B', 'B'],
            'distance': [1.0, 0.5],
            'phenotype': ['A', 'A']
        })
        expected = self._convert_expected_to_categorical(expected)

        df_res_srt = df_result.sort_values('cellid').reset_index(drop=True)
        exp_srt = expected.sort_values('cellid').reset_index(drop=True)
        assert_frame_equal(df_res_srt, exp_srt, check_dtype=False)

        # Check 'group' is categorical and in correct order
        self.assertTrue(pd.api.types.is_categorical_dtype(df_res_srt['group']))
        self.assertListEqual(
            list(df_res_srt['group'].cat.categories),
            ['B']
        )

    def test_with_stratify_by(self):
        """
        Test that providing a stratify_by column includes it in the output.
        """
        self.adata.obs['sample_id'] = ['S1', 'S1', 'S2']

        df_result = _prepare_spatial_distance_data(
            adata=self.adata,
            annotation='cell_type',
            distance_from='A',
            spatial_distance='spatial_distance',
            stratify_by='sample_id',
            log=False
        )

        # Phenotype A cells: Cell1(S1), Cell3(S2)
        # Distances:
        # Cell1: A=0.0, B=1.0, sample_id=S1
        # Cell3: A=0.5, B=0.5, sample_id=S2
        expected = pd.DataFrame({
            'cellid': ['Cell1', 'Cell1', 'Cell3', 'Cell3'],
            'group': ['A', 'B', 'A', 'B'],
            'distance': [0.0, 1.0, 0.5, 0.5],
            'phenotype': ['A', 'A', 'A', 'A'],
            'sample_id': ['S1', 'S1', 'S2', 'S2']
        })
        expected = self._convert_expected_to_categorical(
            expected, stratify_by='sample_id'
        )

        df_res_srt = df_result.sort_values(['cellid', 'group']).reset_index(
            drop=True
        )
        exp_srt = expected.sort_values(['cellid', 'group']).reset_index(
            drop=True
        )
        assert_frame_equal(df_res_srt, exp_srt, check_dtype=False)

    def test_log_transform(self):
        """
        Test that log1p transform is correctly applied to distances.
        """
        df_result = _prepare_spatial_distance_data(
            adata=self.adata,
            annotation='cell_type',
            distance_from='A',
            spatial_distance='spatial_distance',
            log=True
        )

        # Original distances for A-phenotype cells:
        # Cell1: A=0.0 -> log1p(0.0)=0.0, B=1.0 -> log1p(1.0)=~0.693147
        # Cell3: A=0.5 -> log1p(0.5)=~0.405465, B=0.5 -> ~0.405465
        df_sub = df_result.set_index(['cellid', 'group'])['distance']
        self.assertAlmostEqual(df_sub.loc[('Cell1', 'A')], 0.0, places=5)
        self.assertAlmostEqual(df_sub.loc[('Cell1', 'B')],
                               np.log1p(1.0), places=5)
        self.assertAlmostEqual(df_sub.loc[('Cell3', 'A')],
                               np.log1p(0.5), places=5)
        self.assertAlmostEqual(df_sub.loc[('Cell3', 'B')],
                               np.log1p(0.5), places=5)

    def test_error_no_distance_from(self):
        """
        Test that missing `distance_from` raises correct ValueError.
        """
        with self.assertRaises(ValueError) as ctx:
            _prepare_spatial_distance_data(
                adata=self.adata,
                annotation='cell_type',
                spatial_distance='spatial_distance'
            )
        expected_msg = (
            "Please specify the 'distance_from' phenotype. This indicates "
            "the reference group from which distances are measured."
        )
        self.assertEqual(str(ctx.exception), expected_msg)

    def test_error_spatial_distance_key_missing(self):
        """
        Test that missing spatial_distance key raises correct ValueError.
        """
        del self.adata.obsm['spatial_distance']
        with self.assertRaises(ValueError) as ctx:
            _prepare_spatial_distance_data(
                adata=self.adata,
                annotation='cell_type',
                distance_from='A',
                spatial_distance='spatial_distance'
            )
        expected_msg = (
            "'spatial_distance' does not exist in the provided dataset. "
            "Please run 'calculate_nearest_neighbor' first to compute and "
            "store spatial distance. Available keys: []"
        )
        self.assertEqual(str(ctx.exception), expected_msg)

    def test_error_missing_phenotypes_in_distance_map(self):
        """
        Test error when phenotypes exist in annotation but not in the
        distance_map columns.
        """
        data = np.array([[1.0], [2.0], [3.0]])
        # Add 'C' to annotation, ensuring annotation check passes for 'C'
        obs = pd.DataFrame({'cell_type': ['A', 'C', 'A']},
                           index=['Cell1', 'Cell2', 'Cell3'])
        # distance_map without 'C'
        spatial_dist_df = pd.DataFrame(
            data=[[0.0, 1.0],
                  [1.0, 0.0],
                  [0.5, 0.5]],
            index=['Cell1', 'Cell2', 'Cell3'],
            columns=['A', 'B']
        )
        self.adata = anndata.AnnData(X=data, obs=obs)
        self.adata.obsm['spatial_distance'] = spatial_dist_df

        with self.assertRaises(ValueError) as ctx:
            _prepare_spatial_distance_data(
                adata=self.adata,
                annotation='cell_type',
                distance_from='A',
                distance_to='C',  # 'C' in obs but not in distance_map
                spatial_distance='spatial_distance'
            )
        expected_msg = (
            "Phenotypes ['C'] not found in columns of 'spatial_distance'. "
            "Columns present: ['A', 'B']"
        )
        self.assertEqual(str(ctx.exception), expected_msg)


if __name__ == '__main__':
    unittest.main()
