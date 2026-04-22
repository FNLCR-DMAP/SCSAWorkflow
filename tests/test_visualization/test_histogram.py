import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
from spac.visualization import histogram
mpl.use('Agg')  # Set the backend to 'Agg' to suppress plot window
from unittest import mock


class TestHistogram(unittest.TestCase):
    def setUp(self):
        # Number of cells and genes
        n_cells, n_genes = 100, 3

        # Create a data matrix with fixed values: [1, 2, 3], [2, 3, 4], ...
        X = np.array([[(j+1) + i for j in range(n_genes)]
                      for i in range(n_cells)])

        annotation_values = ['A', 'B']
        annotation_types = ['cell_type_1', 'cell_type_2']
        cell_range = [f'cell_{i}' for i in range(1, n_cells+1)]

        # Create annotations with a repetitive pattern:
        # ['A', 'B', 'A', 'B', ...]
        annotation = pd.DataFrame({
            'annotation1': [annotation_values[i % 2] for i in range(n_cells)],
            'annotation2': [annotation_types[i % 2] for i in range(n_cells)],
        }, index=cell_range)

        var = pd.DataFrame(index=['marker1', 'marker2', 'marker3'])

        self.adata = anndata.AnnData(
            X.astype(np.float32), obs=annotation, var=var
        )

        # Create default layer
        self.adata.layers['Default'] = X.astype(np.float32)

    def tearDown(self):
        # Closes all figures to prevent memory issues
        plt.close('all')

    def _make_many_groups_adata(self, n_groups=25):
        """Create a compact AnnData fixture with one row per unique group."""
        X = np.arange(1, n_groups + 1, dtype=np.float32).reshape(-1, 1)
        obs = pd.DataFrame(
            {'many_groups': [f'g{i}' for i in range(n_groups)]},
            index=[f'cell_{i}' for i in range(n_groups)],
        )
        var = pd.DataFrame(index=['marker1'])
        return anndata.AnnData(X, obs=obs, var=var)

    def _make_long_label_facet_adata(self, include_short=False):
        """Create small categorical facet fixtures for long-label geometry tests."""
        obs = {
            'annotation2': ['g1', 'g1', 'g1', 'g1',
                            'g2', 'g2', 'g2', 'g2',
                            'g3', 'g3', 'g3', 'g3'],
        }
        if include_short:
            obs['annotation_short'] = pd.Categorical(
                ['A', 'B', 'C', 'D'] * 3,
                categories=['A', 'B', 'C', 'D'],
            )
        obs['annotation_long'] = pd.Categorical(
            [
                'Activated T/B Cell',
                'Cytotoxic T Cell',
                'Follicular Dendritic Cell',
                'Regulatory T Cell',
            ] * 3,
            categories=[
                'Activated T/B Cell',
                'Cytotoxic T Cell',
                'Follicular Dendritic Cell',
                'Regulatory T Cell',
            ],
        )
        return anndata.AnnData(
            np.arange(1, 13, dtype=np.float32).reshape(-1, 1),
            obs=pd.DataFrame(obs, index=[f'cell_{i}' for i in range(12)]),
            var=pd.DataFrame(index=['marker1']),
        )

    def test_both_feature_and_annotation(self):
        err_msg = ("Cannot pass both feature and annotation,"
                   " choose one.")
        with self.assertRaisesRegex(ValueError, err_msg):
            histogram(
                self.adata,
                feature='marker1',
                annotation='annotation1'
            )

    def test_histogram_feature(self):
        # Define bin edges so that each bin covers exactly one integer value
        bin_edges = list(np.linspace(0.5, 100.5, 101))

        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            bins=bin_edges
        ).values()

        # Check the histogram bars
        bars = ax.patches
        # Expecting 100 bars for 100 distinct values
        self.assertEqual(len(bars), 100)

        # Check the height and position of each bar
        for i, bar in enumerate(bars):
            # Each bar should have a height of 1
            self.assertAlmostEqual(bar.get_height(), 1)

            # Bar centers should match the values from 1 to 100
            expected_center = i + 1
            self.assertAlmostEqual(
                bar.get_x() + bar.get_width() / 2, expected_center
            )

        # Check that ax is an Axes object
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_histogram_annotation(self):
        fig, ax, df = histogram(
            self.adata, 
            annotation='annotation1'
        ).values()
        total_annotation = len(self.adata.obs['annotation1'])
        # Assuming the y-axis of the histogram represents counts
        # (not frequencies or probabilities)
        self.assertEqual(sum(p.get_height() for p in ax.patches),
                         total_annotation)
        # Check that ax is an Axes object
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_histogram_feature_group_by(self):
        fig, axs, df = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=False
        ).values()
        self.assertEqual(len(axs), 2)
        self.assertIsInstance(axs[0], mpl.axes.Axes)
        self.assertIsInstance(axs[1], mpl.axes.Axes)
        # Check the number of axes returned
        unique_annotations = self.adata.obs['annotation2'].nunique()
        self.assertEqual(len(axs), unique_annotations)

    def test_histogram_feature_group_by_one_label_annotation(self):

        # Setup an annotation with one value
        n_cells, n_genes = 100, 3
        # Create a data matrix with fixed values: [1, 2, 3], [2, 3, 4], ...
        X = np.array([[(j+1) + i for j in range(n_genes)]
                      for i in range(n_cells)])

        cell_range = [f'cell_{i}' for i in range(1, n_cells+1)]

        # Create annotations with one label 'A'
        annotation = pd.DataFrame({
            'annotation1': ['A' for i in range(n_cells)],
        }, index=cell_range)

        var = pd.DataFrame(index=['marker1', 'marker2', 'marker3'])

        adata = anndata.AnnData(
            X.astype(np.float32), obs=annotation, var=var
        )

        fig, axs, df = histogram(
            adata,
            feature='marker1',
            group_by='annotation1',
            together=False
        ).values()
        # Since only one plot is created, axs is an Axes object
        self.assertIsInstance(axs, mpl.axes.Axes)
        # Wrap axs in a list for consistent handling
        axs = [axs]

        # Check the number of returned axes matches the unique group count
        unique_annotations = adata.obs['annotation1'].nunique()
        self.assertEqual(len(axs), unique_annotations)

    def test_x_log_scale_transformation(self):
        """Test that x_log_scale applies log1p transformation."""
        original_values = self.adata.X[
            :, self.adata.var_names.get_loc('marker1')
        ].flatten()

        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            x_log_scale=True
        ).values()

        # Check that x-axis label is updated
        self.assertEqual(ax.get_xlabel(), 'log(marker1)')

        # Since seaborn's histplot does not expose the data directly,
        # check that the x-axis limits encompass the transformed data
        x_min, x_max = ax.get_xlim()
        transformed_values = np.log1p(original_values)
        expected_min = transformed_values.min()
        expected_max = transformed_values.max()

        # Check that x-axis limits include the transformed data range
        self.assertLessEqual(x_min, expected_min)
        self.assertGreaterEqual(x_max, expected_max)

    @mock.patch('builtins.print')
    def test_negative_values_x_log_scale(self, mock_print):
        """Test that negative values disable x_log_scale
        and print a message."""
        # Introduce negative values
        self.adata.X[0, self.adata.var_names.get_loc('marker1')] = -1

        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            x_log_scale=True
        ).values()

        # Check that x-axis label is not changed
        self.assertEqual(ax.get_xlabel(), 'marker1')

        # Check that a message was printed
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        expected_msg = (
            "There are negative values in the data, disabling x_log_scale."
        )
        self.assertIn(expected_msg, print_calls)

    def test_title(self):
        """Test that title changes based on 'layer' information"""
        fig, ax, df = histogram(
            self.adata, 
            feature='marker1'
        ).values()
        self.assertEqual(ax.get_title(), 'Layer: Original')

        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            layer='Default'
        ).values()
        self.assertEqual(ax.get_title(), f'Layer: Default')

        fig, ax, df =  histogram(
            self.adata, 
            annotation='annotation1', 
            layer='Default'
        ).values()
        self.assertEqual(ax.get_title(), '')

    def test_y_log_scale_axis(self):
        """Test that y_log_scale sets y-axis to log scale."""
        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            y_log_scale=True
        ).values()
        self.assertEqual(ax.get_yscale(), 'log')

    def test_y_log_scale_label(self):
        """Test that y-axis label is updated when y_log_scale is True."""
        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            y_log_scale=True
        ).values()
        self.assertEqual(ax.get_ylabel(), 'log(Count)')

    def test_y_axis_label_based_on_stat(self):
        """Test that y-axis label changes based on the 'stat' parameter."""
        # Test default stat ('count')
        fig, ax, df = histogram(
            self.adata, 
            feature='marker1'
        ).values()
        self.assertEqual(ax.get_ylabel(), 'Count')

        # Test 'frequency' stat
        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            stat='frequency'
        ).values()
        self.assertEqual(ax.get_ylabel(), 'Frequency')

        # Test 'density' stat
        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            stat='density'
        ).values()
        self.assertEqual(ax.get_ylabel(), 'Density')

        # Test 'probability' stat
        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            stat='probability'
        ).values()
        self.assertEqual(ax.get_ylabel(), 'Probability')

    def test_y_log_scale_with_different_stats(self):
        """Test y-axis label when y_log_scale is True
        and different stats are used."""
        fig, ax, df = histogram(
            self.adata, feature='marker1', y_log_scale=True, stat='density'
        ).values()
        self.assertEqual(ax.get_ylabel(), 'log(Density)')

    def test_group_by_together_with_y_log_scale(self):
        """Test that group_by and y_log_scale work together."""
        fig, ax, df = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=True,
            y_log_scale=True
        ).values()
        self.assertEqual(ax.get_yscale(), 'log')
        self.assertEqual(ax.get_ylabel(), 'log(Count)')

    def test_group_by_separate_with_y_log_scale(self):
        """Test that group_by creates separate plots with y_log_scale."""
        fig, axs, df = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=False,
            y_log_scale=True
        ).values()
        # Ensure axs is iterable
        if not isinstance(axs, list):
            axs = [axs]
        for ax in axs:
            self.assertEqual(ax.get_yscale(), 'log')
            self.assertEqual(ax.get_ylabel(), 'log(Count)')

    def test_group_by_max_groups_default_guardrail_rejects_excess_groups(self):
        """Default threshold should reject excessive grouped facet plotting."""
        adata = self._make_many_groups_adata(n_groups=25)
        with self.assertRaisesRegex(ValueError, "exceeds `max_groups`"):
            histogram(
                adata,
                feature='marker1',
                group_by='many_groups',
                facet=True,
            )

    def test_group_by_max_groups_override_allows_grouped_plot(self):
        """Custom positive max_groups should allow larger grouped plots."""
        n_groups = 25
        adata = self._make_many_groups_adata(n_groups=n_groups)

        fig, axs, _ = histogram(
            adata,
            feature='marker1',
            group_by='many_groups',
            facet=True,
            max_groups=30,
        ).values()
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        self.assertEqual(len(axs), n_groups)

    def test_group_by_max_groups_unlimited_disables_guardrail(self):
        """max_groups='unlimited' should disable grouped guardrail validation."""
        adata = self._make_many_groups_adata(n_groups=25)

        fig, ax, _ = histogram(
            adata,
            feature='marker1',
            group_by='many_groups',
            together=True,
            max_groups='unlimited',
        ).values()
        self.assertIsNotNone(fig)
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_group_by_max_groups_none_uses_default_threshold(self):
        """Explicit None should resolve to default threshold behavior."""
        adata = self._make_many_groups_adata(n_groups=25)
        with self.assertRaisesRegex(ValueError, "exceeds `max_groups`"):
            histogram(
                adata,
                feature='marker1',
                group_by='many_groups',
                together=True,
                max_groups=None,
            )

    def test_group_by_invalid_max_groups_raises_value_error(self):
        """Invalid max_groups values should fail fast."""
        for value in [0, "bad", True]:
            with self.subTest(max_groups=value):
                with self.assertRaises(ValueError):
                    histogram(
                        self.adata,
                        feature='marker1',
                        group_by='annotation2',
                        together=True,
                        max_groups=value,
                    )

    def test_overlay_options(self):
        fig, ax, df = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=True,
            multiple="layer",
            element="step"
        ).values()
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_layer(self):
        # Create synthetic data with known values for a subset of the data
        subset_size = 3
        synthetic_data = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])

        # Update a subset of the main matrix with different values to ensure
        # the layer is used
        self.adata.X[:subset_size, :] = np.array([
            [10, 10, 10],
            [20, 20, 20],
            [30, 30, 30]
        ])

        # Add the synthetic data as a layer for the subset
        self.adata.layers['layer1'] = np.zeros_like(self.adata.X)
        self.adata.layers['layer1'][:subset_size, :] = synthetic_data

        # Plot the histogram using the layer data for the subset
        fig, ax, df = histogram(
            self.adata[:subset_size],
            feature='marker1',
            layer='layer1',
            bins=[0.5, 1.5, 2.5, 3.5]
        ).values()

        # Check the histogram bars
        bars = ax.patches
        self.assertEqual(len(bars), 3)  # Expecting 3 bars for synthetic data

        # Check the height and position of each bar to match the synthetic data
        for i, value in enumerate([1, 2, 3]):
            # Each bar should have a height of 1
            self.assertAlmostEqual(bars[i].get_height(), 1)
            # Bar centers should match the synthetic values
            self.assertAlmostEqual(
                bars[i].get_x() + bars[i].get_width() / 2, value
            )

    def test_ax_passed_as_argument(self):
        # Supported mode 1: single-axes histogram with external ax.
        fig, ax = plt.subplots()
        returned_fig, returned_ax, df = histogram(
            self.adata,
            feature='marker1',
            ax=ax
        ).values()
        # Check that the passed ax is the one that is returned
        self.assertEqual(id(returned_ax), id(ax))

        # Check that the passed fig is the one that is returned
        self.assertIs(fig, returned_fig)

        # Supported mode 2: grouped+together histogram with external ax.
        fig_grouped, ax_grouped = plt.subplots()
        returned_grouped_fig, returned_grouped_ax, _ = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=True,
            ax=ax_grouped,
        ).values()

        self.assertIs(fig_grouped, returned_grouped_fig)
        self.assertIs(ax_grouped, returned_grouped_ax)

        # Check that returned axes are valid Axes objects.
        self.assertIsInstance(returned_ax, mpl.axes.Axes)
        self.assertIsInstance(returned_grouped_ax, mpl.axes.Axes)

    def test_external_ax_guardrail_modes(self):
        # Reject grouped-separate mode with external ax.
        fig_1, ax_1 = plt.subplots()
        with self.assertRaisesRegex(
            ValueError,
            "External ax is only supported for single-axes histogram"
        ):
            histogram(
                self.adata,
                feature='marker1',
                group_by='annotation2',
                together=False,
                ax=ax_1,
            )

        # Reject facet mode with external ax.
        fig_2, ax_2 = plt.subplots()
        with self.assertRaisesRegex(
            ValueError,
            "External ax is only supported for single-axes histogram"
        ):
            histogram(
                self.adata,
                feature='marker1',
                group_by='annotation2',
                facet=True,
                ax=ax_2,
            )

        # Positive external-ax modes are covered in test_ax_passed_as_argument.

    def test_default_first_feature(self):
        with self.assertWarns(UserWarning) as warning:
            bin_edges = list(np.linspace(0.5, 100.5, 101))
            fig, ax, df = histogram(self.adata, bins=bin_edges).values()

        warning_message = ("No feature or annotation specified. "
                           "Defaulting to the first feature: 'marker1'.")
        self.assertEqual(str(warning.warning), warning_message)

        # Check that ax is an Axes object
        self.assertIsInstance(ax, mpl.axes.Axes)

        # Check the number of bars matches the number of cells
        bars = ax.patches
        self.assertEqual(len(bars), 100)

    def test_histogram_feature_integer_bins(self):
        custom_bins = 10  # Specify number of bins as an integer

        fig, ax, df = histogram(
            self.adata, 
            feature='marker1', 
            bins=custom_bins
        ).values()

        # Check the number of bins used
        bars = ax.patches
        self.assertEqual(len(bars), custom_bins)

        # Check that ax is an Axes object
        self.assertIsInstance(ax, mpl.axes.Axes)

    def test_default_bins_calculation(self):
        """No bins argument should use Rice-rule fallback."""
        expected_bins = max(int(2 * (self.adata.shape[0] ** (1 / 3))), 1)

        # No bins parameter passed
        fig, ax, df = histogram(self.adata, feature='marker1').values()
        self.assertEqual(len(ax.patches), expected_bins)
        self.assertEqual(len(df), expected_bins)
        self.assertEqual(
            set(df.columns),
            {'count', 'bin_left', 'bin_right', 'bin_center'},
        )

    def test_default_like_bins_calculation(self):
        """Default-like bins values should use Rice-rule fallback."""
        expected_bins = max(int(2 * (self.adata.shape[0] ** (1 / 3))), 1)

        for bins_value in [None, 'auto', 'none', '']:
            with self.subTest(bins=bins_value):
                fig, ax, df = histogram(
                    self.adata,
                    feature='marker1',
                    bins=bins_value,
                ).values()

                self.assertEqual(len(ax.patches), expected_bins)
                self.assertEqual(len(df), expected_bins)
                self.assertEqual(
                    set(df.columns),
                    {'count', 'bin_left', 'bin_right', 'bin_center'},
                )

    def test_grouped_separate_ignores_multiple(self):
        """Grouped separate mode should ignore irrelevant multiple settings."""
        fig, axs, _ = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            together=False,
            multiple="fill",
        ).values()
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertIsInstance(axs, list)
        self.assertEqual(
            len(axs),
            self.adata.obs["annotation2"].dropna().nunique(),
        )

    def test_facet_requires_group_by(self):
        """Test that facet mode requires group_by parameter"""
        with self.assertRaisesRegex(
            ValueError,
            "group_by must be specified when facet=True."
        ):
            histogram(
                self.adata,
                feature='marker1',
                facet=True,
            )

    def test_facet_conflicts_with_together_true(self):
        """Test that facet mode conflicts with together=True"""
        with self.assertRaisesRegex(
            ValueError,
            "Cannot use together=True with facet=True"
        ):
            histogram(
                self.adata,
                feature='marker1',
                group_by='annotation2',
                together=True,
                facet=True,
            )

    def test_facet_plot_smoke_and_structure(self):
        """Facet path returns expected structure and plotted content."""
        fig, ax, df = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            facet=True,
        ).values()

        # Basic structure checks
        self.assertIsNotNone(fig)
        self.assertIsNotNone(df)
        self.assertIsInstance(ax, (list, np.ndarray),
                              "Facet output should be a multi-axis collection.")

        # Check the number of facet axes matches group count
        unique_groups = self.adata.obs['annotation2'].dropna().unique()
        self.assertEqual(len(ax), len(unique_groups),
                         f"Expected {len(unique_groups)}"
                         f" facet plots, got {len(ax)}.")

        # Lightweight bar-level presence checks only.
        for i, axis in enumerate(ax):
            self.assertGreater(
                len(axis.patches),
                0,
                f"Facet {i} should contain at least one bar patch."
            )

    def test_facet_plot_titles_and_label_policy(self):
        """Facet titles map to groups and labels follow figure-level policy."""
        fig, ax, df = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            facet=True,
        ).values()

        # Ensure ax is iterable for consistent handling
        ax = ax if isinstance(ax, (list, np.ndarray)) else [ax]
        unique_groups = self.adata.obs['annotation2'].dropna().unique()

        # Titles must map to expected groups and labels must be per-figure.
        for i, axis in enumerate(ax):
            title = axis.get_title()
            self.assertTrue(title, f"Facet {i} is missing a title.")
            self.assertTrue(any(str(group) in title
                            for group in unique_groups),
                            f"Title '{title}' does not contain"
                            f"any expected group names.")
            self.assertEqual(axis.get_xlabel(), '',
                             f"Facet {i} x-label should be empty.")
            self.assertEqual(axis.get_ylabel(), '',
                             f"Facet {i} y-label should be empty.")

        # Figure-level labels should be set in facet mode.
        self.assertIsNotNone(fig._supxlabel)
        self.assertIsNotNone(fig._supylabel)
        self.assertEqual(fig._supxlabel.get_text(), 'marker1')
        self.assertEqual(fig._supylabel.get_text(), 'Count')

    def test_facet_plot_density_stat_label_policy(self):
        """Facet figure-level y label reflects non-default stat mapping."""
        fig, ax, df = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            facet=True,
            stat='density',
        ).values()

        # Check that figure-level y label reflects 'density' stat when specified.
        self.assertIsNotNone(fig._supylabel)
        self.assertEqual(fig._supylabel.get_text(), 'Density')

    def test_facet_plot_categorical_annotation(self):
        """Test facet mode with categorical annotations"""
        fig, axs, df = histogram(
            self.adata,
            annotation='annotation1',
            group_by='annotation2',
            facet=True,
        ).values()

        # Ensure axs is iterable for consistent handling
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        expected_groups = self.adata.obs['annotation2'].dropna().nunique()
        self.assertEqual(len(axs), expected_groups)

        # Check that data is plotted in each facet
        for axis in axs:
            self.assertGreater(len(axis.patches), 0)

        # Check figure-level labels are set appropriately
        self.assertIsNotNone(fig._supxlabel)
        self.assertIsNotNone(fig._supylabel)
        self.assertEqual(fig._supxlabel.get_text(), 'annotation1')
        self.assertEqual(fig._supylabel.get_text(), 'Count')

    def test_facet_plot_numeric_annotation(self):
        """Facet mode should support numeric annotations sourced from obs."""
        adata = self.adata.copy()
        adata.obs['annotation_numeric'] = np.arange(
            adata.n_obs,
            dtype=np.float32,
        )

        fig, axs, _ = histogram(
            adata,
            annotation='annotation_numeric',
            group_by='annotation2',
            facet=True,
        ).values()

        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        expected_groups = adata.obs['annotation2'].dropna().nunique()
        self.assertIsNotNone(fig)
        self.assertEqual(len(axs), expected_groups)

        for axis in axs:
            self.assertGreater(len(axis.patches), 0)

        self.assertIsNotNone(fig._supxlabel)
        self.assertIsNotNone(fig._supylabel)
        self.assertEqual(fig._supxlabel.get_text(), 'annotation_numeric')
        self.assertEqual(fig._supylabel.get_text(), 'Count')

    def test_facet_ncol_layout_hints(self):
        """Facet ncol supports positive int and documented auto behavior."""
        # Explicit two-column layout should create two facet columns.
        fig, axs, _ = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            facet=True,
            facet_ncol=2,
        ).values()
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        x_positions = {round(axis.get_position().x0, 4) for axis in axs}
        self.assertGreaterEqual(len(x_positions), 2)

        # Documented default-like input should use auto layout (one column for 3 groups).
        fig, axs, _ = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            facet=True,
            facet_ncol='auto',
        ).values()
        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        x_positions = {round(axis.get_position().x0, 4) for axis in axs}
        self.assertEqual(len(x_positions), 1)

        # Invalid values should fail fast.
        with self.assertRaises(ValueError):
            histogram(
                self.adata,
                feature='marker1',
                group_by='annotation2',
                facet=True,
                facet_ncol='bad',
            )
        with self.assertRaises(ValueError):
            histogram(
                self.adata,
                feature='marker1',
                group_by='annotation2',
                facet=True,
                facet_ncol=0,
            )

    def test_facet_figure_size_hints(self):
        """Facet figure-size hints should accept valid values and sanitize invalid ones."""
        # Check that valid figure size hints are applied to the facet figure.
        fig, _, _ = histogram(
            self.adata,
            feature='marker1',
            group_by='annotation2',
            facet=True,
            facet_fig_width=11,
            facet_fig_height=3.5,
        ).values()
        self.assertAlmostEqual(fig.get_figwidth(), 11.0, places=2)
        self.assertAlmostEqual(fig.get_figheight(), 3.5, places=2)

        # Invalid hints should fail fast.
        for width, height in [('wide', 'tall'), (-1, 0)]:
            with self.subTest(facet_fig_width=width, facet_fig_height=height):
                with self.assertRaises(ValueError):
                    histogram(
                        self.adata,
                        feature='marker1',
                        group_by='annotation2',
                        facet=True,
                        facet_fig_width=width,
                        facet_fig_height=height,
                    )

    def test_facet_figure_size_hints_require_pair(self):
        """One-sided facet figure-size hints should raise a ValueError."""
        with self.assertRaises(ValueError):
            histogram(
                self.adata,
                feature='marker1',
                group_by='annotation2',
                facet=True,
                facet_fig_width=11,
            )
        with self.assertRaises(ValueError):
            histogram(
                self.adata,
                feature='marker1',
                group_by='annotation2',
                facet=True,
                facet_fig_height=3.5,
            )

    def test_non_facet_figure_size_hints_are_ignored(self):
        """Non-facet calls should ignore facet-only figure-size hints."""
        baseline_fig, baseline_ax, _ = histogram(
            self.adata,
            feature='marker1',
            facet=False,
        ).values()

        for hint_kwargs in (
            {'facet_fig_width': 8},
            {'facet_fig_height': 5},
            {'facet_fig_width': 8, 'facet_fig_height': 5},
        ):
            with self.subTest(hints=hint_kwargs):
                fig, ax, _ = histogram(
                    self.adata,
                    feature='marker1',
                    facet=False,
                    **hint_kwargs,
                ).values()
                self.assertAlmostEqual(
                    fig.get_figwidth(),
                    baseline_fig.get_figwidth(),
                    places=6,
                )
                self.assertAlmostEqual(
                    fig.get_figheight(),
                    baseline_fig.get_figheight(),
                    places=6,
                )
                self.assertGreater(len(ax.patches), 0)
                self.assertEqual(len(ax.patches), len(baseline_ax.patches))

    def test_facet_tick_rotation_zero_matches_default_behavior(self):
        """Explicit zero rotation should match omitted rotation behavior."""
        fig_default, _, _ = histogram(
            self.adata,
            annotation='annotation1',
            group_by='annotation2',
            facet=True,
        ).values()
        fig_zero, _, _ = histogram(
            self.adata,
            annotation='annotation1',
            group_by='annotation2',
            facet=True,
            facet_tick_rotation=0,
        ).values()

        self.assertAlmostEqual(fig_default.get_figwidth(), fig_zero.get_figwidth(), places=6)
        self.assertAlmostEqual(fig_default.get_figheight(), fig_zero.get_figheight(), places=6)

    def test_facet_long_label_geometry_adjustment_without_size_hints(self):
        """Long rotated categorical labels should increase default facet geometry."""
        adata = self._make_long_label_facet_adata(include_short=True)

        fig_short, _, _ = histogram(
            adata,
            annotation='annotation_short',
            group_by='annotation2',
            facet=True,
            facet_ncol=2,
            facet_tick_rotation=45,
        ).values()
        fig_long, _, _ = histogram(
            adata,
            annotation='annotation_long',
            group_by='annotation2',
            facet=True,
            facet_ncol=2,
            facet_tick_rotation=45,
        ).values()

        self.assertGreater(fig_long.get_figwidth(), fig_short.get_figwidth())
        self.assertGreater(fig_long.get_figheight(), fig_short.get_figheight())

    def test_facet_long_label_geometry_respects_explicit_size_hints(self):
        """Explicit facet figure-size hints should remain authoritative."""
        adata = self._make_long_label_facet_adata()

        fig, _, _ = histogram(
            adata,
            annotation='annotation_long',
            group_by='annotation2',
            facet=True,
            facet_ncol=2,
            facet_tick_rotation=60,
            facet_fig_width=10,
            facet_fig_height=4,
        ).values()

        self.assertAlmostEqual(fig.get_figwidth(), 10.0, places=2)
        self.assertAlmostEqual(fig.get_figheight(), 4.0, places=2)

    def test_facet_plot_shared_bins_consistency_numeric(self):
        """Numeric facets keep shared bins for int/default-like bins inputs."""
        # Unbalanced groups: each group occupies only part of the global range.
        # If bins are computed per-group (bad path), centers/ticks may diverge.
        adata = anndata.AnnData(
            np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]],
                     dtype=np.float32),
            obs=pd.DataFrame(
                {'annotation2': ['g1', 'g1', 'g1', 'g2', 'g2', 'g2']},
            index=[f'cell_{i}' for i in range(6)],
            ),
            var=pd.DataFrame(index=['marker1']),
        )

        # Test one explicit and one default-like bins path.
        for bins_value in [4, None]:
            with self.subTest(bins=bins_value):
                fig, axs, df = histogram(
                    adata,
                    feature='marker1',
                    group_by='annotation2',
                    facet=True,
                    bins=bins_value,
                ).values()

                axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]

                # Capture the bin centers and ticks from the first facet
                first_xlim = np.round(np.array(axs[0].get_xlim()), 6)
                first_xticks = np.round(np.array(axs[0].get_xticks()), 6)
                first_yticks = np.round(np.array(axs[0].get_yticks()), 6)
                first_centers = np.round(
                    np.array([
                        patch.get_x() + patch.get_width() / 2
                        for patch in axs[0].patches
                    ]),
                    6
                )

                # Check that all facets have the same bin centers and ticks
                for axis in axs[1:]:
                    centers = np.round(
                        np.array([
                            patch.get_x() + patch.get_width() / 2
                            for patch in axis.patches
                        ]),
                        6
                    )
                    self.assertTrue(
                        np.array_equal(centers, first_centers),
                        "Facet numeric bin centers should remain shared across panels."
                    )
                    self.assertTrue(
                        np.array_equal(np.round(np.array(axis.get_xlim()), 6), first_xlim),
                        "Facet numeric x-limits should remain shared across panels."
                    )
                    self.assertTrue(
                        np.array_equal(np.round(np.array(axis.get_xticks()), 6), first_xticks),
                        "Facet numeric x-ticks should remain shared across panels."
                    )
                    self.assertTrue(
                        np.array_equal(np.round(np.array(axis.get_yticks()), 6), first_yticks),
                        "Facet numeric y-ticks should remain shared across panels."
                    )

                # Check that the returned DataFrame has expected structure and content
                self.assertEqual(
                    set(df.columns),
                    {'count', 'bin_left', 'bin_right', 'bin_center', 'annotation2'},
                )
                self.assertNotIn('marker1', df.columns)
                self.assertEqual(set(df['annotation2']), {'g1', 'g2'})
                self.assertEqual(df['count'].sum(), adata.n_obs)
                grouped_edges = [
                    (
                        np.round(group_df['bin_left'].to_numpy(), 6),
                        np.round(group_df['bin_right'].to_numpy(), 6),
                    )
                    for _, group_df in df.groupby('annotation2')
                ]
                self.assertEqual(len(grouped_edges), 2)
                self.assertTrue(np.array_equal(grouped_edges[0][0], grouped_edges[1][0]))
                self.assertTrue(np.array_equal(grouped_edges[0][1], grouped_edges[1][1]))

    def test_facet_plot_shared_bins_consistency_categorical(self):
        """Facet categorical bins stay aligned even with missing labels."""
        # Build unbalanced facet groups where some labels are missing per group.
        adata = anndata.AnnData(
            np.arange(1, 10, dtype=np.float32).reshape(-1, 1),
            obs=pd.DataFrame(
            {
                'annotation1': pd.Categorical(
                    ['A', 'A', 'B', 'A', 'C', 'C', 'B', 'C', 'A'],
                    categories=['A', 'B', 'C'],
                ),
                    'annotation2': ['g1', 'g1', 'g1',
                                    'g2', 'g2', 'g2',
                                    'g3', 'g3', 'g3'],
            },
            index=[f'cell_{i}' for i in range(9)],
            ),
            var=pd.DataFrame(index=['marker1']),
        )

        fig, axs, df = histogram(
            adata,
            annotation='annotation1',
            group_by='annotation2',
            facet=True,
        ).values()

        axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]

        # Guardrail: this fixture must include missing labels per group.
        group_uniques = adata.obs.groupby('annotation2')['annotation1'].nunique()
        self.assertTrue(any(group_uniques < 3))

        # Check that bin centers are shared across facets
        global_centers = set()
        for axis in axs:
            global_centers.update(
                np.round(
                    [patch.get_x() + patch.get_width() / 2
                     for patch in axis.patches],
                    6,
                )
            )
        self.assertEqual(
            len(global_centers),
            3,
            "Expected 3 categorical slots (A/B/C) to be preserved globally."
        )

        # Check that ticks are shared across facets
        first_xticks = np.round(axs[0].get_xticks(), 6)
        first_yticks = np.round(np.array(axs[0].get_yticks()), 6)
        for axis in axs[1:]:
            self.assertTrue(np.array_equal(np.round(axis.get_xticks(), 6), first_xticks))
            self.assertTrue(
                np.array_equal(np.round(np.array(axis.get_yticks()), 6), first_yticks),
                "Facet categorical y-ticks should remain shared across panels."
            )

        # Check that the returned DataFrame has expected structure and content
        self.assertEqual(
            set(df.columns),
            {'count', 'bin_left', 'bin_right', 'bin_center', 'annotation2'},
        )
        self.assertNotIn('annotation1', df.columns)
        self.assertEqual(set(df['annotation2']), {'g1', 'g2', 'g3'})
        self.assertEqual(df['count'].sum(), adata.n_obs)
        self.assertEqual(
            {str(value) for value in df['bin_center'].unique()},
            {'A', 'B', 'C'},
        )
        for _, group_df in df.groupby('annotation2'):
            self.assertEqual(
                {str(value) for value in group_df['bin_center']},
                {'A', 'B', 'C'},
            )

    def test_facet_plot_categorical_annotation_ignores_bins(self):
        """Facet categorical annotations should ignore custom bins values."""
        fig_small, axs_small, _ = histogram(
            self.adata,
            annotation='annotation1',
            group_by='annotation2',
            facet=True,
            bins=2,
        ).values()
        fig_large, axs_large, _ = histogram(
            self.adata,
            annotation='annotation1',
            group_by='annotation2',
            facet=True,
            bins=99,
        ).values()

        axs_small = axs_small if isinstance(axs_small, (list, np.ndarray)) else [axs_small]
        axs_large = axs_large if isinstance(axs_large, (list, np.ndarray)) else [axs_large]

        self.assertEqual(len(axs_small), len(axs_large))

        axis_small = axs_small[0]
        axis_large = axs_large[0]
        small_centers = [
            patch.get_x() + patch.get_width() / 2
            for patch in axis_small.patches
        ]
        large_centers = [
            patch.get_x() + patch.get_width() / 2
            for patch in axis_large.patches
        ]
        self.assertEqual(small_centers, large_centers)
        self.assertEqual(
            [tick.get_text() for tick in axis_small.get_xticklabels()],
            [tick.get_text() for tick in axis_large.get_xticklabels()],
        )


if __name__ == '__main__':
    unittest.main()
