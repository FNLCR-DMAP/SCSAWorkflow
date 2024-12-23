import unittest
from unittest import mock
import pandas as pd
import anndata
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spac.visualization import boxplot
matplotlib.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestBoxplot(unittest.TestCase):

    def setUp(self):
        """Set up testing environment."""
        X = pd.DataFrame({
            'feature1': [1, 3, 5, 7],
            'feature2': [2, 4, 6, 8],
        })

        annotation = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2'
            ]
        })

        self.adata = anndata.AnnData(X=X, obs=annotation)

    def test_returns_correct_types(self):
        """Test if correct types are returned."""
        fig, ax, df = boxplot(self.adata)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)

    def test_second_annotation_mode(self):
        """Test if second annotation mode works as expected."""
        self.adata.obs['treatment'] = ['Drug1', 'Drug1', 'Drug2', 'Drug2']
        fig, ax, df = boxplot(
            self.adata,
            annotation='phenotype',
            second_annotation='treatment',
            orient='v'
        )
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        for label in ['phenotype1', 'phenotype2']:
            self.assertIn(label, xtick_labels)
        legend_labels = [text.get_text() for text in ax.legend().get_texts()]
        for label in ['Drug1', 'Drug2']:
            self.assertIn(label, legend_labels)

    def test_single_annotation_single_feature(self):
        """Test if annotation mode works as expected with a single feature."""
        fig, ax, df = boxplot(
            self.adata, features=['feature1'],
            annotation='phenotype', orient='v'
        )
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        expected_unique_labels = ['phenotype1', 'phenotype2']
        for label in expected_unique_labels:
            self.assertIn(label, xtick_labels)

    def test_single_annotation_multiple_features(self):
        """Test when only one annotation but multiple features."""
        fig, ax, df = boxplot(
            self.adata,
            features=['feature1', 'feature2'],
            annotation='phenotype', orient='v'
        )
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        expected_unique_labels = ['feature1', 'feature2']
        for label in expected_unique_labels:
            self.assertIn(label, xtick_labels)

    def test_multiple_features_mode(self):
        """Test if multiple features mode works as expected."""
        fig, ax, df = boxplot(
            self.adata,
            features=['feature1', 'feature2'],
            orient='v'
        )

        # Get x-tick labels from the plot
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]

        # Check if x-tick labels match the feature names
        expected_features = ['feature1', 'feature2']
        self.assertEqual(set(xtick_labels), set(expected_features))

    def test_orient_kwarg(self):
        """Test for the orient parameter passed via **kwargs."""
        fig, ax, df = boxplot(
            self.adata, features=['feature1'],
            annotation='phenotype', orient='h'
        )
        # Check if the x-axis ticks contain the expected annotations
        ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        expected_unique_labels = ['phenotype1', 'phenotype2']
        for label in expected_unique_labels:
            self.assertIn(label, ytick_labels)

    def test_log_scale(self):
        """Test for the log_scale parameter."""
        # Hard-coded expected values after np.log1p transformation
        expected_values = np.array(
            [0.693147, 1.386294, 1.791759, 2.079441], dtype=np.float64
        )

        fig, ax, df = boxplot(
            self.adata, features=['feature1'], log_scale=True
        )

        # Check that the y-axis label is 'log(Intensity)'
        self.assertEqual(ax.get_ylabel(), 'log(Intensity)')

        # Check that the data has been transformed via np.log1p
        transformed_values = df['feature1'].values
        np.testing.assert_allclose(
            transformed_values, expected_values, rtol=1e-5
        )

        # Check that the y-axis scale is 'linear'
        self.assertEqual(ax.get_yscale(), 'linear')

        # Test with zero values
        self.adata.X[0, 0] = 0  # Introduce a zero value

        # Hard-coded expected values after np.log1p transformation with zero
        expected_values_zero = np.array(
            [0.0, 1.386294, 1.791759, 2.079441], dtype=np.float64
        )

        fig, ax, df = boxplot(
            self.adata, features=['feature1'], log_scale=True
        )

        # Check that the y-axis label is still 'log(Intensity)'
        self.assertEqual(ax.get_ylabel(), 'log(Intensity)')

        # Check that the data has been transformed via np.log1p
        transformed_values = df['feature1'].values
        np.testing.assert_allclose(
            transformed_values, expected_values_zero, rtol=1e-5
        )

        # The y-axis scale should still be 'linear'
        self.assertEqual(ax.get_yscale(), 'linear')

    def test_log1p_transformation(self):
        """Test if np.log1p transformation is applied correctly."""
        # Introduce a dataset with zeros
        X = pd.DataFrame({
            'feature1': [0, 1, 2, 3]
        })

        annotation = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2'
            ]
        })

        adata = anndata.AnnData(X=X, obs=annotation)

        # Hard-coded expected values after np.log1p transformation
        expected_values = np.array(
            [0.0, 0.693147, 1.098612, 1.386294], dtype=np.float32
        )

        # Create a boxplot and capture the transformed DataFrame
        fig, ax, transformed_df = boxplot(
            adata, features=['feature1'], log_scale=True
        )

        # Extract the log1p transformed values from the DataFrame for plotting
        transformed_values = transformed_df['feature1'].values.flatten()

        # Debugging: print DataFrame after transformation
        print("DataFrame after log1p transformation:\n", transformed_values)

        # Compare the transformed values with the expected values
        np.testing.assert_allclose(
            transformed_values, expected_values, rtol=1e-5
        )

    @mock.patch('builtins.print')
    def test_negative_values(self, mock_print):
        """Test the negative values disable log scale and print a message."""
        # Introduce a dataset with negative values
        X = pd.DataFrame({
            'feature1': [-1, 0, 1, 2]
        })

        annotation = pd.DataFrame({
            'phenotype': [
                'phenotype1',
                'phenotype1',
                'phenotype2',
                'phenotype2'
            ]
        })
        annotation.index = annotation.index.astype(str)
        adata = anndata.AnnData(X=X.astype(np.float32), obs=annotation)

        # Create a boxplot and capture the print output
        fig, ax, df = boxplot(adata, features=['feature1'], log_scale=True)

        # Extract the printed messages
        print_calls = [call.args[0] for call in mock_print.call_args_list]

        # Expected message
        expected_msg = (
            "There are negative values in this data, disabling the log scale."
        )

        self.assertIn(expected_msg, print_calls)

        # Ensure the y-axis label is 'Intensity'
        self.assertEqual(ax.get_ylabel(), 'Intensity')

        # Expected values (should be the same as input
        # since loig_scale is disabled)
        expected_values = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        # Check that the data has not been transformed
        np.testing.assert_array_equal(
            df['feature1'].values,
            expected_values
        )

    def test_single_feature_orientation(self):
        """
        Test single feature plotting to use the correct axis
        based on orientation.
        """
        # Test for vertical orientation
        fig, ax, df = boxplot(self.adata, features=['feature1'], orient='v')
        self.assertEqual(ax.get_ylabel(), 'Intensity')

        # Test for horizontal orientation
        fig, ax, df = boxplot(self.adata, features=['feature1'], orient='h')
        self.assertEqual(ax.get_xlabel(), 'Intensity')

    def test_axis_labels(self):
        """Test x-axis, y-axis, and log labeling."""
        # Test for vertical orientation without log scale
        fig, ax, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='v'
        )
        self.assertEqual(ax.get_xlabel(), 'phenotype')
        self.assertEqual(ax.get_ylabel(), 'Intensity')

        # Test for vertical orientation with log scale
        fig, ax, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='v',
            log_scale=True
        )
        self.assertEqual(ax.get_xlabel(), 'phenotype')
        self.assertEqual(ax.get_ylabel(), 'log(Intensity)')

        # Test for horizontal orientation without log scale
        fig, ax, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='h'
        )
        self.assertEqual(ax.get_xlabel(), 'Intensity')
        self.assertEqual(ax.get_ylabel(), 'phenotype')

        # Test for horizontal orientation with log scale
        fig, ax, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='h',
            log_scale=True
        )
        self.assertEqual(ax.get_xlabel(), 'log(Intensity)')
        self.assertEqual(ax.get_ylabel(), 'phenotype')

    def test_single_feature_labeling(self):
        """Test if single feature name is displayed correctly on the x-axis."""
        fig, ax, df = boxplot(self.adata, features=['feature1'], orient='v')
        xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        self.assertEqual(xtick_labels, ['feature1'])

        fig, ax, df = boxplot(self.adata, features=['feature1'], orient='h')
        ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        self.assertEqual(ytick_labels, ['feature1'])


if __name__ == '__main__':
    unittest.main()
