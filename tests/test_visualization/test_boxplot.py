import unittest
from unittest import mock
import pandas as pd
import anndata
import numpy as np
import matplotlib
import plotly.graph_objects as go
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
        # Test non-interactive mode
        fig, df, metrics = boxplot(self.adata, interactive=False, return_metrics=True)
        self.assertIsInstance(fig, str)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(metrics, pd.DataFrame)

        # Test interactive mode
        fig, df = boxplot(self.adata, interactive=True)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(df, pd.DataFrame)


    def test_single_annotation_single_feature(self):
        """Test if annotation mode works as expected with a single feature."""
        fig, df = boxplot(
            self.adata, features=['feature1'],
            annotation='phenotype', 
            orient='v',
            interactive=True
        )

        # Get boxplot names from figure
        box_name_data = [box.name for box in fig.data]

        # Check if box names match the expected phenotype annotations
        expected_unique_labels = ['phenotype1', 'phenotype2']
        for label in expected_unique_labels:
            self.assertIn(label, box_name_data)


    def test_single_annotation_multiple_features(self):
        """Test when only one annotation but multiple features."""
        fig, df = boxplot(
            self.adata,
            features=['feature1', 'feature2'],
            annotation='phenotype', 
            orient='v',
            interactive=True,
        )

        # Get x-axis labels from figure
        x_labels = fig.data[0].x

        # Check if x-axis labels match the feature names
        expected_features = ['feature1', 'feature2']
        self.assertEqual(set(x_labels), set(expected_features))


    def test_multiple_features_mode(self):
        """Test if multiple features mode works as expected."""
        fig, df = boxplot(
            self.adata,
            features=['feature1', 'feature2'],
            orient='v',
            interactive=True,
        )

        #  Get x-axis labels from figure
        x_labels = [x_data for boxplot in fig.data for x_data in boxplot.x]

        # Check if x-axis labels match the feature names
        expected_features = ['feature1', 'feature2']
        self.assertEqual(set(x_labels), set(expected_features))

    def test_orient(self):
        """Test for the orient parameter."""
        fig, df = boxplot(
            self.adata, features=['feature1'],
            annotation='phenotype', 
            orient='h',
            interactive=True,
        )
        # Get y-axis labels from figure
        y_labels = [y_data for boxplot in fig.data for y_data in boxplot.y]

        # Check if y-axis labels match the feature names (meaning orientation is correct)
        expected_features = ['feature1']
        self.assertEqual(set(y_labels), set(expected_features))

    def test_log_scale(self):
        """Test for the log_scale parameter."""
        # Hard-coded expected values after np.log1p transformation
        expected_values = np.array(
            [0.693147, 1.386294, 1.791759, 2.079441], dtype=np.float64
        )

        fig, df = boxplot(
            self.adata, 
            features=['feature1'],
            log_scale=True,
            interactive=True
        )

        # Check that the y-axis label is 'log(Intensity)'
        self.assertEqual(fig.layout.yaxis.title.text, 'log(Intensity)')

        # Check that the data has been transformed via np.log1p
        transformed_values = df['feature1'].values
        np.testing.assert_allclose(
            transformed_values, expected_values, rtol=1e-5
        )

        # Test with zero values
        self.adata.X[0, 0] = 0  # Introduce a zero value

        # Hard-coded expected values after np.log1p transformation with zero
        expected_values_zero = np.array(
            [0.0, 1.386294, 1.791759, 2.079441], dtype=np.float64
        )

        fig, df = boxplot(
            self.adata, 
            features=['feature1'], 
            log_scale=True,
            interactive=True
        )

        # Check that the y-axis label is still 'log(Intensity)'
        self.assertEqual(fig.layout.yaxis.title.text, 'log(Intensity)')

        # Check that the data has been transformed via np.log1p
        transformed_values = df['feature1'].values
        np.testing.assert_allclose(
            transformed_values, expected_values_zero, rtol=1e-5
        )

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
        fig, df = boxplot(
            adata, features=['feature1'], log_scale=True
        )

        # Extract the log1p transformed values from the DataFrame for plotting
        transformed_values = df['feature1'].values.flatten()

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
        fig, df = boxplot(
            adata, 
            features=['feature1'], 
            log_scale=True,
            interactive=True
        )

        # Extract the printed messages
        print_calls = [call.args[0] for call in mock_print.call_args_list]

        # Expected message
        expected_msg = (
            "There are negative values in this data, disabling the log scale."
        )

        self.assertIn(expected_msg, print_calls)

        # Ensure the y-axis label is 'Intensity'
        self.assertEqual(fig.layout.yaxis.title.text, 'Intensity')

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
        fig, df = boxplot(
            self.adata, 
            features=['feature1'], 
            orient='v', 
            interactive=True
        )
        self.assertEqual(fig.layout.yaxis.title.text, 'Intensity')

        # Test for horizontal orientation
        fig, df = boxplot(self.adata, features=['feature1'], orient='h', interactive=True)
        self.assertEqual(fig.layout.xaxis.title.text, 'Intensity')

    def test_axis_labels(self):
        """Test x-axis, y-axis, and log labeling."""
        # Test for vertical orientation without log scale
        fig, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='v',
            interactive=True
        )
        self.assertEqual(fig.layout.xaxis.title.text, 'phenotype')
        self.assertEqual(fig.layout.yaxis.title.text, 'Intensity')

        # Test for vertical orientation with log scale
        fig, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='v',
            log_scale=True,
            interactive=True
        )
        self.assertEqual(fig.layout.xaxis.title.text, 'phenotype')
        self.assertEqual(fig.layout.yaxis.title.text, 'log(Intensity)')

        # Test for horizontal orientation without log scale
        fig, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='h',
            interactive=True
        )
        self.assertEqual(fig.layout.xaxis.title.text, 'Intensity')
        self.assertEqual(fig.layout.yaxis.title.text, 'phenotype')

        # Test for horizontal orientation with log scale
        fig, df = boxplot(
            self.adata,
            features=['feature1'],
            annotation='phenotype',
            orient='h',
            log_scale=True,
            interactive=True
        )
        self.assertEqual(fig.layout.xaxis.title.text, 'log(Intensity)')
        self.assertEqual(fig.layout.yaxis.title.text, 'phenotype')

    def test_single_feature_labeling(self):
        """Test if single feature name is displayed correctly on the x-axis."""
        fig, df = boxplot(
            self.adata, 
            features=['feature1'], 
            orient='v',
            interactive=True,
        )
        x_labels = [x_data for boxplot in fig.data for x_data in boxplot.x]
        self.assertEqual(x_labels, ['feature1'])

        fig, df = boxplot(
            self.adata, 
            features=['feature1'], 
            orient='h',
            interactive=True,
        )
        y_labels = [y_data for boxplot in fig.data for y_data in boxplot.y]
        self.assertEqual(y_labels, ['feature1'])


if __name__ == '__main__':
    unittest.main()
