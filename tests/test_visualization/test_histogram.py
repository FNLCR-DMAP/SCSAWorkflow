import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
from spac.visualization import histogram
mpl.use('Agg')  # Set the backend to 'Agg' to suppress plot window


class TestRefactoredHistogram(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        X = np.random.rand(100, 3)
        annotation_values = ['A', 'B']
        annotation_types = ['cell_type_1', 'cell_type_2']
        cell_range = [f'cell_{i}' for i in range(1, 101)]
        annotation = pd.DataFrame({
            'annotation1': np.random.choice(annotation_values, size=100),
            'annotation2': np.random.choice(annotation_types, size=100),
        }, index=cell_range)
        var = pd.DataFrame(index=['marker1', 'marker2', 'marker3'])
        self.adata = anndata.AnnData(
            X.astype(np.float32), obs=annotation, var=var
        )

    def test_both_feature_and_annotation(self):
        err_msg = ("Cannot pass both feature_name and "
                   "annotation_name, choose one")
        with self.assertRaisesRegex(ValueError, err_msg):
            histogram(
                self.adata,
                feature_name='marker1',
                annotation_name='annotation1'
            )

    def test_histogram_feature_name(self):
        fig, ax = histogram(self.adata, feature_name='marker1')
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertIsInstance(ax[0], mpl.axes.Axes)

    def test_histogram_annotation_name(self):
        fig, ax = histogram(self.adata, annotation_name='annotation1')
        total_annotation = len(self.adata.obs['annotation1'])
        self.assertEqual(sum(p.get_height() for p in ax[0].patches),
                         total_annotation)

    def test_histogram_feature_group_by(self):
        fig, axs = histogram(
            self.adata,
            feature_name='marker1',
            group_by='annotation2',
            together=False
        )
        self.assertEqual(len(axs), 2)
        self.assertIsInstance(axs[0], mpl.axes.Axes)
        self.assertIsInstance(axs[1], mpl.axes.Axes)

    def test_log_scale(self):
        fig, ax = histogram(self.adata, feature_name='marker1', log_scale=True)
        self.assertTrue(ax[0].get_xscale() == 'log')

    def test_overlay_options(self):
        fig, ax = histogram(
            self.adata,
            feature_name='marker1',
            group_by='annotation2',
            together=True,
            multiple="layer",
            element="step"
        )
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertIsInstance(ax[0], mpl.axes.Axes)

    def test_layer(self):
        # Add a layer to the adata
        self.adata.layers['layer1'] = np.random.rand(100, 3)
        fig, ax = histogram(self.adata, feature_name='marker1', layer='layer1')
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertIsInstance(ax[0], mpl.axes.Axes)

    def test_ax_passed_as_argument(self):
        fig, ax = plt.subplots()
        returned_fig, returned_ax = histogram(
            self.adata,
            feature_name='marker1',
            ax=ax
        )
        self.assertIs(ax, returned_ax[0])
        self.assertIs(fig, returned_fig)


if __name__ == '__main__':
    unittest.main()
