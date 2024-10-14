
import pandas as pd
import numpy as np
import anndata
import unittest
from spac.spatial_analysis import ripley_l
from spac.visualization import plot_ripley_l
import matplotlib.pyplot as plt


class TestPlotRipleyL(unittest.TestCase):

    def create_dummy_dataset(self, dataframe=None):
        # Create a mock AnnData object for testing

        # Add four points to form a rectangle that is 1 * 2
        if dataframe is None:
            dictionary = [
                {'phenotype': 'A', 'feature': 10, 'spatial_x': 0,  'spatial_y': 0},
                {'phenotype': 'A', 'feature': 20, 'spatial_x': 1,  'spatial_y': 0},
                {'phenotype': 'A', 'feature': 20, 'spatial_x': 0,  'spatial_y': 2},
                {'phenotype': 'A', 'feature': 20, 'spatial_x': 1,  'spatial_y': 2},
            ]

            dataframe = pd.DataFrame(dictionary)

        # Extract the feature column as numpy array
        features = dataframe['feature'].to_numpy().reshape(-1, 1)

        # Convert features to float32
        features = features.astype(np.float32)

        # Extract the spatial coordinates as numpy array
        spatial_coords = dataframe[['spatial_x', 'spatial_y']].to_numpy()

        # Create anndata object
        adata = anndata.AnnData(X=features)

        adata.obs["phenotype"] = pd.Categorical(dataframe["phenotype"])
        adata.obsm["spatial"] = spatial_coords
        return adata

    def setUp(self):
        # Create a mock AnnData object for testing
        self.adata = self.create_dummy_dataset()

    def test_simulations(self):
        """
        Test turning on an off simulations
        """

        phenotype_name = ["A"]
        n_cells = 200
        phenotypes = phenotype_name * n_cells
        features = np.random.rand(n_cells)
        # Generate spatial_x at random float position in the square
        x_max = 200
        y_max = 200
        spatial_x = np.random.rand(n_cells) * x_max
        spatial_y = np.random.rand(n_cells) * y_max

        # Keep radius relatively small to avoid boundary adjustment
        radii = [0, 1, 2,  3, 4, 5]

        # Create a dataframe out of phenotypes, features, spatial coordinates
        dictionary = {'phenotype': phenotypes, 'feature': features,
                      'spatial_x': spatial_x, 'spatial_y': spatial_y}
        dataframe = pd.DataFrame(dictionary)

        self.adata = self.create_dummy_dataset(dataframe)

        ripley_l(
            self.adata,
            annotation="phenotype",
            phenotypes=["A", "A"],
            distances=radii,
            n_simulations=10,
        )

        # Test simulaitons is off
        fig = plot_ripley_l(
            self.adata,
            phenotypes=("A", "A"),
            sims=False
        )

        # Check that the legend has two values
        legends = fig.get_axes()[0].get_legend().get_texts()
        self.assertEqual(len(legends), 1)

        # Test simulations is on
        fig = plot_ripley_l(
            self.adata,
            phenotypes=("A", "A"),
            sims=True
        )

        # Check that the legend has two values
        legends = fig.get_axes()[0].get_legend().get_texts()
        self.assertEqual(len(legends), 2)

        # Check that the second legend is "simulations"
        self.assertEqual(
            legends[1].get_text(),
            "Simulations(all):10 runs"
        )
        # save the plot
        # fig.savefig("simulations.png")

    def test_regions(self):
        """
        Test plotting multiple regions
        """

        phenotype_name = ["A"]
        n_cells = 200
        phenotypes = phenotype_name * n_cells * 2
        features = np.random.rand(n_cells * 2)
        # Generate spatial_x at random float position in the square
        x_max = 200
        y_max = 200
        # Cells for region 'region1' are randomly scattered
        region1_spatial_x = np.random.rand(n_cells) * x_max
        region1_spatial_y = np.random.rand(n_cells) * y_max

        # Cells for region 'region2' are clustered
        region2_spatial_x = np.random.rand(n_cells) * x_max / 8
        region2_spatial_y = np.random.rand(n_cells) * y_max / 8

        # concatenate the spatial coordinates
        spatial_x = np.concatenate((region1_spatial_x, region2_spatial_x))
        spatial_y = np.concatenate((region1_spatial_y, region2_spatial_y))

        region = ['day1'] * n_cells + ['day2'] * n_cells

        # Keep radius relatively small to avoid boundary adjustment
        radii = [0, 1, 2,  3, 4, 5]

        # Create a dataframe out of phenotypes, features, spatial coordinates
        dictionary = {'phenotype': phenotypes, 'feature': features,
                      'spatial_x': spatial_x, 'spatial_y': spatial_y,
                      'day': region}
        dataframe = pd.DataFrame(dictionary)

        self.adata = self.create_dummy_dataset(dataframe)
        self.adata.obs["day"] = pd.Categorical(dataframe["day"])
        ripley_l(
            self.adata,
            annotation="phenotype",
            phenotypes=["A", "A"],
            distances=radii,
            n_simulations=100,
            regions="day"
        )

        # Test simulaitons is off
        fig = plot_ripley_l(
            self.adata,
            phenotypes=("A", "A"),
            regions=["day1"],
            sims=False
        )

        # Check that the legend has two values
        legends = fig.get_axes()[0].get_legend().get_texts()
        self.assertEqual(len(legends), 1)

        # Check that one of the legend texts includes "region1"
        # and the other includes "region2"
        legend_texts = [text.get_text() for text in legends]
        self.assertTrue(any("day1" in legend for legend in legend_texts))

    def test_two_phenotypes(self):
        """
        Test plotting one region with two phenotypes
        """

        phenotype_A = ["A"]
        phenotype_B = ["B"]
        n_cells_A = 200
        n_cells_B = 100
        n_cells = n_cells_A + n_cells_B
        phenotypes = phenotype_A * n_cells_A + phenotype_B * n_cells_B
        features = np.random.rand(n_cells)

        # Generate spatial_x at random float position in the square
        x_max = 200
        y_max = 200

        spatial_x = np.random.rand(n_cells) * x_max
        spatial_y = np.random.rand(n_cells) * y_max

        # Keep radius relatively small to avoid boundary adjustment
        radii = [0, 1, 2,  3, 4, 5]

        # Create a dataframe out of phenotypes, features, spatial coordinates
        dictionary = {'phenotype': phenotypes, 'feature': features,
                      'spatial_x': spatial_x, 'spatial_y': spatial_y}
        dataframe = pd.DataFrame(dictionary)

        self.adata = self.create_dummy_dataset(dataframe)
        ripley_l(
            self.adata,
            annotation="phenotype",
            phenotypes=["A", "B"],
            distances=radii,
            n_simulations=100
        )

        fig = plot_ripley_l(
            self.adata,
            phenotypes=("A", "B"),
            sims=True
        )

        # fig.savefig("two_phenotypes.png")

        # Test that one legend shows the correct number of A and B cells
        legends = fig.get_axes()[0].get_legend().get_texts()
        text = legends[0].get_text()
        self.assertTrue("(200, 100)" in text)

    def test_no_phenotypes(self):
        """
        Test Ripley does not has passed phenotypes
        """
        adata = self.create_dummy_dataset()
        distances = [5]
        phenotypes = ['A', 'B']

        ripley_l(
            adata=adata,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
        )

        expected_error_message = (
            'No Ripley L results found for the specified pair of phenotypes.\n'
            'Center Phenotype: "A"\n'
            'Neighbor Phenotype: "A"\n'
            'Exisiting unique pairs:   center_phenotype neighbor_phenotype\n'
            '0                A                  B'
        )

        # Check that calling plot_ripley_l raises the expected error with the exact message
        with self.assertRaisesRegex(ValueError, expected_error_message):
            plot_ripley_l(
                adata,
                phenotypes=("A", "A"),
                regions="region",
                sims=True
            )

    def test_warning_no_region_phenotypes(self):
        """
        Test Ripley does not has passed phenotypes
        """
        adata = self.create_dummy_dataset()
        distances = [5]
        phenotypes = ['A', 'C']

        ripley_l(
            adata=adata,
            annotation='phenotype',
            phenotypes=phenotypes,
            distances=distances,
        )

        expected_warning_message = (
            'WARNING, phenotype "C" not found in region "all",'
            ' skipping Ripley L.'
        )

        with self.assertLogs(level='WARNING') as log:
            # Call the function that triggers the warning
            plot_ripley_l(
                adata,
                phenotypes=("A", "C"),
                sims=True
            )

        # Check that the expected warning message is in the logs
        self.assertTrue(
            any(expected_warning_message in text for text in log.output))

    def test_no_ripley_l(self):
        """
        Test Ripley computation did not run
        """
        adata = self.create_dummy_dataset()

        expected_error_message = (
            'Ripley L results not found in the analsyis'
        )

        # Check that calling plot_ripley_l raises the expected error with the exact message
        with self.assertRaisesRegex(ValueError, expected_error_message):
            plot_ripley_l(
                adata,
                phenotypes=("A", "A"),
                regions="region",
                sims=True
            )


if __name__ == "__main__":
    unittest.main()
