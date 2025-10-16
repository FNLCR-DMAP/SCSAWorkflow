#!/usr/bin/env python3
"""
Visualization Benchmark Script
===============================

A command-line tool for benchmarking visualization functions from the SPAC
package. This script measures the performance of various plotting functions
with different dataset sizes and configurations.

Usage:
    # Benchmark with a custom dataset
    python visualization_benchmark.py --dataset path/to/data.pkl --visualizations boxplot histogram

    # Benchmark with generated random data
    python visualization_benchmark.py --min-datapoints 1000 --max-datapoints 10000 \
        --increment 1000 --visualizations both --output-plot

Author: SPAC Development Team
Date: October 2025
"""

import argparse
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Import SPAC visualization functions
from spac.visualization import boxplot, boxplot_interactive


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the benchmark script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments containing benchmark configuration.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark visualization functions from the SPAC package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark with custom dataset
  %(prog)s --dataset data.pkl --visualizations boxplot histogram --output-plot

  # Benchmark with a single dataset size
  %(prog)s --size 5000 --visualizations both --output-plot

  # Benchmark with generated data range
  %(prog)s --min-datapoints 1000 --max-datapoints 5000 --increment 1000 \\
           --visualizations both --output-plot

  # Run only histogram benchmarks with single size
  %(prog)s --size 2000 --visualizations histogram
        """
    )

    # Dataset input options
    data_group = parser.add_argument_group("Dataset Options")
    data_group.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to input dataset file (pickle or anndata-readable format). "
             "If not provided, random datasets will be generated."
    )

    # Random data generation options
    random_group = parser.add_argument_group(
        "Random Data Generation Options",
        "Used when --dataset is not provided"
    )
    random_group.add_argument(
        "--size",
        type=int,
        default=None,
        metavar="N",
        help="Single dataset size to benchmark. If specified, overrides "
             "min/max/increment options."
    )
    random_group.add_argument(
        "--min-datapoints",
        type=int,
        default=1000,
        metavar="N",
        help="Minimum number of datapoints for generated datasets (default: 1000). "
             "Ignored if --size is specified."
    )
    random_group.add_argument(
        "--max-datapoints",
        type=int,
        default=10000,
        metavar="N",
        help="Maximum number of datapoints for generated datasets (default: 10000). "
             "Ignored if --size is specified."
    )
    random_group.add_argument(
        "--increment",
        type=int,
        default=1000,
        metavar="N",
        help="Increment step for datapoint sizes (default: 1000). "
             "Ignored if --size is specified."
    )

    # Visualization options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--visualizations",
        type=str,
        nargs="+",
        choices=["boxplot", "histogram", "both"],
        default=["both"],
        metavar="TYPE",
        help="Which visualizations to benchmark: boxplot, histogram, or both "
             "(default: both)"
    )

    # Plot parameter options
    plot_group = parser.add_argument_group(
        "Plot Parameters",
        "Optional parameters for visualization functions. If not specified, "
        "sensible defaults will be auto-discovered from the dataset."
    )
    plot_group.add_argument(
        "--annotation",
        type=str,
        default=None,
        metavar="NAME",
        help="Annotation column to use for grouping (e.g., cell_type). "
             "If not provided, the first categorical annotation will be used."
    )
    plot_group.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME",
        help="Feature(s) to plot. Provide one or more feature names separated by spaces. "
             "If not provided, the first 3 features will be used."
    )
    plot_group.add_argument(
        "--layer",
        type=str,
        default=None,
        metavar="NAME",
        help="Layer to use for plotting (e.g., normalized, log_transformed). "
             "If not provided, the main data matrix (adata.X) will be used."
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-plot",
        action="store_true",
        help="Generate and save a plot comparing benchmark results"
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        metavar="DIR",
        help="Directory to save benchmark results (default: benchmark_results)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate arguments
    _validate_arguments(args)

    return args


def _validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate the parsed command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments to validate.

    Raises
    ------
    ValueError
        If any argument validation fails.
    SystemExit
        If validation fails, exits with error message.
    """
    # Validate dataset path if provided
    if args.dataset is not None:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"Error: Dataset file not found: {args.dataset}", file=sys.stderr)
            sys.exit(1)
        if not dataset_path.is_file():
            print(f"Error: Dataset path is not a file: {args.dataset}", file=sys.stderr)
            sys.exit(1)

    # Validate random data generation parameters
    if args.dataset is None:
        if args.size is not None:
            # Single size mode
            if args.size <= 0:
                print("Error: --size must be positive", file=sys.stderr)
                sys.exit(1)
        else:
            # Range mode
            if args.min_datapoints <= 0:
                print("Error: --min-datapoints must be positive", file=sys.stderr)
                sys.exit(1)
            if args.max_datapoints <= 0:
                print("Error: --max-datapoints must be positive", file=sys.stderr)
                sys.exit(1)
            if args.min_datapoints > args.max_datapoints:
                print("Error: --min-datapoints cannot exceed --max-datapoints", 
                      file=sys.stderr)
                sys.exit(1)
            if args.increment <= 0:
                print("Error: --increment must be positive", file=sys.stderr)
                sys.exit(1)

    # Normalize "both" in visualizations list
    if "both" in args.visualizations:
        args.visualizations = ["boxplot", "histogram"]
    else:
        # Remove duplicates while preserving order
        args.visualizations = list(dict.fromkeys(args.visualizations))


def _discover_plot_parameters(adata: ad.AnnData, args: argparse.Namespace) -> dict:
    """
    Auto-discover sensible default parameters for plotting from the dataset.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to discover parameters from.
    args : argparse.Namespace
        Command-line arguments that may contain user-specified overrides.

    Returns
    -------
    dict
        Dictionary containing discovered/selected parameters:
        - annotation: str or None
        - features: list of str
        - layer: str or None
    """
    params = {}
    
    # Discover or use specified annotation
    if args.annotation is not None:
        # Validate user-specified annotation
        if args.annotation not in adata.obs.columns:
            print(f"Error: Annotation '{args.annotation}' not found in dataset.", 
                  file=sys.stderr)
            print(f"Available annotations: {', '.join(adata.obs.columns.tolist())}", 
                  file=sys.stderr)
            sys.exit(1)
        params['annotation'] = args.annotation
        print(f"  Using specified annotation: {args.annotation}")
    else:
        # Auto-discover: use first categorical column
        categorical_cols = [col for col in adata.obs.columns 
                          if pd.api.types.is_categorical_dtype(adata.obs[col]) or
                             adata.obs[col].dtype == 'object']
        if categorical_cols:
            params['annotation'] = categorical_cols[0]
            print(f"  Auto-discovered annotation: {params['annotation']}")
            print(f"    (Available: {', '.join(categorical_cols)})")
        else:
            params['annotation'] = None
            print("  No categorical annotations found, will use None")
    
    # Discover or use specified features
    if args.features is not None:
        # Validate user-specified features
        invalid_features = [f for f in args.features if f not in adata.var_names]
        if invalid_features:
            print(f"Error: Features not found in dataset: {', '.join(invalid_features)}", 
                  file=sys.stderr)
            print(f"Available features: {', '.join(adata.var_names[:10].tolist())}"
                  f"{'...' if len(adata.var_names) > 10 else ''}", 
                  file=sys.stderr)
            sys.exit(1)
        params['features'] = args.features
        print(f"  Using specified features: {', '.join(args.features)}")
    else:
        # Auto-discover: use first 3 features (or all if fewer than 3)
        n_features = min(3, len(adata.var_names))
        params['features'] = adata.var_names[:n_features].tolist()
        print(f"  Auto-discovered features: {', '.join(params['features'])}")
        if len(adata.var_names) > 3:
            print(f"    (Total available: {len(adata.var_names)})")
    
    # Discover or use specified layer
    if args.layer is not None:
        # Validate user-specified layer
        if args.layer not in adata.layers.keys():
            print(f"Error: Layer '{args.layer}' not found in dataset.", 
                  file=sys.stderr)
            available = ', '.join(adata.layers.keys()) if adata.layers.keys() else 'None'
            print(f"Available layers: {available}", file=sys.stderr)
            sys.exit(1)
        params['layer'] = args.layer
        print(f"  Using specified layer: {args.layer}")
    else:
        # Auto-discover: use first layer if available, otherwise None (main matrix)
        if len(adata.layers.keys()) > 0:
            params['layer'] = list(adata.layers.keys())[0]
            print(f"  Auto-discovered layer: {params['layer']}")
            print(f"    (Available: {', '.join(adata.layers.keys())})")
        else:
            params['layer'] = None
            print("  No layers found, will use main matrix (adata.X)")
    
    return params


def load_dataset(dataset_path: str) -> ad.AnnData:
    """
    Load a dataset from file, supporting both pickle and anndata formats.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file. Supports pickle files (.pkl, .pickle) 
        and any format readable by anndata (e.g., .h5ad, .zarr, .loom).

    Returns
    -------
    ad.AnnData
        Loaded AnnData object containing the dataset.

    Raises
    ------
    ValueError
        If the file format is not supported or loading fails.
    SystemExit
        If the loaded object is not an AnnData instance.
    """
    file_path = Path(dataset_path)
    file_suffix = file_path.suffix.lower()

    print(f"Loading dataset from: {dataset_path}")

    try:
        # Try loading as pickle file first for .pkl and .pickle extensions
        if file_suffix in ['.pkl', '.pickle']:
            print("  Detected pickle format, loading with pickle...")
            with open(file_path, 'rb') as f:
                adata = pickle.load(f)
            
            # Ensure the loaded object is an AnnData instance
            if not isinstance(adata, ad.AnnData):
                print(f"Error: Pickle file does not contain an AnnData object. "
                      f"Found type: {type(adata).__name__}", file=sys.stderr)
                sys.exit(1)
        
        else:
            # Try loading with anndata for other formats
            print(f"  Attempting to load with anndata...")
            adata = ad.read(dataset_path)

        # Validate that we have a proper AnnData object
        if not isinstance(adata, ad.AnnData):
            print(f"Error: Loaded object is not an AnnData instance. "
                  f"Found type: {type(adata).__name__}", file=sys.stderr)
            sys.exit(1)

        # Print basic dataset information
        print(f"  Successfully loaded dataset:")
        print(f"    Observations (cells): {adata.n_obs}")
        print(f"    Variables (features): {adata.n_vars}")
        if adata.obs.columns.size > 0:
            print(f"    Annotations: {', '.join(adata.obs.columns[:5].tolist())}"
                  f"{'...' if len(adata.obs.columns) > 5 else ''}")
        print()

        return adata

    except FileNotFoundError:
        print(f"Error: File not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    except pickle.UnpicklingError as e:
        print(f"Error: Failed to unpickle file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}", file=sys.stderr)
        print(f"  Supported formats: pickle (.pkl, .pickle) or anndata-readable "
              f"formats (.h5ad, .zarr, .loom, etc.)", file=sys.stderr)
        sys.exit(1)


def generate_random_dataset(n_obs: int, random_state: int = 42) -> ad.AnnData:
    """
    Generate a random AnnData object with realistic clustering for benchmarking.

    Creates a synthetic dataset with 5 features, 5 annotations, and 3 data layers.
    Uses sklearn's make_blobs to generate data with natural clustering patterns
    rather than pure noise.

    Parameters
    ----------
    n_obs : int
        Number of observations (cells/datapoints) to generate.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    ad.AnnData
        Generated AnnData object with the following structure:
        - X: Main feature matrix (n_obs × 5 features)
        - obs: DataFrame with 5 categorical annotations
        - var: Feature names (marker_1 through marker_5)
        - layers: 3 transformed versions of the data
          (normalized, log_transformed, scaled)
    """
    np.random.seed(random_state)
    
    # Generate base data with natural clustering using make_blobs
    # Create 5 features with 5 cluster centers for realistic structure
    n_features = 5
    n_centers = 5
    
    X, cluster_labels = make_blobs(
        n_samples=n_obs,
        n_features=n_features,
        centers=n_centers,
        cluster_std=1.5,
        random_state=random_state
    )
    
    # Make values positive (common in biological data) and add some variation
    X = np.abs(X) + np.random.exponential(scale=2.0, size=X.shape)
    
    # Create feature names
    feature_names = [f"marker_{i+1}" for i in range(n_features)]
    
    # Create 5 annotations with varying numbers of unique values
    # Assign based on clusters and add some randomness for realism
    
    # Annotation 1: cell_type (5 categories)
    cell_types = [f"Type_{chr(65+i)}" for i in range(5)]  # Type_A, Type_B, etc.
    cell_type = np.array([cell_types[i % 5] for i in cluster_labels])
    
    # Annotation 2: phenotype (4 categories) - partially correlated with clusters
    phenotypes = [f"Pheno_{i+1}" for i in range(4)]
    phenotype = np.array([phenotypes[i % 4] for i in cluster_labels])
    # Add some randomness to ~20% of assignments
    random_mask = np.random.random(n_obs) < 0.2
    phenotype[random_mask] = np.random.choice(phenotypes, size=random_mask.sum())
    
    # Annotation 3: region (3 categories) - more random
    regions = ["Region_X", "Region_Y", "Region_Z"]
    region = np.random.choice(regions, size=n_obs)
    
    # Annotation 4: batch (3 categories) - random assignment
    batches = ["Batch_1", "Batch_2", "Batch_3"]
    batch = np.random.choice(batches, size=n_obs)
    
    # Annotation 5: treatment (2 categories) - binary, roughly balanced
    treatments = ["Control", "Treated"]
    treatment = np.random.choice(treatments, size=n_obs, p=[0.5, 0.5])
    
    # Create observations DataFrame
    obs = pd.DataFrame({
        'cell_type': pd.Categorical(cell_type),
        'phenotype': pd.Categorical(phenotype),
        'region': pd.Categorical(region),
        'batch': pd.Categorical(batch),
        'treatment': pd.Categorical(treatment)
    })
    
    # Create the base AnnData object
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = feature_names
    
    # Create 3 layers with different transformations
    
    # Layer 1: Normalized (min-max normalization per feature)
    X_normalized = np.zeros_like(X)
    for i in range(n_features):
        feature_min = X[:, i].min()
        feature_max = X[:, i].max()
        X_normalized[:, i] = (X[:, i] - feature_min) / (feature_max - feature_min)
    adata.layers['normalized'] = X_normalized
    
    # Layer 2: Log-transformed (log1p to handle zeros)
    adata.layers['log_transformed'] = np.log1p(X)
    
    # Layer 3: Scaled (standardized with mean=0, std=1)
    scaler = StandardScaler()
    adata.layers['scaled'] = scaler.fit_transform(X)
    
    return adata


def benchmark_boxplot(
    adata: ad.AnnData,
    annotation: str,
    features: List[str],
    layer: str = None
) -> Tuple[float, dict]:
    """
    Benchmark the standard boxplot function.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to plot.
    annotation : str
        Annotation column for grouping.
    features : List[str]
        List of features to plot.
    layer : str, optional
        Layer to use for plotting.

    Returns
    -------
    tuple
        (execution_time, result_dict) where result_dict contains the plot outputs.
    """
    # Close any existing plots to avoid memory issues
    plt.close('all')
    
    start_time = time.time()
    
    try:
        # Call boxplot function
        fig, ax, df = boxplot(
            adata,
            annotation=annotation,
            features=features,
            layer=layer
        )
        
        execution_time = time.time() - start_time
        
        # Clean up to free memory
        plt.close(fig)
        
        return execution_time, {'fig': fig, 'ax': ax, 'df': df}
    
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"    Error in boxplot: {e}")
        return execution_time, {'error': str(e)}


def benchmark_boxplot_interactive(
    adata: ad.AnnData,
    annotation: str,
    features: List[str],
    layer: str = None
) -> Tuple[float, dict]:
    """
    Benchmark the interactive boxplot function.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object to plot.
    annotation : str
        Annotation column for grouping.
    features : List[str]
        List of features to plot.
    layer : str, optional
        Layer to use for plotting.

    Returns
    -------
    tuple
        (execution_time, result_dict) where result_dict contains the plot outputs.
    """
    # Close any existing plots to avoid memory issues
    plt.close('all')
    
    start_time = time.time()
    
    try:
        # Call boxplot_interactive function with showfliers='downsample'
        result = boxplot_interactive(
            adata,
            annotation=annotation,
            features=features,
            layer=layer,
            showfliers='downsample'
        )
        
        execution_time = time.time() - start_time
        
        return execution_time, result
    
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"    Error in boxplot_interactive: {e}")
        return execution_time, {'error': str(e)}


def run_boxplot_benchmarks(
    datasets: List[ad.AnnData],
    plot_params: dict
) -> pd.DataFrame:
    """
    Run boxplot benchmarks on all provided datasets.

    Parameters
    ----------
    datasets : List[ad.AnnData]
        List of datasets to benchmark on.
    plot_params : dict
        Dictionary containing plot parameters (annotation, features, layer).

    Returns
    -------
    pd.DataFrame
        DataFrame containing benchmark results with columns:
        - n_obs: number of observations
        - boxplot_time: execution time for boxplot
        - boxplot_interactive_time: execution time for boxplot_interactive
        - speedup_factor: ratio of times (boxplot / boxplot_interactive)
    """
    print("Running boxplot benchmarks...")
    print("=" * 70)
    
    results = []
    
    for i, adata in enumerate(datasets, 1):
        n_obs = adata.n_obs
        print(f"\nDataset {i}/{len(datasets)}: {n_obs} observations")
        
        # Benchmark standard boxplot
        print("  Testing boxplot...")
        boxplot_time, boxplot_result = benchmark_boxplot(
            adata,
            annotation=plot_params['annotation'],
            features=plot_params['features'],
            layer=plot_params['layer']
        )
        print(f"    Time: {boxplot_time:.4f} seconds")
        
        # Benchmark interactive boxplot
        print("  Testing boxplot_interactive...")
        interactive_time, interactive_result = benchmark_boxplot_interactive(
            adata,
            annotation=plot_params['annotation'],
            features=plot_params['features'],
            layer=plot_params['layer']
        )
        print(f"    Time: {interactive_time:.4f} seconds")
        
        # Calculate speedup factor
        if interactive_time > 0:
            speedup = boxplot_time / interactive_time
            print(f"    Speedup factor: {speedup:.2f}x "
                  f"({'boxplot_interactive' if speedup > 1 else 'boxplot'} is faster)")
        else:
            speedup = None
        
        results.append({
            'n_obs': n_obs,
            'boxplot_time': boxplot_time,
            'boxplot_interactive_time': interactive_time,
            'speedup_factor': speedup
        })
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    
    return pd.DataFrame(results)


def main():
    """
    Main entry point for the benchmark script.
    
    Orchestrates the benchmarking process including argument parsing,
    data loading/generation, running benchmarks, and outputting results.
    """
    # Parse command-line arguments
    args = parse_arguments()

    print("=" * 70)
    print("SPAC Visualization Benchmark")
    print("=" * 70)
    print()

    # Display configuration
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset if args.dataset else 'Generated'}")
    if args.dataset is None:
        if args.size is not None:
            print(f"  Dataset size: {args.size}")
        else:
            print(f"  Datapoint range: {args.min_datapoints} - {args.max_datapoints}")
            print(f"  Increment: {args.increment}")
    print(f"  Visualizations: {', '.join(args.visualizations)}")
    print(f"  Output plot: {args.output_plot}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Step 1: Load or generate datasets
    if args.dataset is not None:
        # Load user-provided dataset
        adata = load_dataset(args.dataset)
        datasets = [adata]
    else:
        # Generate random datasets for benchmarking
        print("Generating random datasets for benchmarking...")
        datasets = []
        
        if args.size is not None:
            # Single size mode
            dataset_sizes = [args.size]
        else:
            # Range mode
            dataset_sizes = range(
                args.min_datapoints,
                args.max_datapoints + 1,
                args.increment
            )
        
        for size in dataset_sizes:
            print(f"  Generating dataset with {size} observations...")
            adata = generate_random_dataset(n_obs=size, random_state=42)
            datasets.append(adata)
        
        print(f"  Successfully generated {len(datasets)} dataset(s)")
        if args.size is not None:
            print(f"  Size: {args.size}")
        else:
            print(f"  Size range: {args.min_datapoints} to {args.max_datapoints}")
        print()

    # Step 2: Auto-discover plot parameters from the first dataset
    print("Discovering plot parameters...")
    plot_params = _discover_plot_parameters(datasets[0], args)
    print()

    print(f"Plot parameters to be used:")
    print(f"  Annotation: {plot_params['annotation']}")
    print(f"  Features: {', '.join(plot_params['features'])}")
    print(f"  Layer: {plot_params['layer']}")
    print()

    # Step 3: Run benchmarks based on selected visualizations
    all_results = {}
    
    if 'boxplot' in args.visualizations:
        # Run boxplot vs boxplot_interactive benchmark
        boxplot_results = run_boxplot_benchmarks(datasets, plot_params)
        all_results['boxplot'] = boxplot_results
        
        # Display summary
        print("\nBoxplot Benchmark Summary:")
        print(boxplot_results.to_string(index=False))
        print()
    
    # TODO: Add histogram benchmarks when requested
    
    # Step 4: Save results to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for viz_type, results_df in all_results.items():
        output_file = output_dir / f"{viz_type}_benchmark_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    # Step 5: Generate comparison plots if requested
    if args.output_plot and len(all_results) > 0:
        print("\nGenerating comparison plots...")
        # TODO: Implement plot generation
        print("  Plot generation not yet implemented")
    
    print("\n" + "=" * 70)
    print("Benchmark execution complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
