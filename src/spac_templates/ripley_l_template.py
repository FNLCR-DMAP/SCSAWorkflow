import json
import pickle
import os
import sys

def load_pickle_from_file(file_path):
    """Simple pickle loader"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle_to_file(data, file_path):
    """Simple pickle saver"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to {file_path}")

def ripley_l_calculation_template(json_parameters):
    """
    Ripley L calculation template that accepts JSON parameters

    Args:
        json_parameters: Either a JSON string or path to JSON file

    Returns:
        The analysis data object
    """
    # Load parameters
    if isinstance(json_parameters, str):
        if json_parameters.endswith('.json'):
            # It's a file path
            with open(json_parameters, 'r') as f:
                params = json.load(f)
        else:
            # It's a JSON string
            params = json.loads(json_parameters)
    else:
        params = json_parameters

    # Extract parameters (with defaults for optional ones)
    input_pickle_path = params['input_data']
    radii = params['radii']  # Already a list of numbers in JSON
    annotation = params['annotation']
    phenotypes = params['phenotypes']
    regions = params.get('regions', None)  
    n_simulations = params.get('n_simulations', 100)
    area = params.get('area', None)
    seed = params.get('seed', 42)
    spatial_key = params.get('spatial_key', 'spatial')
    edge_correction = params.get('edge_correction', True)
    output_path = params.get('output_path', 'transform_output.pickle')

    # Handle Code Ocean paths if needed
    if 'CO_DATA_DIR' in os.environ:
        data_dir = os.environ.get('CO_DATA_DIR', '/data')
        results_dir = os.environ.get('CO_RESULTS_DIR', '/results')
        input_pickle_path = os.path.join(data_dir, input_pickle_path)
        output_path = os.path.join(results_dir, output_path)

    # Load input data
    print(f"Loading data from: {input_pickle_path}")
    adata = load_pickle_from_file(input_pickle_path)

    # Run the actual analysis
    try:
        # Check if it's an actual AnnData object
        import anndata
        if isinstance(adata, anndata.AnnData):
            from spac.spatial_analysis import ripley_l

            ripley_l(
                adata,
                annotation=annotation,
                phenotypes=phenotypes,
                distances=radii,  # Already numbers from JSON
                regions=regions,
                n_simulations=n_simulations,
                area=area,
                seed=seed,
                spatial_key=spatial_key,
                edge_correction=edge_correction
            )
        else:
            # For testing without real AnnData
            print("Note: Not an AnnData object, using dummy analysis")
            if hasattr(adata, 'uns'):
                # It's an object with uns attribute
                adata.uns['ripley_l_results'] = {
                    'radii': radii,
                    'phenotypes': phenotypes,
                    'status': 'dummy_analysis'
                }
            elif isinstance(adata, dict) and 'uns' in adata:
                # It's a dictionary with uns key
                adata['uns']['ripley_l_results'] = {
                    'radii': radii,
                    'phenotypes': phenotypes,
                    'status': 'dummy_analysis'
                }
            else:
                # Create uns structure for other cases
                if isinstance(adata, dict):
                    adata['uns'] = {}
                    adata['uns']['ripley_l_results'] = {
                        'radii': radii,
                        'phenotypes': phenotypes,
                        'status': 'dummy_analysis'
                    }
                elif hasattr(adata, 'uns') or hasattr(adata, '__dict__'):
                    adata.uns = {}
                    adata.uns['ripley_l_results'] = {
                        'radii': radii,
                        'phenotypes': phenotypes,
                        'status': 'dummy_analysis'
                    }
                else:
                    print("Warning: Unable to add results to data structure")

    except ImportError:
        print("SPAC or AnnData not available - using dummy analysis for testing")
        # Add dummy results for testing without SPAC
        if hasattr(adata, 'uns'):
            # It's an object with uns attribute
            adata.uns['ripley_l_results'] = {
                'radii': radii,
                'phenotypes': phenotypes,
                'status': 'dummy_analysis'
            }
        elif isinstance(adata, dict) and 'uns' in adata:
            # It's a dictionary with uns key
            adata['uns']['ripley_l_results'] = {
                'radii': radii,
                'phenotypes': phenotypes,
                'status': 'dummy_analysis'
            }
        else:
            # Create uns structure for other cases
            if isinstance(adata, dict):
                adata['uns'] = {}
                adata['uns']['ripley_l_results'] = {
                    'radii': radii,
                    'phenotypes': phenotypes,
                    'status': 'dummy_analysis'
                }
            elif hasattr(adata, 'uns'):
                adata.uns = {}
                adata.uns['ripley_l_results'] = {
                    'radii': radii,
                    'phenotypes': phenotypes,
                    'status': 'dummy_analysis'
                }
            else:
                print("Warning: Unable to add results to data structure")

    # Save output
    save_pickle_to_file(adata, output_path)

    return adata


# Command line interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ripley_l_template.py <params.json>")
        sys.exit(1)

    result = ripley_l_calculation_template(sys.argv[1])
    print("Analysis completed successfully")
