"""Configuration for Setup Analysis tool"""

TOOL_CONFIG = {
    'outputs': {
        'analysis': 'analysis_output.pickle'
    },
    'list_params': ['Features_to_Analyze'],
    'column_params': ['X_Coordinate_Column', 'Y_Coordinate_Column', 
                      'Features_to_Analyze', 'Annotation_s_'],
    'input_key': 'Upstream_Dataset'
}