"""Configuration for Load CSV Files tool"""

def preprocess_load_csv(params):
    """Special preprocessing for Load CSV Files"""
    import os
    
    # Handle CSV_Files path
    if 'CSV_Files' in params:
        if isinstance(params['CSV_Files'], list) and len(params['CSV_Files']) == 1:
            params['CSV_Files'] = params['CSV_Files'][0]
        
        # If single file, get directory
        if isinstance(params['CSV_Files'], str) and os.path.isfile(params['CSV_Files']):
            params['CSV_Files'] = os.path.dirname(params['CSV_Files'])
    
    return params

TOOL_CONFIG = {
    'outputs': {
        'DataFrames': 'dataframe_folder'
    },
    'list_params': ['String_Columns'],
    'column_params': [],
    'preprocess': preprocess_load_csv
}