# spac/__init__.py

import os
import glob


# Define the subdirectory where you want to find Python modules
subdirectory = "src"

# Get a list of all Python module files in the package directory
module_files = glob.glob(os.path.join(os.path.dirname(__file__), subdirectory, "*.py"))

# Exclude the "__init__.py" file itself from the list
module_files = [f for f in module_files if not f.endswith("__init__.py")]

# Dynamically import all functions from all modules
functions = []
for module_file in module_files:
    module_name = os.path.basename(module_file)[:-3]  # Remove the ".py" extension
    module = __import__(f"{__name__}.{module_name}", fromlist=["*"])
    module_functions = [name for name in dir(module) if callable(getattr(module, name))]
    functions.extend(module_functions)

# Define the package version before using it in __all__
__version__ = "0.7.4"

# Define a __all__ list to specify which functions should be considered public
__all__ = functions
