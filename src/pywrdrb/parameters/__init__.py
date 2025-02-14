# import sys
# import os
# sys.path.insert(0, os.path.abspath('./'))

from .ffmp import *
from .starfit import STARFITReservoirRelease
from .ensemble import FlowEnsemble, PredictionEnsemble
from .general import LaggedReservoirRelease
from .lower_basin_ffmp import *
from .banks import IERQRelease_step1

'''
import os
import importlib
import inspect

def register_all_parameters():
    # Get the directory path of this module
    parameters_dir = os.path.dirname(__file__)
    module_names = [
        file[:-3]
        for file in os.listdir(parameters_dir)
        if file.endswith(".py") and file != "__init__.py"
    ]

    for module_name in module_names:
        # Import the module dynamically
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Iterate over attributes in the module
        for name, obj in inspect.getmembers(module):
            # Check if the attribute is a class and has a 'register' method
            if inspect.isclass(obj) and hasattr(obj, "register") and callable(obj.register):
                obj.register()

# Automatically call the register function when the package is imported
register_all_parameters()
'''