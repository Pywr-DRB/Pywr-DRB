"""
The init file for the pywrdrb.parameters module.

Overview: 
Import all parameter classes from the parameters submodules to flatten the structure. 
The pywrdrb.parameters module contains customized classes for pywr package. Users can 
view the paratemers as customized rules in the DRB system.
 
Change Log:
Chung-Yi Lin, 2025-05-02, None
"""
from pywrdrb.parameters.ffmp import *
from pywrdrb.parameters.starfit import STARFITReservoirRelease
from pywrdrb.parameters.ensemble import FlowEnsemble, PredictionEnsemble
from pywrdrb.parameters.general import LaggedReservoirRelease
from pywrdrb.parameters.lower_basin_ffmp import *
from pywrdrb.parameters.banks import IERQRelease_step1

# Old code to automatically register all parameters in the package.
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