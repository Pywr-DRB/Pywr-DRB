"""
This file control the interface of the pywrdrb module.

Overview:
This module is designed to customize the interface of how we want the users to access
the pywrdrb functionalities.

Change Log:
Chung-Yi Lin, 2025-05-02, None
TJA, 2025-11-26, Added logging configuration to disable Pywr debug logging for performance.
"""
from importlib.metadata import version as _version, PackageNotFoundError as _PackageNotFoundError

try:
    __version__ = _version("pywrdrb")
except _PackageNotFoundError:
    __version__ = "unknown"

# Configure logging BEFORE importing pywr to suppress debug messages
# This reduces overhead from millions of debug() calls during simulation
import logging
import os

# Set pywr logging level to WARNING unless explicitly overridden
# Users can set PYWR_LOG_LEVEL=DEBUG to enable verbose logging if needed
_pywr_log_level = os.environ.get('PYWR_LOG_LEVEL', 'WARNING').upper()
_pywr_logger = logging.getLogger('pywr')
_pywr_logger.setLevel(getattr(logging, _pywr_log_level, logging.WARNING))
# Prevent propagation to root logger and add null handler to minimize overhead
_pywr_logger.propagate = False
if not _pywr_logger.handlers:
    _pywr_logger.addHandler(logging.NullHandler())

# Import pywr modules to be accessed through pywrdrb
from pywr.model import Model

# pywr's compiled core emits ~one debug() call per node/parameter per timestep
# (profiling shows ~177k/short-run, virtually all from the 'pywr._model' logger;
# millions over a long/large-ensemble run). Even with the level set to WARNING,
# each call still pays Python call + isEnabledFor dispatch overhead — measurable
# seconds at HPC scale. No-op the debug method on pywr's loggers so those calls
# return immediately. WARNING/INFO/ERROR are untouched, and setting
# PYWR_LOG_LEVEL=DEBUG restores full debug logging.
if _pywr_log_level != 'DEBUG':
    def _pywr_noop_debug(*args, **kwargs):
        return None
    for _name in list(logging.root.manager.loggerDict):
        if _name == 'pywr' or _name.startswith('pywr.'):
            logging.getLogger(_name).debug = _pywr_noop_debug
    # Cover the dominant logger explicitly (created at the pywr import above).
    logging.getLogger('pywr._model').debug = _pywr_noop_debug

# Run path_manager first!!! 
from pywrdrb.path_manager import *
reset_pn() # Create a new global pathnavigator instance

from pywrdrb.recorder import *
from pywrdrb.model_builder import *
try:
    from pywrdrb import pre
except ImportError:
    # pre requires geopandas/dataretrieval; these are optional data-retrieval
    # dependencies that may not be available in all environments (e.g., NumPy 2.x
    # with a shapely compiled against NumPy 1.x).  Simulation and optimization
    # work without pre; only USGS data-download utilities are affected.
    # Narrowed to ImportError so genuine code errors in pre/ are not masked.
    pre = None
from pywrdrb.load.data_loader import Data

# All "parameters" need to be registered such that they can be accessed in pywr when loading a model file.
# We register the parameter classes right after they are defined. But some time it they are not correctly registered.
# In that case, we need to register them again here. To ensure that all parameters are registered whenever pywrdrb is imported.

# CL: I think the better way is to register them here. But the code work for now, so I will leave it as is.
# If we encounter issues with the parameters, then we do the structural change.
from pywrdrb.parameters.ffmp import *
VolBalanceNYCDemand.register()

# CL's temporary output parser (Maybe is a good idea to just keep it here)
import h5py
def hdf5_to_dict(file_path):
    """
    Convert an HDF5 file to a nested dictionary.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
        
    Returns
    -------
    dict
        A nested dictionary representation of the HDF5 file.
    """
    def recursive_dict(group):
        d = {}
        for key, item in group.items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                d[key] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                d[key] = recursive_dict(item)
        return d

    with h5py.File(file_path, 'r') as f:
        data_dict = recursive_dict(f)

    return data_dict