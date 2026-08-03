import warnings

from pywrdrb.pre.flows import *
try:
    from pywrdrb.pre.obs_data_retrieval import *
except (ImportError, AttributeError) as e:
    # obs_data_retrieval requires geopandas → shapely compiled against NumPy 1.x.
    # Fails under NumPy 2.x with AttributeError: _ARRAY_API not found.
    # USGS data-retrieval utilities are not needed for simulation or optimization.
    warnings.warn(
        f"pywrdrb.pre.obs_data_retrieval could not be imported and its "
        f"utilities are unavailable: {e}"
    )
from pywrdrb.pre.predict_diversions import *
from pywrdrb.pre.predict_inflows import *
from pywrdrb.pre.extrapolate_nyc_nj_diversions import *
from pywrdrb.pre.flood_node_inflows import *
from pywrdrb.pre.generate_presimulated_releases import (
    STARFITOfflineSimulator,
    STARFITReleaseEnsemblePreprocessor,
)
