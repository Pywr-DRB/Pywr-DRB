"""
Contains metadata for pywrdrb nodes, their inflow source IDs and other relationships. 

Overview:
This script stores many dictionaries that are used for different purposes, including:
- map pywrdrb node names to their inflow source IDs for different datasets (e.g., NWM, NHM, WEAP, etc.)
- define the upstream nodes for each pywrdrb node
- define the immediate downstream node for each pywrdrb node

Technical Notes:
- These relationships are important for pywrdrb funcitonality
- The "*_site_matches" dictionaries are used to map pywrdrb node names to their inflow source IDs for different datasets, including:
    - observatuional data (uses None when no data is available)
    - NHMv1.0
    - NWMv2.0
    - WRF-Hydro
- These IDs are unique to each source dataset. 
- These mappings are used during the preprocessing for different datasets

Links:
- NA

Change Log:
TJA, 2025-05-06, Add docstrings & delete old WEAP mapping since no longer supported.
"""

import os
from pywrdrb.path_manager import get_pn_object
pn = get_pn_object()

RAW_DATA_DIR = pn.observations.get_str() + os.sep + "_raw"
STORAGE_CURVE_DIR = pn.observations.get_str() + os.sep + "_raw" + os.sep + "_storage_curves" + os.sep

## Set of all upstream nodes for every downstream node
# {node: [all, upstream, nodes]}
upstream_nodes_dict = {
    "01425000": ["cannonsville"],
    "01417000": ["pepacton"],
    "delLordville": ["cannonsville", "pepacton", "01425000", "01417000"],
    "01436000": ["neversink"],
    "01433500": ["mongaupeCombined"],
    "delMontague": [
        "cannonsville",
        "pepacton",
        "01425000",
        "01417000",
        "delLordville",
        "prompton",
        "wallenpaupack",
        "shoholaMarsh",
        "mongaupeCombined",
        "neversink",
        "01436000",
        "01433500",
    ],
    "01449800": ["beltzvilleCombined"],
    "01447800": ["fewalter"],
    "delDRCanal": [
        "cannonsville",
        "pepacton",
        "01425000",
        "01417000",
        "delLordville",
        "prompton",
        "wallenpaupack",
        "shoholaMarsh",
        "mongaupeCombined",
        "neversink",
        "01436000",
        "01433500",
        "delMontague",
        "beltzvilleCombined",
        "fewalter",
        "merrillCreek",
        "hopatcong",
        "nockamixon",
        "01449800",
        "01447800",
    ],
    "delTrenton": [
        "cannonsville",
        "pepacton",
        "01425000",
        "01417000",
        "delLordville",
        "prompton",
        "wallenpaupack",
        "shoholaMarsh",
        "mongaupeCombined",
        "neversink",
        "01436000",
        "01433500",
        "delMontague",
        "beltzvilleCombined",
        "fewalter",
        "merrillCreek",
        "hopatcong",
        "nockamixon",
        "01449800",
        "01447800",
        "delDRCanal",
    ],
    "01463620": ["assunpink"],
    "outletAssunpink": ["assunpink", "01463620"],
    "01470960": ["blueMarsh"],
    "outletSchuylkill": [
        "ontelaunee",
        "stillCreek",
        "blueMarsh",
        "greenLane",
        "01470960",
    ],
}


# {node: [immediate_downstream_node]}
immediate_downstream_nodes_dict = {
    "cannonsville": "01425000",
    "pepacton": "01417000",
    "neversink": "01436000",
    "wallenpaupack": "delMontague",
    "prompton": "delMontague",
    "shoholaMarsh": "delMontague",
    "mongaupeCombined": "01433500",
    "beltzvilleCombined": "01449800",
    "fewalter": "01447800",
    "merrillCreek": "delDRCanal",
    "hopatcong": "delDRCanal",
    "nockamixon": "delDRCanal",
    "assunpink": "01463620",
    "ontelaunee": "outletSchuylkill",
    "stillCreek": "outletSchuylkill",
    "blueMarsh": "01470960",
    "greenLane": "outletSchuylkill",
    "01425000": "delLordville",
    "01417000": "delLordville",
    "delLordville": "delMontague",  # PUB until 2006 then '01427207'
    "01436000": "delMontague",
    "01433500": "delMontague",
    "delMontague": "delDRCanal",
    "01449800": "delDRCanal",
    "01447800": "delDRCanal",
    "delDRCanal": "delTrenton",  ### note DRCanal and Trenton are treated as being coincident, with DRCanal having the physical catchment inflows and withdrawals. DRCanal is where NJ deliveries leave from, and delTrenton is where min flow is enforced, so that this is downstream of deliveries.
    "delTrenton": "output_del",
    "01463620": "outletAssunpink",
    "outletAssunpink": "output_del",
    "01470960": "outletSchuylkill",
    "outletSchuylkill": "output_del",
}

### the time delay/lag (days) between each 
# node and its immediate downstream connection node
downstream_node_lags = {
    "cannonsville": 0,
    "pepacton": 0,
    "neversink": 0,
    "wallenpaupack": 1,
    "prompton": 1,
    "shoholaMarsh": 1,
    "mongaupeCombined": 0,
    "beltzvilleCombined": 0,
    "fewalter": 0,
    "merrillCreek": 1,
    "hopatcong": 1,
    "nockamixon": 1,
    "assunpink": 0,
    "ontelaunee": 2,
    "stillCreek": 2,
    "blueMarsh": 0,
    "greenLane": 1,
    "01425000": 0,
    "01417000": 0,
    "delLordville": 2,
    "01436000": 1,
    "01433500": 0,
    "delMontague": 2,
    "01449800": 2,
    "01447800": 2,
    "delDRCanal": 0,
    "delTrenton": 0,
    "01463620": 0,
    "outletAssunpink": 0,
    "01470960": 2,
    "outletSchuylkill": 0,
}

# Observed, FULL NATURAL FLOW, data source IDs 
# When None, no FULL NATURAL FLOW data is available 
# for that node. E.g., at Trenton, the data is managed, 
# so we say None for this location.
# This mapping is used to determine "PUB" locations for the reconstruction
obs_pub_site_matches = {
    "cannonsville": ["01423000", "0142400103"],  # 0142400103 doesnt start until '96
    "pepacton": [
        "01415000",
        "01414500",
        "01414000",
        "01413500",
    ],  # '01414000' doesnt start until 1996; likely underestimate before
    "neversink": ["01435000"],
    "wallenpaupack": None,
    "prompton": ["01428750"],  ## PUB till 1986-09-30 then '01428750' observation start
    "shoholaMarsh": None,
    "mongaupeCombined": None,
    "beltzvilleCombined": ["01449360"],  ## Not complete inflow
    "fewalter": ["01447720", "01447500"],
    "merrillCreek": None,
    "hopatcong": None,
    "nockamixon": None,
    "assunpink": None,
    "ontelaunee": None,
    "stillCreek": None,
    "blueMarsh": None,
    "greenLane": ["01472199", "01472198"],  # PUB until 1981
    "01425000": None,
    "01417000": None,
    "delLordville": None,  # ['01427207'], # PUB until 2006 then '01427207'
    "01436000": None,
    "01433500": None,
    "delMontague": None,  # ['01438500'],
    "01449800": None,
    "01447800": None,
    "delDRCanal": None,  # ['01463500'], ### note DRCanal and Trenton are treated as being coincident, with DRCanal having the physical catchment inflows and withdrawals. DRCanal is where NJ deliveries leave from, and delTrenton is where min flow is enforced, so that this is downstream of deliveries.
    "delTrenton": None,  # ['01463500'],
    "01463620": None,
    "outletAssunpink": None,  # ['01464000'],
    "01470960": None,
    "outletSchuylkill": None,  # ['01474500']
}

# Observed data USGS IDs
# in this dict, all flows are included (not necessarily full natural flow)
# used to generate data/observations/gage_flow_mgd.csv
obs_site_matches = {
    "cannonsville": ["01423000", "0142400103"],
    "pepacton": [
        "01415000",
        "01414500",
        "01414000",
        "01413500",
    ],  # '01414000' doesnt start until 1996; likely underestimate before
    "neversink": ["01435000"],
    "wallenpaupack": [],  ## PUB
    "prompton": [
        "01428750"
    ],  ## PUB :None till 1986-09-30 then '01428750' observation start
    "shoholaMarsh": [],  ## PUB
    "mongaupeCombined": [],  ## PUB
    "beltzvilleCombined": ["01449360"],  ## Not complete inflow
    "fewalter": ["01447720", "01447500"],
    "merrillCreek": [],  ## PUB
    "hopatcong": [],  ## PUB
    "nockamixon": [],  ## PUB
    "assunpink": [],  ## PUB
    "ontelaunee": [],  ## PUB
    "stillCreek": [],  ## PUB
    "blueMarsh": [],  ## PUB
    "greenLane": ["01472199", "01472198"],
    "01425000": ["01425000"],
    "01417000": ["01417000"],
    "delLordville": ["01427207"],
    "01436000": ["01436000"],
    "01433500": ["01433500"],
    "delMontague": ["01438500"],
    "01449800": ["01449800"],
    "01447800": ["01447800"],
    "delDRCanal": ["01463500"],
    "delTrenton": ["01463500"],
    "01463620": ["01463620"],  # This has periods of missing data
    "outletAssunpink": ["01464000"],
    "01470960": ["01470960"],
    "outletSchuylkill": ["01474500"],
    # Flood monitoring nodes
    "01426500": ["01426500"],  # Hale Eddy (West Branch Delaware)
    "01421000": ["01421000"],  # Fishs Eddy (East Branch Delaware)
    "01436690": ["01436690"],  # Bridgeville (Neversink River)
}

# NHM data IDs
# {node : [NHM segment IDs]}
nhm_site_matches = {
    "cannonsville": ["1562"],
    "pepacton": ["1449"],
    "neversink": ["1645"],
    "wallenpaupack": ["1602"],
    "prompton": ["1586"],
    "shoholaMarsh": [
        "1598"
    ],  ## Note, Shohola has no inlet gauge or NHM segment - use wallenpaupack upstream
    "mongaupeCombined": ["1640"],
    "beltzvilleCombined": ["1710"],
    "fewalter": ["1684", "1691", "1694"],  # Two tributary flows
    "merrillCreek": [
        "1467"
    ],  ## Merrill Creek doesnt have gage or HRU - using nearby small stream flow
    "hopatcong": ["1470"],  ## NOTE, this is downstream but should be good
    "nockamixon": ["1470"],
    "assunpink": [
        "1496"
    ],  ## NOTE, this is downstream of reservoir but above the link, should be good
    "ontelaunee": ["2279"],
    "stillCreek": ["2277"],  ## Note, this is downstream of reservoir and lakes
    "blueMarsh": ["2335"],
    "greenLane": [
        "2289",
        "2290",
    ],  # No suitable HRU or downstream point; use combined tributaries
    "01425000": ["1566"],
    "01417000": ["1444"],
    "delLordville": ["1573"],
    "01436000": ["1638"],
    "01433500": ["1647"],
    "delMontague": ["1659"],
    "01449800": ["1697"],
    "01447800": ["1695"],
    "delDRCanal": ["1498"],
    "delTrenton": ["1498"],
    "01463620": ["1492"],
    "outletAssunpink": ["1493"],
    "01470960": ["2333"],
    "outletSchuylkill": ["2338"],
}

## NWM data IDs are either COMIDs or USGS-IDs (USGS-IDs start with 0)
# {node : [NWM segment IDs]}
## NOTE: for NWM site matches -- 
# We use gauge IDs at actual gauge sites
# and COMIDs for non-gauge locations.
# This is an artifact of labels on the NWM data provided by Aubrey Dugger (NCAR)
nwm_site_matches = {
    "cannonsville": ["2613174"],  # Lake inflow
    "pepacton": ["1748473"],  # Lake inflow
    "neversink": ["4146742"],  # Lake inflow
    "wallenpaupack": ["2741600"],  # Lake inflow
    "prompton": ["2739068"],  # Lake inflow
    "shoholaMarsh": ["120052035"],  # Lake inflow
    "mongaupeCombined": ["4148582"],  # Lake inflow
    "beltzvilleCombined": ["4186689"],  # Lake inflow
    "fewalter": ["4185065"],  # Lake inflow
    "merrillCreek": ["2588031"],  # No NWM lake; using available segment flow
    "hopatcong": ["2585287"],  # Lake inflow
    "nockamixon": [
        "2591099"
    ],  # No NWM lake; using available segment flow  2591187 2591219
    "assunpink": ["2589015"],  # Lake inflow
    "ontelaunee": ["4779981"],  # Lake inflow
    "stillCreek": ["4778721"],  # Lake inflow
    "blueMarsh": ["4782813"],  # Lake inflow
    "greenLane": ["4780087"],  # Lake inflow
    "01425000": ["01425000"],
    "01417000": ["01417000"],
    "delLordville": ["2617364"],
    "01436000": ["01436000"],
    "01433500": ["01433500"],
    "delMontague": ["4151628"],
    "01449800": ["01449800"],
    "01447800": ["01447800"],
    "delDRCanal": ["2590277"],
    "delTrenton": ["2590277"],
    "01463620": ["01463620"],
    "outletAssunpink": ["2590137"],
    "01470960": ["01470960"],
    "outletSchuylkill": ["4784841"],
}

### WRF-Hydro site matches
# {node : [WRF-Hydro reach code]}
wrf_hydro_site_matches = {
    "cannonsville": ["2613174"],  # Lake inflow
    "pepacton": ["1748473"],  # Lake inflow
    "neversink": ["4146742"],  # Lake inflow
    "wallenpaupack": ["2741600"],  # Lake inflow
    "prompton": ["2739068"],  # Lake inflow
    "shoholaMarsh": ["120052035"],  # Lake inflow
    "mongaupeCombined": ["4148582"],  # Lake inflow
    "beltzvilleCombined": ["4186689"],  # Lake inflow
    "fewalter": ["4185065"],  # Lake inflow
    "merrillCreek": ["2588031"],  # No NWM lake; using available segment flow
    "hopatcong": ["2585287"],  # Lake inflow
    "nockamixon": [
        "2591099"
    ],  # No NWM lake; using available segment flow  2591187 2591219
    "assunpink": ["2589015"],  # Lake inflow
    "ontelaunee": ["4779981"],  # Lake inflow
    "stillCreek": ["4778721"],  # Lake inflow
    "blueMarsh": ["4782813"],  # Lake inflow
    "greenLane": ["4780087"],  # Lake inflow
    "01425000": ["2614238"],
    "01417000": ["1748727"],
    "delLordville": ["2617364"],
    "01436000": ["4147432"],
    "01433500": ["4150156"],
    "delMontague": ["4151628"],
    "01449800": ["4187341"],
    "01447800": ["4186403"],
    "delDRCanal": ["2590277"],
    "delTrenton": ["2590277"],
    "01463620": ["2590117"],
    "outletAssunpink": ["2590137"],
    "01470960": ["4783213"],
    "outletSchuylkill": ["4784841"],
}



### A single list containing all USGS IDs for all gauges
all_flow_gauges = []
# Start by adding gauges from the obs_site_matches dictionary
for node, gauges in obs_site_matches.items():
    if len(gauges) > 0:
        all_flow_gauges.extend(gauges)

# Add DR Canal gauge, which used for the diversion extrapolation
all_flow_gauges.extend(["01460440"])

# Supplemental gauges which are not model nodes, but are retrieved and kept
# as their own columns in gage_flow_mgd.csv for observation comparisons.
# 01429000: Lackawaxen River at Prompton (measures Prompton reservoir releases)
supplemental_flow_gauges = ["01429000"]
all_flow_gauges.extend(supplemental_flow_gauges)

# Remove duplicates
all_flow_gauges = list(set(all_flow_gauges))

storage_gauge_map = {
    "beltzvilleCombined": ["01449790"],
    "fewalter": ["01447780"],
    "prompton": ["01428900"],
    "blueMarsh": ["01470870"],
    "cannonsville": ["01423910"],
    "pepacton": ["01414750"],
    "neversink": ["01435900"],
}

storage_curves = {
    "01449790": f"{STORAGE_CURVE_DIR}beltzvilleCombined_storage_curve.csv", #"beltzvilleCombined"
    "01447780": f"{STORAGE_CURVE_DIR}fewalter_storage_curve.csv", #fewalter
    "01428900": f"{STORAGE_CURVE_DIR}prompton_storage_curve.csv", #prompton
    "01470870": f"{STORAGE_CURVE_DIR}blueMarsh_storage_curve.csv", #blueMarsh
    "01414750": f"{STORAGE_CURVE_DIR}pepacton_storage_curve.csv",       # Pepacton
    "01423910": f"{STORAGE_CURVE_DIR}cannonsville_storage_curve.csv",   # Cannonsville
    "01435900": f"{STORAGE_CURVE_DIR}neversink_storage_curve.csv"       # Neversink
}

nyc_reservoirs = ["pepacton", "cannonsville", "neversink"]


class TopologyDictionaries:
    """
    Static class for managing network topology dictionaries with optional flood monitoring nodes.

    This class provides efficient access to topology dictionaries with built-in caching.
    When multiple ModelBuilder instances are created in parallel (e.g., ensemble simulations),
    the topology dictionaries are computed once and reused, avoiding redundant copying and
    modification operations.

    Class Attributes
    ----------------
    _base_cache : dict or None
        Cached original topology dictionaries (without flood nodes)
    _flood_cache : dict or None
        Cached modified topology dictionaries (with flood nodes)

    Methods
    -------
    get(include_flood_nodes=False)
        Return topology dictionaries, optionally with flood monitoring nodes (uses caching)
    _build_base_dicts()
        Build original topology dictionaries (internal helper method)
    _build_modified_dicts_with_flood_nodes()
        Build modified topology with flood nodes inserted (internal helper method)

    Examples
    --------
    # Get original topology (cached after first call)
    >>> dicts = TopologyDictionaries.get(include_flood_nodes=False)
    >>> downstream = dicts[0]
    >>> downstream['01425000']
    'delLordville'

    # Get flood monitoring topology (cached after first call)
    >>> dicts = TopologyDictionaries.get(include_flood_nodes=True)
    >>> downstream = dicts[0]
    >>> downstream['01425000']
    '01426500'
    """

    # Class-level caches for topology dictionaries
    _base_cache = None
    _flood_cache = None

    @classmethod
    def _build_base_dicts(cls):
        """
        Build base topology dictionaries (original model without flood nodes).

        Returns
        -------
        tuple of dict
            Eight topology dictionaries with original routing.
        """
        return (
            immediate_downstream_nodes_dict.copy(),
            upstream_nodes_dict.copy(),
            downstream_node_lags.copy(),
            obs_site_matches.copy(),
            obs_pub_site_matches.copy(),
            nhm_site_matches.copy(),
            nwm_site_matches.copy(),
            wrf_hydro_site_matches.copy(),
        )

    @classmethod
    def _build_modified_dicts_with_flood_nodes(cls):
        """
        Build modified topology dictionaries with flood monitoring nodes inserted.

        This method creates copies of the original dictionaries and modifies them to insert
        three flood monitoring nodes into the network routing:
        - 01426500 (Hale Eddy) on West Branch Delaware River
        - 01421000 (Fishs Eddy) on East Branch Delaware River
        - 01436690 (Bridgeville) on Neversink River

        Routing changes:
        - West Branch: 01425000 → 01426500 → delLordville
        - East Branch: 01417000 → 01421000 → delLordville
        - Neversink: 01436000 → 01436690 → delMontague

        Returns
        -------
        tuple of dict
            Eight topology dictionaries with flood monitoring nodes.
        """
        # Create deep copies of original dictionaries
        modified_downstream = immediate_downstream_nodes_dict.copy()
        modified_upstream = {}
        for key, value in upstream_nodes_dict.items():
            modified_upstream[key] = value.copy() if isinstance(value, list) else value

        modified_lags = downstream_node_lags.copy()
        modified_obs = obs_site_matches.copy()
        modified_obs_pub = obs_pub_site_matches.copy()
        modified_nhm = nhm_site_matches.copy()
        modified_nwm = nwm_site_matches.copy()
        modified_wrf = wrf_hydro_site_matches.copy()

        # Modify downstream routing to insert flood monitoring nodes
        # West Branch: Stilesville → Hale Eddy → Lordville
        modified_downstream["01425000"] = "01426500"
        modified_downstream["01426500"] = "delLordville"

        # East Branch: Downsville → Fishs Eddy → Lordville
        modified_downstream["01417000"] = "01421000"
        modified_downstream["01421000"] = "delLordville"

        # Neversink: Release → Bridgeville → Montague
        modified_downstream["01436000"] = "01436690"
        modified_downstream["01436690"] = "delMontague"

        # Add upstream relationships for flood monitoring nodes
        modified_upstream["01426500"] = ["cannonsville", "01425000"]
        modified_upstream["01421000"] = ["pepacton", "01417000"]
        modified_upstream["01436690"] = ["neversink", "01436000"]

        # Update convergence nodes to include flood monitoring nodes in their upstream lists
        modified_upstream["delLordville"] = (
            upstream_nodes_dict["delLordville"] +
            ["01426500", "01421000"]
        )
        modified_upstream["delMontague"] = (
            upstream_nodes_dict["delMontague"] +
            ["01436690"]
        )
        modified_upstream["delDRCanal"] = (
            upstream_nodes_dict["delDRCanal"] +
            ["01426500", "01421000", "01436690"]
        )
        modified_upstream["delTrenton"] = (
            upstream_nodes_dict["delTrenton"] +
            ["01426500", "01421000", "01436690"]
        )

        # Add zero-day travel time lags for flood monitoring segments
        # (short distances between nodes, fast routing)
        modified_lags["01426500"] = 0  # Hale Eddy → Lordville
        modified_lags["01421000"] = 0  # Fishs Eddy → Lordville
        modified_lags["01436690"] = 0  # Bridgeville → Montague

        # Add USGS observation site matches for flood monitoring nodes
        modified_obs["01426500"] = ["01426500"]
        modified_obs["01421000"] = ["01421000"]
        modified_obs["01436690"] = ["01436690"]

        modified_obs_pub["01426500"] = ["01426500"]
        modified_obs_pub["01421000"] = ["01421000"]
        modified_obs_pub["01436690"] = ["01436690"]

        # NHM/NWM/WRF-Hydro segment matches left empty for now
        # To be added when segment mapping is completed:
        # modified_nhm["01426500"] = [segment_ids]
        # modified_nwm["01426500"] = [segment_ids]
        # modified_wrf["01426500"] = [segment_ids]

        return (
            modified_downstream,
            modified_upstream,
            modified_lags,
            modified_obs,
            modified_obs_pub,
            modified_nhm,
            modified_nwm,
            modified_wrf,
        )

    @classmethod
    def get(cls, include_flood_nodes=False):
        """
        Return topology dictionaries, optionally modified to include flood monitoring nodes.

        Uses class-level caching for efficiency when multiple ModelBuilder instances
        are created (e.g., parallel ensemble simulations). Dictionaries are computed
        once and reused on subsequent calls.

        Parameters
        ----------
        include_flood_nodes : bool, optional
            If True, insert flood monitoring nodes (01426500, 01421000, 01436690) into
            network topology by modifying routing between existing nodes and downstream
            convergence points. If False, return original topology without flood nodes.
            Default is False (original model).

        Returns
        -------
        tuple of dict
            Eight dictionaries in this order:
            - immediate_downstream_nodes_dict : Routing from each node to next downstream node
            - upstream_nodes_dict : All upstream contributors for each node
            - downstream_node_lags : Travel time (days) from node to downstream
            - obs_site_matches : USGS gage IDs for observed data (managed + natural)
            - obs_pub_site_matches : USGS gage IDs for full natural flow data only
            - nhm_site_matches : NHMv1.0 segment IDs
            - nwm_site_matches : NWMv2.1 segment IDs
            - wrf_hydro_site_matches : WRF-Hydro segment IDs

        Notes
        -----
        Flood monitoring nodes, with the local drainage area each one adds
        (USGS drainage areas; see pywrdrb.pre.flood_node_inflows.DRAINAGE_AREAS):
        - 01426500 (Hale Eddy): West Branch Delaware River below Stilesville, +139 sq mi
        - 01421000 (Fishs Eddy): East Branch Delaware River below Downsville, +412 sq mi
          (includes the large unregulated Beaver Kill)
        - 01436690 (Bridgeville): Neversink River below the reservoir release, +78.4 sq mi

        When flood nodes are included, routing is modified as follows:
        - West Branch: 01425000 → 01426500 → delLordville (was: 01425000 → delLordville)
        - East Branch: 01417000 → 01421000 → delLordville (was: 01417000 → delLordville)
        - Neversink: 01436000 → 01436690 → delMontague (was: 01436000 → delMontague)

        All flood segments use zero-day travel time lags (short distances, fast routing).

        Performance
        -----------
        - First call: Dictionaries are computed and cached (O(n) copy operations)
        - Subsequent calls: Cached dictionaries are returned (O(1) lookup)
        - Thread-safe for parallel model building
        """
        # Return cached version if available
        if not include_flood_nodes:
            if cls._base_cache is not None:
                return cls._base_cache
            # Build and cache base dictionaries
            cls._base_cache = cls._build_base_dicts()
            return cls._base_cache
        else:
            if cls._flood_cache is not None:
                return cls._flood_cache
            # Build and cache modified dictionaries with flood nodes
            cls._flood_cache = cls._build_modified_dicts_with_flood_nodes()
            return cls._flood_cache

