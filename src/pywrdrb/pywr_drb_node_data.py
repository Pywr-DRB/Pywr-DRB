"""
This script stores dictionaries which specify the connections between
Pywr-DRB nodes and data sources for different inflow datasets.

There are unique node source matches depending on whether the dataset is:
- Observed data only
- Reconstructed historic data with PUB
- NHMv10
- NWMv2.1
- WEAP
"""

## Set of all upstream nodes for every downstream node
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

### the time delay/lag (days) between each node and its downstream connection node
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
}


## NHM segment IDs
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


### match for WEAP results file (24Apr2023, gridmet, NatFlows) corresponding to each node in Pywr-DRB.
### results files should be f'{match}_GridMet_NatFlows.csv'. these results are in MM3/day.
WEAP_24Apr2023_gridmet_NatFlows_matches = {
    "cannonsville": ["RES_DelAbvCannon"],
    "pepacton": ["RES_DelAbvPepacton"],
    "neversink": ["RES_AbvNeversink"],
    "wallenpaupack": ["RES_Wallenpaupack"],
    "prompton": ["RES_Prompton"],
    "shoholaMarsh": ["RES_ShoholaMarsh"],
    "mongaupeCombined": ["RES_RioMongaupe"],
    "beltzvilleCombined": ["RES_Beltzville"],
    "fewalter": ["RES_FEWalter"],
    "merrillCreek": None,
    "hopatcong": ["RES_Hopatcong"],
    "nockamixon": ["RES_Nockamixon"],
    "assunpink": ["RES_Assunpink"],
    "ontelaunee": ["RES_Ontelaunee"],
    "stillCreek": None,
    "blueMarsh": ["RES_BlueMarsh_01470779"],
    "greenLane": ["RES_GreenLane"],
    "01425000": ["West Brnch Del BlwCannnon_01425000"],
    "01417000": ["DelawareBlwPepactonRes_01417000"],
    "delLordville": ["DelAtLordville_01427207"],
    "01436000": ["NeversinkBlwRes_01436000"],
    "01433500": ["Mongaup_014433500"],
    "delMontague": ["Delaware River at Montague_01438500"],
    "01449800": ["PohopocoBlwBeltzville_0149800"],
    "01447800": ["Leigh_01447800"],
    "delDRCanal": ["Delaware at Trenton_01463500"],
    "delTrenton": ["Delaware at Trenton_01463500"],
    "01463620": ["Assunpink_01463620"],
    "outletAssunpink": ["Assunpink_01464000"],
    "01470960": ["Tulpenhocken_01470960"],
    "outletSchuylkill": ["Schuykill_01474500"],
}

WEAP_29June2023_gridmet_NatFlows_matches = WEAP_24Apr2023_gridmet_NatFlows_matches

### WRF-Hydro site matches
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
