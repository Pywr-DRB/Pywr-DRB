"""
Contains simple lists that are used throughout the package for different purposes.

Overview: 
The lists serve different purposes, and are used in many contexts.  

Technical Notes: 
- This can probably be improved and/or removed in the future:
    - i.e., it may be added to the Options class or as a sepearate data class
    - but we should think about what we want before making changes.
- For now, DO NOT DELETE since these are important.
- We should also look for redundant versions of these lists elsewhere.
    
Links: 
- NA
 
Change Log:
TJA, 2025-05-05, Add docs.
"""

reservoir_list = [
    "cannonsville",
    "pepacton",
    "neversink",
    "wallenpaupack",
    "prompton",
    "shoholaMarsh",
    "mongaupeCombined",
    "beltzvilleCombined",
    "fewalter",
    "merrillCreek",
    "hopatcong",
    "nockamixon",
    "assunpink",
    "ontelaunee",
    "stillCreek",
    "blueMarsh",
    "greenLane",
]

reservoir_list_nyc = reservoir_list[:3]

majorflow_list = [
    "delLordville",
    "delMontague",
    "delDRCanal",
    "delTrenton",
    "outletAssunpink",
    "outletSchuylkill",
    "01425000",
    "01417000",
    "01436000",
    "01433500",
    "01449800",
    "01447800",
    "01463620",
    "01470960",
]
majorflow_list_figs = ["delMontague", "delTrenton", "outletSchuylkill"]

# The USGS gage data available downstream of reservoirs
reservoir_link_pairs = {
    "cannonsville": "01425000",
    "pepacton": "01417000",
    "neversink": "01436000",
    "mongaupeCombined": "01433500",
    "beltzvilleCombined": "01449800",
    "fewalter": "01447800",
    "assunpink": "01463620",
    "blueMarsh": "01470960",
}

starfit_reservoir_list = [
    "wallenpaupack",
    "prompton",
    "shoholaMarsh",
    "mongaupeCombined",
    "beltzvilleCombined",
    "fewalter",
    "merrillCreek",
    "hopatcong",
    "nockamixon",
    "assunpink",
    "ontelaunee",
    "stillCreek",
    "blueMarsh",
    "greenLane",
]


modified_starfit_reservoir_list = ["blueMarsh", "beltzvilleCombined", "fewalter"]

seasons_dict = {
    m: "DJF"
    if m in (12, 1, 2)
    else "MAM"
    if m in (3, 4, 5)
    else "JJA"
    if m in (6, 7, 8)
    else "SON"
    for m in range(1, 13)
}


drbc_lower_basin_reservoirs = [
    "beltzvilleCombined",
    "blueMarsh",
    "nockamixon",
]  # 'wallenpaupack' at comission request; not implemented
