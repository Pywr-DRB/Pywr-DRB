"""
Lists of results_set options and descriptions.

Overview: 
The results_set specifications are important for the output processing.
The main reason we use results_set is so that we can use the node names for the data, 
e.g., we want to have access to both the following:
    - inflows (pd.DataFrame) with node names as columns
    - res_storage (pd.DataFrame) with node names as columns
This helps to avoid the full parameter names when processing data, since the 
full parameter names are long and harder to interpret. 

Technical Notes: 
- results_set are used as arguments for all data loaders, including:
        - Observation
        - HydrologicModelFlow
        - Output
- To understand more about how these relate to the actual model parameters, look at source code for pywrdrb/load/get_results.py

Links: 
- A description of these options is on the docs: https://pywr-drb.github.io/Pywr-DRB/results_set_options.html
 
Change Log:
TJA, 2025-05-05, Add docs.
"""


# All results_set options and descriptions
pywrdrb_results_set_descriptions = {
    "all": "All simulation data using pywrdrb model naming convention.",
    "reservoir_downstream_gage": "Streamflow at downstream gage below reservoirs (MGD).",
    "res_storage": "Reservoir storage volume (MG).",
    "major_flow": "Streamflow at major flow points of interest (MGD).",
    "res_release": "Reservoir releases (MGD).",
    "downstream_release_target": "Downstream release targets at Montague & Trenton (MGD).",
    "inflow": "Inflow at each node or reservoir (MGD).",
    "catchment_withdrawal": "Withdrawal at each catchment (MGD).",
    "catchment_consumption": "Consumption at each catchment (MGD).",
    "prev_flow_catchmentWithdrawal": "Previous flow at catchment withdrawal nodes (MGD).",
    "max_flow_catchmentWithdrawal": "Maximum flow at catchment withdrawal nodes (MGD).",
    "max_flow_catchmentConsumption": "Maximum flow at catchment consumption nodes (MGD).",
    "res_level": "FFMP drought level.",
    "ffmp_level_boundaries": "FFMP level boundaries.",
    "mrf_target": "MRF targets at Montague & Trenton (MGD).",
    "nyc_release_components": "NYC reservoir releases, by release type (MGD).",
    "lower_basin_mrf_contributions": "Lower basin reservoir contributions to minimum flow targets (MGD).",
    "ibt_demands": "Diversion demands for NYC and NJ (MGD).",
    "ibt_diversions": "Diversion deliveries for NYC and NJ (MGD).",
    "mrf_targets": "streamflow targets for Montague and Trenton (MGD).",
    "all_mrf": "All MRF data.",
    "temperature": "Temperature data.",
}


# Different results_set options which are available for different datasets
pywrdrb_results_set_opts = list(pywrdrb_results_set_descriptions.keys())
obs_results_set_opts = ['major_flow', 'reservoir_downstream_gage', 'res_storage']
hydrologic_model_results_set_opts = ['major_flow', 'reservoir_downstream_gage']