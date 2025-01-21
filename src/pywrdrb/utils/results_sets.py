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


pywrdrb_results_set_opts = list(pywrdrb_results_set_descriptions.keys())
obs_results_set_opts = ['major_flow', 'reservoir_downstream_gage', 'res_storage']
hydrologic_model_results_set_opts = ['major_flow', 'reservoir_downstream_gage']


base_results_set_descriptions = {
    "reservoir_downstream_gage": "Streamflow at downstream gage below reservoirs (MGD).",
    "major_flow": "Streamflow at major flow points of interest (MGD).",
    "res_storage": "Reservoir storage volume (MG).",
    "gage_flow": "Streamflow at gage locations (MGD).",
    "catchment_inflow": "Catchment inflow at Pywr-DRB node (MGD).",
}

base_results_set_opts = list(base_results_set_descriptions.keys())

