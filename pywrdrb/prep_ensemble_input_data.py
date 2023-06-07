"""
Prepares inflow data for a large ensemble of synthetic timeseries. 

Loops through the prep_input_data process for each realization.

"""

import pandas as pd

from utils.directories import input_dir
from pywr_drb_node_data import upstream_nodes_dict


if __name__ == "__main__":

    
    ### read in observed, NHM, & NWM data at gages downstream of reservoirs
    ### use same set of dates for all.
    N_SCENARIOS = 10
    inflow_type = 'obs_pub'

    start_date = '2001-01-01'
    end_date = '2031-12-23'
    
    columns = [f'scenario_{i}' for i in range(N_SCENARIOS)]
    scenario_flow = pd.read_csv(f'{input_dir}synthetic_ensembles/{inflow_type}/gage_flow_{inflow_type}_scenario_0.csv', sep = ',', index_col=0, parse_dates=True)
    
    ensemble_gage_flows = {}
    ensemble_inflows = {}
    for node in scenario_flow.columns:
        ensemble_gage_flows[node] = {}
        ensemble_inflows[node] = {}
    
    for i in range(N_SCENARIOS):

        scenario_flow = pd.read_csv(f'{input_dir}synthetic_ensembles/{inflow_type}/gage_flow_{inflow_type}_scenario_{i}.csv', sep = ',', index_col=0, parse_dates=True)
        scenario_flow = scenario_flow.loc[start_date:end_date, :]
        
        for node in scenario_flow.columns:
            ensemble_gage_flows[node][f'scenario_{i}'] = scenario_flow[node].values
        
        for node, upstreams in upstream_nodes_dict.items():
            scenario_flow[node] -= scenario_flow.loc[:, upstreams].sum(axis=1)
            scenario_flow[node].loc[scenario_flow[node] < 0] = 0

        for node in scenario_flow.columns:
            ensemble_inflows[node][f'scenario_{i}'] = scenario_flow[node].values

    ## Export DFs containing all scenarios for each node
    for node in scenario_flow.columns:
        gage_flow_df = pd.DataFrame(ensemble_gage_flows[node], index=pd.to_datetime(scenario_flow.index))
        inflow_df = pd.DataFrame(ensemble_inflows[node], index=pd.to_datetime(scenario_flow.index))
        
        gage_flow_df.to_csv(f'{input_dir}synthetic_ensembles/{inflow_type}/ensemble_gage_flow_{node}.csv', sep = ',')
        inflow_df.to_csv(f'{input_dir}synthetic_ensembles/{inflow_type}/ensemble_inflow_{node}.csv', sep = ',')