"""
Prepares inflow data for a large ensemble of inflow timeseries.
This can be used for either the historic reconstruction ensemble or synehtic ensemble. 

Loops through the prep_input_data process for each realization to calculate node inflows, 
exports HDF5 file containing all realizations (datasets) for each node (key).
"""

import pandas as pd
import sys

from utils.directories import input_dir
from utils.hdf5 import extract_realization_from_hdf5, get_hdf5_realization_numbers
from utils.hdf5 import export_ensemble_to_hdf5

from pywr_drb_node_data import obs_pub_site_matches
from prep_input_data import subtract_upstream_catchment_inflows


def main(inflow_type='obs_pub', scenario_type = 'historic_ensemble'):
    """Runs the prep_input_data for each realization in a steamflow ensemble. 
    Exports HDF5 file containing node (key) realization (columns) timeseries.

    Args:
        inflow_type (str): Model inflow type, Default is obs_pub.
        scenario_type (str): Options: historic_ensemble, synthetic_ensemble
    """

    if scenario_type == 'historic_ensemble':
        filename = f'historic_reconstruction_daily_{fdc_doner_type}_ensemble_mgd.hdf5'
    elif scenario_type == 'synthetic_ensemble':
        ## This is outdated and needs to be upped to hdf5
        filename = f'{inflow_type}/gage_flow_{inflow_type}_scenario_0.csv'

    # Intialize storage
    ensemble_gage_flows = {}
    ensemble_inflows = {}
    for node, sites in obs_pub_site_matches.items():
        ensemble_gage_flows[node] = {}
        ensemble_inflows[node] = {}


    # Open the HDF5 and get a list of the key IDs: 'realization_{ID}`
    realization_ids = get_hdf5_realization_numbers(f'{input_dir}/{scenario_type}s/{filename}')
    
    # Open one dataframe to retrieve datetime ranges
    check_inflow_df = extract_realization_from_hdf5(f'{input_dir}/{scenario_type}s/{filename}',
                                                    realization= 1)
    check_inflow_df.index = pd.to_datetime(check_inflow_df.index)
    
    start_date = check_inflow_df.index[0]
    end_date = check_inflow_df.index[-1]
    print(f'Preparing inflows for {scenario_type} between {start_date} and {end_date}.')
        
    for i in realization_ids:

        # Load the df containing flows
        realization_streamflow = extract_realization_from_hdf5(f'{input_dir}/{scenario_type}s/{filename}', realization=i)
        realization_streamflow.index = pd.to_datetime(realization_streamflow.index)        
        realization_streamflow = realization_streamflow.loc[start_date:end_date, :]
        
        ## Match gauges with PywrDRB nodes
        for node, sites in obs_pub_site_matches.items():
            if node == 'cannonsville':
                realization_nodeflow = pd.DataFrame(realization_streamflow.loc[:, sites].sum(axis=1))
                realization_nodeflow.columns = [node]
                realization_nodeflow.index = pd.to_datetime(realization_nodeflow.index)
            else:
                if sites == None:
                    realization_nodeflow[node] = realization_streamflow[node]
                else:
                    realization_nodeflow[node] = realization_streamflow.loc[:,sites].sum(axis=1)
                            
        # Subtract upstream flows to get just inflows to catchment
        realization_inflows = subtract_upstream_catchment_inflows(realization_nodeflow)

        ### Store data ###
        ## We want a df containing all realization timeseries for a single node        
        # Re-arrange by node-realization
        for node, sites in obs_pub_site_matches.items():
            ensemble_gage_flows[node][f'realization_{i}'] = realization_nodeflow[node].values
            ensemble_inflows[node][f'realization_{i}'] = realization_inflows[node].values

    ### EXPORTING ###
    df_ensemble_gage_flows = {}
    df_ensemble_inflows = {}
    # Convert to df with datetime index
    for node, sites in obs_pub_site_matches.items():
        df_ensemble_gage_flows[node] = pd.DataFrame(ensemble_gage_flows[node], columns=ensemble_gage_flows[node].keys(),
                                                    index=pd.to_datetime(check_inflow_df.index))
        df_ensemble_inflows[node] = pd.DataFrame(ensemble_inflows[node], columns=ensemble_inflows[node].keys(),
                                                 index=pd.to_datetime(check_inflow_df.index))

    print(f'Columns are:{df_ensemble_inflows["cannonsville"].columns}')
    # Export to hdf5
    export_ensemble_to_hdf5(df_ensemble_gage_flows, output_file=f'{input_dir}{scenario_type}s/gage_flow_ensemble.hdf5')
    export_ensemble_to_hdf5(df_ensemble_inflows, output_file=f'{input_dir}{scenario_type}s/catchment_inflow_ensemble.hdf5')
        
    return 



if __name__ == "__main__":

    # Specifications
    inflow_type = 'obs_pub'
    scenario_type = 'historic_ensemble'  # Or synthetic_ensemble    
    fdc_doner_type = sys.argv[1] #nhmv10 or nwmv21
    output_type = 'hdf5'

    # Run processing
    main(inflow_type= inflow_type, scenario_type=scenario_type)
