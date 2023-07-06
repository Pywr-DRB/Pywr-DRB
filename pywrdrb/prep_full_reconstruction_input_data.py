"""
Prepares model input data (inflows, demands, and NYC diversions) for the extended historic reconstruction (1955-2022).
"""

from utils.directories import input_dir
from pywr_drb_node_data import upstream_nodes_dict

import pandas as pd

from pywr_drb_node_data import obs_site_matches, obs_pub_site_matches, upstream_nodes_dict
from utils.directories import input_dir
from data_processing.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from data_processing.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions

from prep_input_data import read_csv_data, match_gages


## Specifications
start_date = '1950-01-01'
end_date = '2022-12-31'

# Reconstruction alternative specs
obs_pub_donor_fdc = 'nhmv10'            # Options: 'nwmv21', 'nhmv10'
regression_nhm_inflow_scaling = False   # If true, Cannonsville and Pep. inflows increase following NHM-based regression to estimate HRU inflows


if __name__ == "__main__":
    
    # Hist. Reconst. names are based on method specs
    hist_reconst_filename = f'historic_reconstruction_daily_{obs_pub_donor_fdc}'
    hist_reconst_filename = f'{hist_reconst_filename}_NYCscaled' if regression_nhm_inflow_scaling else hist_reconst_filename

    df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', start_date, end_date, units = 'cms', source = 'USGS')
    df_obs.index = pd.to_datetime(df_obs.index)
    
    df_obs_pub = pd.read_csv(f'{input_dir}modeled_gages\{hist_reconst_filename}_mgd.csv', 
                             sep=',', index_col=0, parse_dates=True).loc[start_date:end_date, :]
    df_obs_pub.index = pd.to_datetime(df_obs_pub.index)
    
    ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
    df_obs = match_gages(df_obs, 'obs', site_matches_id= obs_site_matches, upstream_nodes_dict= upstream_nodes_dict)
    df_obs_pub = match_gages(df_obs_pub, 'obs_pub', site_matches_id= obs_pub_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    ### now get NYC diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nyc_diversion = extrapolate_NYC_NJ_diversions('nyc')
    nyc_diversion.to_csv(f'{input_dir}deliveryNYC_ODRM_extrapolated.csv', index=False)

    ### now get NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nj_diversion = extrapolate_NYC_NJ_diversions('nj')
    nj_diversion.to_csv(f'{input_dir}deliveryNJ_WEAP_23Aug2022_gridmet_extrapolated.csv', index=False)

    ### get catchment demands based on DRBC data
    sw_demand = disaggregate_DRBC_demands()
    sw_demand.to_csv(f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv', index_label='node')
