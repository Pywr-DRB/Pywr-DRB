"""
Organize data records into appropriate format for Pywr-DRB simulations.
Generate any additional input data (e.g., predicted inflows/diversions).
"""
 
from utils.directories import input_dir
from pre.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from pre.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions
from pre.predict_inflows_diversions import predict_inflows_diversions
from pre.predict_inflows_diversions import predict_ensemble_inflows_diversions
from pre.prep_input_data_functions import *


if __name__ == "__main__":

    ### read in observed data
    df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', 
                           '1945/01/01', '2022/12/31',
                           units = 'cms', source = 'USGS')

    ### now get NYC & NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    # download_USGS_data_NYC_NJ_diversions()    ### dont need to rerun this every time
    extrapolate_NYC_NJ_diversions('nyc')
    extrapolate_NYC_NJ_diversions('nj')

    ### Create predicted future Montague & Trenton inflows & NJ diversions 
    # based on lagged regressions, for scheduling NYC releases    
    predict_inflows_diversions('obs_pub_nhmv10_ObsScaled', '1945/01/01', '2022/12/31')
    predict_inflows_diversions('obs_pub_nwmv21_ObsScaled', '1945/01/01', '2022/12/31')
    
    predict_ensemble_inflows_diversions('obs_pub_nhmv10_ObsScaled_ensemble', '1945/01/01', '2022/12/31')
    predict_ensemble_inflows_diversions('obs_pub_nwmv21_ObsScaled_ensemble', '1945/01/01', '2022/12/31')

    ## get catchment demands based on DRBC data
    sw_demand = disaggregate_DRBC_demands()