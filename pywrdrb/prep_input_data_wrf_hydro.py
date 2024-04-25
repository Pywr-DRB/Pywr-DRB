"""
Organize data records into appropriate format for Pywr-DRB.

Observed records (USGS gages) & WRF-Hydro modeled estimates for multiple WRF-Hydro configurations.

"""
 
import numpy as np
import pandas as pd
import datetime

from pywr_drb_node_data import obs_site_matches, wrf_hydro_site_matches

from utils.directories import input_dir

from pre.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from pre.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions
from pre.predict_inflows_diversions import predict_inflows_diversions
from pre.prep_input_data_functions import match_gages


if __name__ == "__main__":

    ### set random seed for consistency
    np.random.seed(1)
    
    # WRF-Hydro model configuration to use 
    # must have corresponding CSV in input_dir/modeled_gages/ already (see Input-Data-Retrival repo)
    config = {
        'climate': '1960s',
        'calibration': 'calib',
        'landcover': 'nlcd2016',
    }

    # Dates depending on config climate:
    date_ranges = {
        '1960s': ('1959-10-01', '1969-12-31'),
        'aorc': ('1979-10-01', '2021-12-31'),
        '2050s': ('2051-10-01', '2061-12-31'),
    }
    
    start_data = date_ranges[config['climate']][0]
    end_date = date_ranges[config['climate']][1]
    
    dataset_label = f'wrf{config["climate"]}_{config["calibration"]}_{config["landcover"]}'
    source_fname = f'{input_dir}modeled_gages/streamflow_daily_{dataset_label}.csv'
    
    
    # load WRF-Hydro model outputs
    df_wrf_hydro = pd.read_csv(source_fname, index_col=0, parse_dates=True) 
    
    
    ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
    match_gages(df_wrf_hydro, dataset_label=dataset_label, site_matches_id= wrf_hydro_site_matches)
    
    ### create hybrid datasets (not implemented for WRF-Hydro, but should be possible)
    # create_hybrid_modeled_observed_datasets('nhmv10', df_nhm.index)

    # Get NYC & NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    # Don't need to run this every time; it is already saved in input_dir for 1952-2022
    # download_USGS_data_NYC_NJ_diversions()    
    # extrapolate_NYC_NJ_diversions('nyc', make_figs=True)
    # extrapolate_NYC_NJ_diversions('nj', make_figs=True)

    # ### create predicted future Montague & Trenton inflows & NJ diversions based on lagged regressions, for scheduling NYC releases
    np.random.seed(1)
    start_date_training = '1983/10/01'
    end_date_training = '2008/01/01'
    predict_inflows_diversions('nhmv10', start_date_training, end_date_training, make_figs=True)
    
    ### get catchment demands based on DRBC data
    # sw_demand = disaggregate_DRBC_demands() # only need to run this once