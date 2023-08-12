"""
Organize data records into appropriate format for Pywr-DRB.

Observed records (USGS gages) & modeled estimates (NHM, NWM, WEAP).

"""
 
import numpy as np
import pandas as pd
import datetime

from pywr_drb_node_data import obs_site_matches, obs_pub_site_matches, nhm_site_matches, nwm_site_matches, \
                               upstream_nodes_dict, WEAP_29June2023_gridmet_NatFlows_matches, downstream_node_lags
from utils.constants import cfs_to_mgd, cms_to_mgd, cm_to_mg, mcm_to_mg
from utils.directories import input_dir, weap_dir
from utils.lists import reservoir_list_nyc
from pre.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from pre.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions, download_USGS_data_NYC_NJ_diversions
from pre.predict_Montague_Trenton_inflows import predict_Montague_Trenton_inflows
from pre.prep_input_data_functions import *



if __name__ == "__main__":
    
    ### read in observed, NHM, & NWM data
    ### use same set of dates for all.
    start_date = '1983/10/01'
    end_date = '2016/12/31'

    df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', start_date, end_date,
                           units = 'cms', source = 'USGS')

    df_nhm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nhmv10_mgd.csv', start_date, end_date,
                           units = 'mgd', source = 'nhm')

    df_nwm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nwmv21_mgd.csv', start_date, end_date,
                           units = 'mgd', source = 'nwmv21')

    assert ((df_obs.index == df_nhm.index).mean() == 1) and ((df_nhm.index == df_nwm.index).mean() == 1)

    ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
    match_gages(df_nhm, 'nhmv10', site_matches_id= nhm_site_matches)
    match_gages(df_obs, 'obs', site_matches_id= obs_site_matches)
    match_gages(df_nwm, 'nwmv21', site_matches_id= nwm_site_matches)



    ### process WEAP data to compatible format
    prep_WEAP_data()

    ### create hybrid datasets
    combine_modeled_observed_datasets('nhmv10', 'nhmv10', df_nhm.index)
    combine_modeled_observed_datasets('nhmv10', 'nwmv21', df_nhm.index)

    ### create predicted future Montague & Trenton inflows based on lagged regressions, for scheduling NYC releases
    start_date_training = '1983/10/01'
    end_date_training = '2008/01/01'
    predict_Montague_Trenton_inflows('nhmv10', start_date_training, end_date_training)
    predict_Montague_Trenton_inflows('nwmv21', start_date_training, end_date_training)
    predict_Montague_Trenton_inflows('nhmv10_withNYCObsScaled', start_date_training, end_date_training)
    predict_Montague_Trenton_inflows('nwmv21_withNYCObsScaled', start_date_training, end_date_training)

    ### now get NYC & NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    ### uses obs_pub_nhmv10_NYCScaling for inflow regressions & extrapolation -> this needs to be created first from Historic_reconstruction repo.
    # download_USGS_data_NYC_NJ_diversions()    ### dont need to rerun this every time
    extrapolate_NYC_NJ_diversions('nyc')
    extrapolate_NYC_NJ_diversions('nj')

    ### get catchment demands based on DRBC data
    sw_demand = disaggregate_DRBC_demands()



