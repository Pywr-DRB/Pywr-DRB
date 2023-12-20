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
from pre.predict_inflows_diversions import predict_inflows_diversions
from pre.predict_inflows_diversions import predict_ensemble_inflows_diversions
from pre.prep_input_data_functions import *
from pre.reorganize_data import combine_nwmv21_datasets


if __name__ == "__main__":

    # Option to re-predict inflows/diversions across the ensemble inputs
    prepare_prediction_ensemble = False

    ### set random seed for consistency
    np.random.seed(1)

    ### read in observed, NHM, & NWM data
    ### use same set of dates for all.
    start_date = '1983/10/01'
    end_date = '2016/12/31'

    #
    # df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', '1945/01/01', '2022/12/31',
    #                        units = 'cms', source = 'USGS')
    #
    # df_nhm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nhmv10_mgd.csv', start_date, end_date,
    #                        units = 'mgd', source = 'nhm')
    #
    # df_nwm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nwmv21_mgd.csv', start_date, end_date,
    #                        units = 'mgd', source = 'nwmv21')
    #
    # #assert ((df_obs.index == df_nhm.index).mean() == 1) and ((df_nhm.index == df_nwm.index).mean() == 1)
    #
    # ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
    # match_gages(df_nhm, 'nhmv10', site_matches_id= nhm_site_matches)
    # match_gages(df_obs, 'obs', site_matches_id= obs_site_matches)
    # match_gages(df_nwm, 'nwmv21', site_matches_id= nwm_site_matches)
    #
    #
    #
    # ### process WEAP data to compatible format
    # prep_WEAP_data()
    #
    # ### create hybrid datasets
    # create_hybrid_modeled_observed_datasets('nhmv10', df_nhm.index)
    # create_hybrid_modeled_observed_datasets('nwmv21', df_nwm.index)

    ### now get NYC & NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    # download_USGS_data_NYC_NJ_diversions()    ### dont need to rerun this every time
    # extrapolate_NYC_NJ_diversions('nyc', make_figs=True)
    # extrapolate_NYC_NJ_diversions('nj', make_figs=True)

    # ### create predicted future Montague & Trenton inflows & NJ diversions based on lagged regressions, for scheduling NYC releases
    start_date_training = '1983/10/01'
    end_date_training = '2008/01/01'
    predict_inflows_diversions('nhmv10', start_date_training, end_date_training, make_figs=True)
    predict_inflows_diversions('nwmv21', start_date_training, end_date_training, make_figs=True)
    predict_inflows_diversions('nhmv10_withObsScaled', start_date_training, end_date_training, make_figs=True)
    predict_inflows_diversions('nwmv21_withObsScaled', start_date_training, end_date_training, make_figs=True)
    #
    # # # Predict for historic reconstructions
    # # predict_inflows_diversions('obs_pub_nhmv10_ObsScaled', '1945/01/01', '2022/12/31')
    # # predict_inflows_diversions('obs_pub_nwmv21_ObsScaled', '1945/01/01', '2022/12/31')
    # #
    # # if prepare_prediction_ensemble:
    # #     predict_ensemble_inflows_diversions('obs_pub_nhmv10_ObsScaled_ensemble', '1945/01/01', '2022/12/31')
    # #     predict_ensemble_inflows_diversions('obs_pub_nwmv21_ObsScaled_ensemble', '1945/01/01', '2022/12/31')
    #
    # ### get catchment demands based on DRBC data
    # sw_demand = disaggregate_DRBC_demands()