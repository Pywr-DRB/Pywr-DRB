"""
Organize data records into appropriate format for Pywr-DRB.

Observed records (USGS gages) & modeled estimates (NHM, NWM, WEAP).

"""
 
import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm

from pywr_drb_node_data import obs_site_matches, obs_pub_site_matches, nhm_site_matches, nwm_site_matches, upstream_nodes_dict
from utils.constants import cfs_to_mgd, cms_to_mgd, cm_to_mg
from utils.directories import input_dir

from data_processing.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from data_processing.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions

# Date range
start_date = '1983/10/01'
end_date = '2016/12/31'

# Directories
weap_dir = input_dir + 'WEAP_23Aug2022_gridmet/'

nhm_inflow_scaling = False
nhm_inflow_scaling_coefs = {'cannonsville': 1.188,
                            'pepacton': 1.737}

def read_modeled_estimates(filename, sep, date_label, site_label, streamflow_label, start_date, end_date):
    '''Reads input streamflows from modeled NHM/NWM estimates, preps for Pywr.
    Returns dataframe.'''

    ### read in data & filter dates
    df = pd.read_csv(filename, sep = sep, dtype = {'site_no': str})
    df.sort_values([site_label, date_label], inplace=True)
    df.index = pd.to_datetime(df[date_label])
    df = df.loc[np.logical_and(df.index >= start_date, df.index <= end_date)]

    ### restructure to have gages as columns
    sites = list(set(df[site_label]))
    ndays = len(set(df[date_label]))
    df_gages = df.iloc[:ndays,:].loc[:, [site_label]]
    for site in sites:
        df_gages[site] = df.loc[df[site_label] == site, streamflow_label]
    df_gages.drop(site_label, axis=1, inplace=True)

    ### convert cms to mgd
    df_gages *= cms_to_mgd

    return df_gages


def read_csv_data(filename, start_date, end_date, units = 'cms', source = 'USGS'):
    """Reads in a pd.DataFrame containing USGS gauge data relevant to the model.
    """
    df = pd.read_csv(filename, sep = ',', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Remove USGS- from column names
    if source == 'USGS':
        df.columns = [i.split('-')[1] for i in df.columns] 
    
    df = df.loc[np.logical_and(df.index >= start_date, df.index <= end_date)]
    if units == 'cms':
        df *= cms_to_mgd
    return df


def match_gages(df, dataset_label, site_matches_id, upstream_nodes_dict):
    '''Matches USGS gage sites to nodes in Pywr-DRB.
    For reservoirs, the matched gages are actually downstream, but assume this flows into reservoir from upstream catchment.
    For river nodes, upstream reservoir inflows are subtracted from the flow at river node USGS gage.
    For nodes related to USGS gages downstream of reservoirs, currently redundant flow with assumed inflow, so subtracted additional catchment flow will be 0 until this is updated.
    Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.'''

    ### 1. Match inflows for each Pywr-DRB node 
    ## 1.1 Reservoir inflows
    for node, site in site_matches_id.items():
        if node == 'cannonsville':
            if (dataset_label == 'obs_pub') and (site == None):
                inflow = pd.DataFrame(df.loc[:, node])
            else:
                inflow = pd.DataFrame(df.loc[:, site].sum(axis=1))
            inflow.columns = [node]
            inflow['datetime'] = inflow.index
            inflow.index = inflow['datetime']
            inflow = inflow.iloc[:, :-1]
        else:
            if (dataset_label == 'obs_pub') and (site == None):
                inflow[node] = df[node]
            else:
                inflow[node] = df[site].sum(axis=1)
                
        if (dataset_label == 'obs_pub') and (nhm_inflow_scaling) and (node in nhm_inflow_scaling_coefs.keys()):
            print(f'Scaling {node} inflow using NHM ratio')
            inflow[node] = inflow[node]*nhm_inflow_scaling_coefs[node]
            
                 
    ## Save full flows to csv 
    # For downstream nodes, this represents the full flow for results comparison
    inflow.to_csv(f'{input_dir}gage_flow_{dataset_label}.csv')

    ### 2. Subtract flows into upstream nodes from mainstem nodes
    # This represents only the catchment inflows
    for node, upstreams in upstream_nodes_dict.items():
        inflow[node] -= inflow.loc[:, upstreams].sum(axis=1)
        inflow[node].loc[inflow[node] < 0] = 0

    ## Save catchment inflows to csv  
    # For downstream nodes, this represents the catchment inflow with upstream node inflows subtracted
    inflow.to_csv(f'{input_dir}catchment_inflow_{dataset_label}.csv')
    return inflow




def get_WEAP_df(filename, datecolumn):
    df = pd.read_csv(filename, header=3)
    df['datetime'] = df[datecolumn]
    df.index = pd.DatetimeIndex(df['datetime'])
    ### fill in leap days, assuming average of 2/28 and 3/1
    df = df.resample('D').sum()
    idxs = [i for i in range(df.shape[0]) if df.index[i].day == 29 and df.index[i].month == 2]
    for i in idxs:
        df.iloc[i, :] = (df.iloc[i - 1, :] + df.iloc[i + 1, :]) / 2
    return df




if __name__ == "__main__":
    
    # Defaults
    obs_pub_donor_fdc = 'nhmv10'

    ### read in observed, NHM, & NWM data
    ### use same set of dates for all.

    df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', start_date, end_date, units = 'cms', source = 'USGS')

    df_obs_pub = pd.read_csv(f'{input_dir}modeled_gages\historic_reconstruction_daily_1960_2022_{obs_pub_donor_fdc}_mgd.csv', 
                         sep=',', index_col=0, parse_dates=True).loc[start_date:end_date, :]
    
    df_nhm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nhmv10_mgd.csv', start_date, end_date, units = 'mgd', source = 'nhm')

    df_nwm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nwmv21_mgd.csv', start_date, end_date, units = 'mgd', source = 'nwmv21')

    assert ((df_obs.index == df_nhm.index).mean() == 1) and ((df_nhm.index == df_nwm.index).mean() == 1)


    ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
    df_nhm = match_gages(df_nhm, 'nhmv10', site_matches_id= nhm_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_obs = match_gages(df_obs, 'obs', site_matches_id= obs_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_obs_pub = match_gages(df_obs_pub, 'obs_pub', site_matches_id= obs_pub_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_nwm = match_gages(df_nwm, 'nwmv21', site_matches_id= nwm_site_matches, upstream_nodes_dict= upstream_nodes_dict)
    
    ### now get NYC diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nyc_diversion = extrapolate_NYC_NJ_diversions('nyc')
    nyc_diversion.to_csv(f'{input_dir}deliveryNYC_ODRM_extrapolated.csv', index=False)

    ### now get NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nj_diversion = extrapolate_NYC_NJ_diversions('nj')
    nj_diversion.to_csv(f'{input_dir}deliveryNJ_WEAP_23Aug2022_gridmet_extrapolated.csv', index=False)


    ### get catchment demands based on DRBC data
    sw_demand = disaggregate_DRBC_demands()
    sw_demand.to_csv(f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv', index_label='node')


    # ### organize WEAP results to use in Pywr-DRB
    # reservoirs = ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'prompton', 'mongaupeCombined',
    #             'beltzvilleCombined', 'blueMarsh', 'nockamixon', 'ontelaunee', 'assunpink']
    # major_flows = ['delLordville', 'delMontague', 'delDRCanal', 'delTrenton', 'outletAssunpink', 'outletSchuylkill', 'outletChristina',
    #               '01425000', '01417000', '01436000', '01433500', '01449800',
    #               '01447800', '01463620', '01470960']
    #
    # ### first get reservoir inflows and outflows from WEAP
    # for reservoir in reservoirs:
    #     filename = f'{weap_dir}WEAP_DRB_23Aug2022_gridmet_ResInOut_{reservoir}.csv'
    #     df = get_WEAP_df(filename, 'Sources and Destinations')
    #
    #     if reservoir == 'cannonsville':
    #         inflows = pd.DataFrame({reservoir: df['Inflow from Upstream']})
    #         releases = pd.DataFrame({reservoir: np.abs(df['Outflow to Downstream'])})
    #     else:
    #         inflows[reservoir] = df['Inflow from Upstream']
    #         releases[reservoir] = np.abs(df['Outflow to Downstream'])
    # ### convert cubic meter to MG
    # inflows *= cm_to_mg
    # releases *= cm_to_mg
    # ### save
    # inflows.to_csv(f'{input_dir}catchment_inflow_WEAP_23Aug2022_gridmet.csv')
    # releases.to_csv(f'{input_dir}releases_WEAP_23Aug2022_gridmet.csv')
    #
    # ### now get modeled reservoir storages
    # for reservoir in reservoirs:
    #     filename = f'{weap_dir}/WEAP_DRB_23Aug2022_gridmet_ResStoreZones_{reservoir}.csv'
    #     df = get_WEAP_df(filename, 'Variable')
    #
    #     if reservoir == 'cannonsville':
    #         storage = pd.DataFrame({reservoir: df['Storage Volume']})
    #     else:
    #         storage[reservoir] = df['Storage Volume']
    # ### convert cubic meter to MG
    # storage *= cm_to_mg
    # ### save
    # storage.to_csv(f'{input_dir}storages_WEAP_23Aug2022_gridmet.csv')
    #
    # ### now get observed reservoir storages
    # for reservoir in reservoirs:
    #     try:
    #         filename = f'{weap_dir}/WEAP_DRB_23Aug2022_gridmet_ResStorGage_{reservoir}.csv'
    #         df = get_WEAP_df(filename, 'Variable')
    #
    #         if reservoir == 'cannonsville':
    #             storageObs = pd.DataFrame({reservoir: df['Observed']})
    #         else:
    #             storageObs[reservoir] = df['Observed']
    #     except:
    #         # print('no observed storage for ', reservoir)
    #         pass
    #
    # ### convert cubic meter to MG
    # storageObs *= cm_to_mg
    # ### save
    # storageObs.to_csv(f'{input_dir}storageObs_WEAP_23Aug2022_gridmet.csv')
    #
    # ### now get flow gages
    # filenames = glob.glob(f'{weap_dir}/*flowGage*')
    # for node in reservoirs + major_flows:
    #     try:
    #         filename = [f for f in filenames if node in f][0]
    #         df = get_WEAP_df(filename, 'Statistic')
    #
    #         if node == 'cannonsville':
    #             flow = pd.DataFrame({node: df['Modeled']})
    #         else:
    #             flow[node] = df['Modeled']
    #     except:
    #         # print('no streamflow gage data for ', reservoir)
    #         pass
    #
    # ### convert cubic meter to MG
    # flow *= cm_to_mg
    # ### save
    # flow.to_csv(f'{input_dir}gage_flow_WEAP_23Aug2022_gridmet.csv')


