"""
Organize data records into appropriate format for Pywr-DRB.

Observed records (USGS gages) & modeled estimates (NHM, NWM, WEAP).

"""
 
import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm
import datetime

from pywr_drb_node_data import obs_site_matches, obs_pub_site_matches, nhm_site_matches, nwm_site_matches, \
                               upstream_nodes_dict, WEAP_29June2023_gridmet_NatFlows_matches, downstream_node_lags
from utils.constants import cfs_to_mgd, cms_to_mgd, cm_to_mg, mcm_to_mg
from utils.directories import input_dir, weap_dir

from data_processing.disaggregate_DRBC_demands import disaggregate_DRBC_demands
from data_processing.extrapolate_NYC_NJ_diversions import extrapolate_NYC_NJ_diversions



def read_modeled_estimates(filename, sep, date_label, site_label, streamflow_label, start_date, end_date):
    """
    Reads input streamflows from modeled NHM/NWM estimates and prepares them for Pywr.

    Args:
        filename (str): The path or filename of the input file.
        sep (str): The separator used in the input file.
        date_label (str): The label for the date column in the input file.
        site_label (str): The label for the site column in the input file.
        streamflow_label (str): The label for the streamflow column in the input file.
        start_date (str): The start date for filtering the data (format: 'YYYY-MM-DD').
        end_date (str): The end date for filtering the data (format: 'YYYY-MM-DD').

    Returns:
        pandas.DataFrame: The resulting dataframe containing the filtered and restructured data.
    """
    
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
    """
    Reads in a pandas DataFrame containing USGS gauge data relevant to the model.

    Args:
        filename (str): The path or filename of the input file.
        start_date (str): The start date for filtering the data (format: 'YYYY-MM-DD').
        end_date (str): The end date for filtering the data (format: 'YYYY-MM-DD').
        units (str, optional): The units of the data. Default is 'cms'.
        source (str, optional): The data source. Default is 'USGS'.

    Returns:
        pandas.DataFrame: The resulting dataframe containing the filtered data.
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

def subtract_upstream_catchment_inflows(inflows):
    """
    Subtracts upstream catchment inflows from the input inflows timeseries.

    Inflow timeseries are cumulative. For each downstream node, this function subtracts the flow into all upstream nodes so
    that it represents only the direct catchment inflows into this node. It also accounts for time lags between distant nodes.

    Args:
        inflows (pandas.DataFrame): The inflows timeseries dataframe.

    Returns:
        pandas.DataFrame: The modified inflows timeseries dataframe with upstream catchment inflows subtracted.
    """
    for node, upstreams in upstream_nodes_dict.items():
        for upstream in upstreams:
            lag = downstream_node_lags[upstream]
            if lag > 0:
                inflows[node].iloc[lag:] -= inflows[upstream].iloc[:-lag].values
                ### subtract same-day flow without lagging for first lag days, since we don't have data before 0 for lagging
                inflows[node].iloc[:lag] -= inflows[upstream].iloc[:lag].values
            else:
                inflows[node] -= inflows[upstream]

        ### if catchment inflow is negative after subtracting upstream, set to 0
        inflows[node].loc[inflows[node] < 0] = 0
    return inflows


def add_upstream_catchment_inflows(inflows):
    """
    Adds upstream catchment inflows to get cumulative flow at downstream nodes. THis is inverse of subtract_upstream_catchment_inflows()

    Inflow timeseries are cumulative. For each downstream node, this function adds the flow into all upstream nodes so
    that it represents cumulative inflows into the downstream node. It also accounts for time lags between distant nodes.

    Args:
        inflows (pandas.DataFrame): The inflows timeseries dataframe.

    Returns:
        pandas.DataFrame: The modified inflows timeseries dataframe with upstream catchment inflows added.
    """
    for node, upstreams in upstream_nodes_dict.items():
        for upstream in upstreams:
            lag = downstream_node_lags[upstream]
            if lag > 0:
                inflows[node].iloc[lag:] += inflows[upstream].iloc[:-lag].values
                ### add same-day flow without lagging for first lag days, since we don't have data before 0 for lagging
                inflows[node].iloc[:lag] += inflows[upstream].iloc[:lag].values
            else:
                inflows[node] += inflows[upstream]

        ### if catchment inflow is negative after adding upstream, set to 0 (note: this shouldnt happen)
        inflows[node].loc[inflows[node] < 0] = 0

    return inflows


def match_gages(df, dataset_label, site_matches_id, upstream_nodes_dict):
    """
    Matches USGS gage sites to nodes in Pywr-DRB.

    For reservoirs, the matched gages are actually downstream, but assume this flows into the reservoir from the upstream catchment.
    For river nodes, upstream reservoir inflows are subtracted from the flow at the river node USGS gage.
    For nodes related to USGS gages downstream of reservoirs, the currently redundant flow with assumed inflow is subtracted, resulting in an additional catchment flow of 0 until this is updated.
    Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        dataset_label (str): The label for the dataset.
        site_matches_id (dict): A dictionary containing the site matches for Pywr-DRB nodes.
        upstream_nodes_dict (dict): A dictionary containing the upstream nodes for each node.

    Returns:
        pandas.DataFrame: The resulting dataframe whose columns are names of Pywr-DRB nodes.
    """
    
    ### 1. Match inflows for each Pywr-DRB node 
    ## 1.1 Reservoir inflows
    for node, site in site_matches_id.items():
        if node == 'cannonsville':
            if ('obs_pub' in dataset_label) and (site == None):
                inflows = pd.DataFrame(df.loc[:, node])
            else:
                inflows = pd.DataFrame(df.loc[:, site].sum(axis=1))
            inflows.columns = [node]
            inflows['datetime'] = inflows.index
            inflows.index = inflows['datetime']
            inflows = inflows.iloc[:, :-1]
        else:
            if ('obs_pub' in dataset_label) and (site == None):
                inflows[node] = df[node]
            else:
                inflows[node] = df[site].sum(axis=1)
                
    if  'obs_pub' not in dataset_label:
        ## Save full flows to csv
        # For downstream nodes, this represents the full flow for results comparison
        inflows.to_csv(f'{input_dir}gage_flow_{dataset_label}.csv')

    ### 2. Inflow timeseries are cumulative. So for each downstream node, subtract the flow into all upstream nodes so
    ###    this represents only direct catchment inflows into this node. Account for time lags between distant nodes.
    inflows = subtract_upstream_catchment_inflows(inflows)

    ## Save catchment inflows to csv  
    # For downstream nodes, this represents the catchment inflow with upstream node inflows subtracted
    inflows.to_csv(f'{input_dir}catchment_inflow_{dataset_label}.csv')

    if 'obs_pub' in dataset_label:
        ## For PUB, to get full gage flow we want to re-add up cumulative flows after doing previous catchment subtraction.
        # For downstream nodes, this represents the full flow for results comparison
        inflows = add_upstream_catchment_inflows(inflows)
        inflows.to_csv(f'{input_dir}gage_flow_{dataset_label}.csv')

    return inflows




def get_WEAP_df(filename):
    ### new file format for 29June2023 WEAP
    df = pd.read_csv(filename)
    df.columns = ['year', 'doy', 'flow', '_']
    df['datetime'] = [datetime.datetime(y, 1, 1) + datetime.timedelta(d - 1) for y, d in zip(df['year'], df['doy'])]
    df.index = pd.DatetimeIndex(df['datetime'])
    # df = df.loc[np.logical_or(df['doy'] != 366, df.index.month != 1)]
    df = df[['flow']]
    return df




if __name__ == "__main__":
    
    ### read in observed, NHM, & NWM data
    ### use same set of dates for all.
    start_date = '1983/10/01'
    end_date = '2016/12/31'

    df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', start_date, end_date, units = 'cms', source = 'USGS')

    df_nhm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nhmv10_mgd.csv', start_date, end_date, units = 'mgd', source = 'nhm')

    df_nwm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nwmv21_mgd.csv', start_date, end_date, units = 'mgd', source = 'nwmv21')

    assert ((df_obs.index == df_nhm.index).mean() == 1) and ((df_nhm.index == df_nwm.index).mean() == 1)

    ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
    df_nhm = match_gages(df_nhm, 'nhmv10', site_matches_id= nhm_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_obs = match_gages(df_obs, 'obs', site_matches_id= obs_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_nwm = match_gages(df_nwm, 'nwmv21', site_matches_id= nwm_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    ### now prep input data for full PUB reconstruction using both NHM & NWM FDC methods
    start_date = '1950-01-01'
    end_date = '2022-12-31'

    # Loop over both NWM & NHM methods for FDC construction
    for obs_pub_donor_fdc in ['nwmv21', 'nhmv10']:
        regression_nhm_inflow_scaling = True

        # Hist. Reconst. names are based on method specs
        hist_reconst_filename = f'historic_reconstruction_daily_{obs_pub_donor_fdc}'
        hist_reconst_filename = f'{hist_reconst_filename}_NYCscaled' if regression_nhm_inflow_scaling else hist_reconst_filename

        df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs_1950_2022_cms.csv', start_date, end_date,
                               units='cms', source='USGS')
        df_obs.index = pd.to_datetime(df_obs.index)

        ### read in PUB reconstruction from Pywr-DRB/DRB-historic-reconstruction repo
        df_obs_pub = pd.read_csv(f'{input_dir}modeled_gages/{hist_reconst_filename}_mgd.csv',
                                 sep=',', index_col=0, parse_dates=True).loc[start_date:end_date, :]
        df_obs_pub.index = pd.to_datetime(df_obs_pub.index)

        ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
        df_obs = match_gages(df_obs, 'obs', site_matches_id=obs_site_matches, upstream_nodes_dict=upstream_nodes_dict)
        df_obs_pub = match_gages(df_obs_pub, f'obs_pub_{obs_pub_donor_fdc}_NYCScaling',
                                 site_matches_id=obs_pub_site_matches, upstream_nodes_dict=upstream_nodes_dict)

    ### now get NYC diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nyc_diversion = extrapolate_NYC_NJ_diversions('nyc')
    nyc_diversion.to_csv(f'{input_dir}deliveryNYC_ODRM_extrapolated.csv', index=False)

    ### now get NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nj_diversion = extrapolate_NYC_NJ_diversions('nj')
    nj_diversion.to_csv(f'{input_dir}deliveryNJ_DRCanal_extrapolated.csv', index=False)


    ### get catchment demands based on DRBC data
    sw_demand = disaggregate_DRBC_demands()
    sw_demand.to_csv(f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv', index_label='node')

    ### organize WEAP results to use in Pywr-DRB - new for 29June2023 WEAP format
    for node, filekey in WEAP_29June2023_gridmet_NatFlows_matches.items():
        if filekey:
            inflow_filename = f'{weap_dir}/{filekey[0]}_GridMet_NatFlows.csv'
            df_inflow = get_WEAP_df(inflow_filename)
            gageflow_filename = f'{weap_dir}/{filekey[0]}_GridMet_ManagedFlows_ObsNYCDiv.csv'
            df_gageflow = get_WEAP_df(gageflow_filename)

        ### We dont have inflows for 2 reservoirs that aren't in WEAP. just set to 0 inflow since they are small anyway.
        ### This wont change overall mass balance because this flow will now be routed through downstream node directly without regulation (next step).
        else:
            df_inflow = df_inflow * 0
            df_gageflow = df_gageflow * 0

        if node == 'cannonsville':
            inflows = pd.DataFrame({node: df_inflow['flow']})
            gageflows = pd.DataFrame({node: df_inflow['flow']})
        else:
            inflows[node] = df_inflow['flow']
            gageflows[node] = df_gageflow['flow']

    ### Inflow timeseries are cumulative. So for each downstream node, subtract the flow into all upstream nodes so
    ###    this represents only direct catchment inflows into this node. Account for time lags between distant nodes.
    inflows = subtract_upstream_catchment_inflows(inflows)

    ### convert cubic meter to MG
    inflows *= cm_to_mg
    gageflows *= cm_to_mg

    ### save
    inflows.to_csv(f'{input_dir}catchment_inflow_WEAP_29June2023_gridmet.csv')
    gageflows.to_csv(f'{input_dir}gage_flow_WEAP_29June2023_gridmet.csv')

