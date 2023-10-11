"""
Organize data records into appropriate format for Pywr-DRB.

Observed records (USGS gages) & modeled estimates (NHM, NWM, WEAP).

"""

import numpy as np
import pandas as pd
import datetime

from pywrdrb.pywr_drb_node_data import obs_site_matches, obs_pub_site_matches, nhm_site_matches, nwm_site_matches, \
                               upstream_nodes_dict, WEAP_29June2023_gridmet_NatFlows_matches, downstream_node_lags
from pywrdrb.utils.constants import cfs_to_mgd, cms_to_mgd, cm_to_mg, mcm_to_mg
from pywrdrb.utils.directories import input_dir, weap_dir
from pywrdrb.utils.lists import reservoir_list_nyc



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
    df_gages = df.iloc[:ndays ,:].loc[:, [site_label]]
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
    inflows = inflows.copy()
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

        ### delTrenton node should have zero catchment inflow because coincident with DRCanal
        ### -> make sure that is still so after subtraction process
        inflows['delTrenton'] *= 0.

    return inflows


def add_upstream_catchment_inflows(inflows, exclude_NYC = False):
    """
    Adds upstream catchment inflows to get cumulative flow at downstream nodes. THis is inverse of subtract_upstream_catchment_inflows()

    Inflow timeseries are cumulative. For each downstream node, this function adds the flow into all upstream nodes so
    that it represents cumulative inflows into the downstream node. It also accounts for time lags between distant nodes.

    Args:
        inflows (pandas.DataFrame): The inflows timeseries dataframe.

    Returns:
        pandas.DataFrame: The modified inflows timeseries dataframe with upstream catchment inflows added.
    """
    ### loop over upstream_nodes_dict in reverse direction to avoid double counting
    inflows = inflows.copy()
    for node in list(upstream_nodes_dict.keys())[::-1]:
        for upstream in upstream_nodes_dict[node]:
            if exclude_NYC == False or upstream not in reservoir_list_nyc:
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


def match_gages(df, dataset_label, site_matches_id):
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

    if 'obs_pub' not in dataset_label:
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



def create_hybrid_modeled_observed_datasets(modeled_inflow_type, datetime_index):
    ### create combo dataset that uses observed scaled data where available & NHM/NWM data everywhere else
    inflow_label_scaled = f'obs_pub_{modeled_inflow_type}_ObsScaled'
    inflow_label_nonScaled = modeled_inflow_type
    inflows = pd.read_csv(f'{input_dir}/catchment_inflow_{inflow_label_nonScaled}.csv')
    scaled_obs = pd.read_csv(f'{input_dir}/catchment_inflow_{inflow_label_scaled}.csv')
    inflows.index = pd.DatetimeIndex(inflows['datetime'])
    scaled_obs.index = pd.DatetimeIndex(scaled_obs['datetime'])
    inflows = inflows.loc[datetime_index]
    scaled_obs = scaled_obs.loc[datetime_index]
    for reservoir in reservoir_list_nyc + ['fewalter','beltzvilleCombined']:
        inflows[reservoir] = scaled_obs[reservoir]

    new_label = inflow_label_nonScaled + '_withObsScaled'

    ## Save catchment inflows to csv
    inflows.to_csv(f'{input_dir}catchment_inflow_{new_label}.csv', index=False)

    ## Now get full gage flow we want to re-add up cumulative flows after doing previous catchment subtraction.
    # For downstream nodes, this represents the full flow for results comparison
    inflows = add_upstream_catchment_inflows(inflows)
    inflows.to_csv(f'{input_dir}gage_flow_{new_label}.csv', index=False)



def get_WEAP_df(filename):
    ### new file format for 29June2023 WEAP
    df = pd.read_csv(filename)
    df.columns = ['year', 'doy', 'flow', '_']
    df['datetime'] = [datetime.datetime(y, 1, 1) + datetime.timedelta(d - 1) for y, d in zip(df['year'], df['doy'])]
    df.index = pd.DatetimeIndex(df['datetime'])
    # df = df.loc[np.logical_or(df['doy'] != 366, df.index.month != 1)]
    df = df[['flow']]
    return df


def prep_WEAP_data():
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


