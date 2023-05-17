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
from utils.disaggregate_DRBC_demands import disaggregate_DRBC_demands

# Date range
start_date = '1983/10/01'
end_date = '2016/12/31'

# Directories
weap_dir = input_dir + 'WEAP_23Aug2022_gridmet/'


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

    # If using prediction in ungauged basins (PUB) for historic reconstruction, load PUB data
    if dataset_label in ['obs_pub']:
        use_pub = True
        df_pub = pd.read_csv(f'./input_data/modeled_gages/drb_pub_predicted_flows_nhmv10_mgd.csv', sep = ',', index_col=0)
        df_pub.index = pd.to_datetime(df_pub.index)
        df_pub = df_pub.loc[df.index.intersection(df_pub.index),:]
    else:
        use_pub = False

    ### 1. Match inflows for each Pywr-DRB node 
    ## 1.1 Reservoir inflows
    for node, site in site_matches_id.items():
        if node == 'cannonsville':
            if site == None:
                inflow = pd.DataFrame(df_pub.loc[:, node])
            else:
                inflow = pd.DataFrame(df.loc[:, site])
            inflow.columns = [node]
            ## reset date column to be 'datetime'
            inflow['datetime'] = inflow.index
            inflow.index = inflow['datetime']
            inflow = inflow.iloc[:, :-1]
        elif (site == None) and use_pub:
            inflow[node] = df_pub.loc[:, node]
        else:
            inflow[node] = df[site].sum(axis=1)
                 
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



def extrapolate_NYC_NJ_diversions(loc):
    '''
    Function for retrieving NYC & NJ historical diversions, and extrapolating into time periods where we don't have
    data based on seasonal flow regressions.
    :param loc: location to extrapolate: either "nyc" or "nj"
    :return: dataframe
    '''

    ### set seed for consistent results
    np.random.seed(1)

    ### get historical diversion data
    if loc == 'nyc':
        diversion = pd.read_excel(f'{input_dir}historic_NYC/Pep_Can_Nev_diversions_daily_2000-2021.xlsx', index_col=0)
        diversion = diversion.iloc[:, :3]
        diversion.index = pd.to_datetime(diversion.index)
        diversion['aggregate'] = diversion.sum(axis=1)
        ### convert CFS to MGD
        diversion *= cfs_to_mgd
    elif loc == 'nj':
        ### now get NJ demands/deliveries
        diversion = pd.read_csv(glob.glob(f'{weap_dir}/*Raritan*')[0], skiprows=1, header=None)
        diversion.columns = ['datetime', 'D_R_Canal', 'dum']
        diversion.index = pd.DatetimeIndex(diversion['datetime'])
        diversion = diversion[['D_R_Canal']]
        ### convert cfs to mgd
        diversion *= cfs_to_mgd
        ### flow becomes negative sometimes, presumably due to storms and/or drought reversing flow. dont count as deliveries
        diversion[diversion < 0] = 0

    ### get historical flow
    flow = pd.read_csv(f'{input_dir}catchment_inflow_obs_pub.csv', index_col=0)
    flow.index = pd.to_datetime(flow.index)

    ### get maximum overlapping timespan for diversions and flow
    flow = flow.loc[np.logical_and(flow.index >= diversion.index.min(), flow.index <= diversion.index.max())]
    diversion = diversion.loc[np.logical_and(diversion.index >= flow.index.min(), diversion.index <= flow.index.max())]
    assert np.all(flow.index == diversion.index)

    ### dataframe of daily states
    if loc == 'nyc':
        df = pd.DataFrame({'diversion':diversion['aggregate'],
                           'flow_log':np.log(flow[['cannonsville','pepacton','neversink']].sum(axis=1)),
                           'm': diversion.index.month,
                           'y': diversion.index.year})
    elif loc == 'nj':
        inflow_nodes = upstream_nodes_dict['delTrenton'] + ['delTrenton']

        df = pd.DataFrame({'diversion': diversion['D_R_Canal'],
                           'flow_log': np.log(flow[inflow_nodes].sum(axis=1)),
                           'm': diversion.index.month,
                           'y': diversion.index.year})

    ### get quarters for separate regressions
    def get_quarter(m):
        if m in (12,1,2):
            return 'DJF'
        elif m in (3,4,5):
            return 'MAM'
        elif m in (6,7,8):
            return 'JJA'
        elif m in (9,10,11):
            return 'SON'

    df['quarter'] = [get_quarter(m) for m in df['m']]
    quarters = ('DJF','MAM','JJA','SON')

    ### dataframe of monthly mean states
    df_m = df.resample('m').mean()
    df_m['quarter'] = [get_quarter(m) for m in df_m['m']]

    ### NJ diversion data are left skewed, so negate and then apply log transform
    if loc == 'nj':
        nj_trans_max = df_m['diversion'].max() + 5
        df_m['diversion'] = np.log(nj_trans_max - df_m['diversion'])


    ### get linear regression model for each quarter
    lrms = {q: sm.OLS(df_m['diversion'].loc[df_m['quarter'] == q],
                      sm.add_constant(df_m['flow_log'].loc[df_m['quarter'] == q])) for q in quarters}
    lrrs = {q: lrms[q].fit() for q in quarters}

    ### now get longer dataset of flows for extrapolation
    flow = pd.read_csv(f'{input_dir}catchment_inflow_obs_pub.csv', index_col=0)
    flow.index = pd.to_datetime(flow.index)

    if loc == 'nyc':
        df_long = pd.DataFrame({'flow_log': np.log(flow[['cannonsville','pepacton','neversink']].sum(axis=1)),
                                'm': flow.index.month,
                                'y': flow.index.year,
                                'q': [get_quarter(m) for m in flow.index.month]})
    elif loc == 'nj':
        df_long = pd.DataFrame({'flow_log': np.log(flow[inflow_nodes].sum(axis=1)),
                                'm': flow.index.month,
                                'y': flow.index.year,
                                'q': [get_quarter(m) for m in flow.index.month]})

    df_long_m = df_long.resample('m').mean()
    df_long_m['quarter'] = [get_quarter(m) for m in df_long_m['m']]

    ### use trained regression model to sample a delivery value for each month based on log flow.
    df_long_m['diversion_pred'] = 0.
    for i in range(df_long_m.shape[0]):
        q = df_long_m['quarter'].iloc[i]
        f = df_long_m['flow_log'].iloc[i]
        lrm = lrms[q]
        lrr = lrrs[q]
        exog = lrm.exog
        exog[:, 1] = f
        ### get randomly sampled value from linear regression model. throw out if negative
        pred = -1
        while pred < 0:
            pred = lrm.get_distribution(lrr.params, scale=np.var(lrr.resid), exog=exog).rvs()[0]
        df_long_m['diversion_pred'].iloc[i] = pred

    ### for NJ, transform data back to original scale
    if loc == 'nj':
        df_m['diversion'] = np.maximum(nj_trans_max - np.exp(df_m['diversion']), 0)
        df_long_m['diversion_pred'] = np.maximum(nj_trans_max - np.exp(df_long_m['diversion_pred']), 0)

    ### now get nearest neighbor in normalized 2d space of log-flow&diversion, within q.
    flow_bounds = [df_m['flow_log'].min(), df_m['flow_log'].max()]
    diversion_bounds = [df_m['diversion'].min(), df_m['diversion'].max()]

    df_m['flow_log_norm'] = (df_m['flow_log'] - flow_bounds[0]) / (flow_bounds[1] - flow_bounds[0])
    df_m['diversion_norm'] = (df_m['diversion'] - diversion_bounds[0]) / (diversion_bounds[1] - diversion_bounds[0])
    df_long_m['flow_log_norm'] = (df_long_m['flow_log'] - flow_bounds[0]) / (flow_bounds[1] - flow_bounds[0])
    df_long_m['diversion_pred_norm'] = (df_long_m['diversion_pred'] - diversion_bounds[0]) / (diversion_bounds[1] - diversion_bounds[0])

    df_long_m['nn'] = -1
    for i in range(df_long_m.shape[0]):
        q = df_long_m['quarter'].iloc[i]
        f = df_long_m['flow_log_norm'].iloc[i]
        n = df_long_m['diversion_pred_norm'].iloc[i]
        df_m_sub = df_m.loc[df_m['quarter'] == q]
        dist_squ = (f - df_m_sub['flow_log_norm']) **2 + (n - df_m_sub['diversion_norm']) **2
        nn = np.argmin(dist_squ)
        df_long_m['nn'].iloc[i] = df_m_sub.index[nn]
    df_long_m['nn'].hist()

    ### now use each month's nearest neighbor to get flow shape for predicted diversion at daily time step
    df_long['diversion_pred'] = -1
    for i,row in df_long_m.iterrows():
        m = row['m']
        y = row['y']
        ### get the daily diversions in nearest neighbor from shorter record
        df_long_idx = df_long.loc[np.logical_and(df_long['m'] == m, df_long['y'] == y)].index
        df_m_match = df_m.loc[row['nn']]
        df_match = df.loc[np.logical_and(df['m'] == df_m_match['m'], df['y'] == df_m_match['y'])]
        ### scale daily diversions based on ratio of monthly prediction to match
        new_diversion = df_match['diversion'].values * row['diversion_pred'] / df_m_match['diversion']
        if np.any(new_diversion < 0):
            print(row, new_diversion)
        ### adjust length of record when months have different number of days
        len_new = len(df_long_idx)
        len_match = len(new_diversion)
        if len_match > len_new:
            new_diversion = new_diversion[:len_new]
        elif len_match < len_new:
            new_diversion = np.append(new_diversion, [new_diversion[-1]]*(len_new - len_match))
        df_long['diversion_pred'].loc[df_long_idx] = new_diversion

    ### Now reload historical diversion dataset, & add extrapolated data for the dates we don't have
    if loc == 'nyc':
        diversion = pd.read_excel(f'{input_dir}historic_NYC/Pep_Can_Nev_diversions_daily_2000-2021.xlsx', index_col=0)
        diversion = diversion.iloc[:,:3]
        diversion.index = pd.to_datetime(diversion.index)
        diversion['aggregate'] = diversion.sum(axis=1)
        ### convert CFS to MGD
        diversion *= cfs_to_mgd

        ### format & save to csv for use in Pywr-DRB
        df_long = df_long.loc[np.logical_or(df_long.index < diversion.index.min(), df_long.index > diversion.index.max())]
        diversion = diversion.append(pd.DataFrame({'aggregate': df_long['diversion_pred']}))
        diversion = diversion.sort_index()
        diversion['datetime'] = diversion.index
        diversion.columns = ['pepacton','cannonsville','neversink','aggregate', 'datetime']
        diversion = diversion.iloc[:, [-1,1,0,2,3]]

    elif loc == 'nj':
        diversion = pd.read_csv(glob.glob(f'{weap_dir}/*Raritan*')[0], skiprows=1, header=None)
        diversion.columns = ['datetime', 'D_R_Canal', 'dum']
        diversion.index = pd.DatetimeIndex(diversion['datetime'])
        diversion = diversion[['D_R_Canal']]
        ### convert cfs to mgd
        diversion *= cfs_to_mgd
        ### flow becomes negative sometimes, presumably due to storms and/or drought reversing flow. dont count as deliveries
        diversion[diversion < 0] = 0

        ### format & save to csv for use in Pywr-DRB
        df_long = df_long.loc[np.logical_or(df_long.index < diversion.index.min(), df_long.index > diversion.index.max())]
        diversion = diversion.append(pd.DataFrame({'D_R_Canal': df_long['diversion_pred']}))
        diversion = diversion.sort_index()
        diversion['datetime'] = diversion.index
        diversion = diversion.iloc[:, [-1, 0]]

    return diversion


if __name__ == "__main__":
    
    ### read in observed, NHM, & NWM data
    ### use same set of dates for all.

    df_obs = read_csv_data(f'{input_dir}usgs_gages/streamflow_daily_usgs.csv', start_date, end_date, units = 'cms', source = 'USGS')

    df_nhm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nhmv10_mgd.csv', start_date, end_date, units = 'mgd', source = 'nhm')

    df_nwm = read_csv_data(f'{input_dir}modeled_gages/streamflow_daily_nwmv21_mgd.csv', start_date, end_date, units = 'mgd', source = 'nwmv21')

    assert ((df_obs.index == df_nhm.index).mean() == 1) and ((df_nhm.index == df_nwm.index).mean() == 1)


    ### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
    df_nhm = match_gages(df_nhm, 'nhmv10', site_matches_id= nhm_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_obs_copy = df_obs.copy()
    df_obs = match_gages(df_obs, 'obs', site_matches_id= obs_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_obs_pub = match_gages(df_obs_copy, 'obs_pub', site_matches_id= obs_pub_site_matches, upstream_nodes_dict= upstream_nodes_dict)

    df_nwm = match_gages(df_nwm, 'nwmv21', site_matches_id= nwm_site_matches, upstream_nodes_dict= upstream_nodes_dict)


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


    ### now get NYC diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nyc_diversion = extrapolate_NYC_NJ_diversions('nyc')
    nyc_diversion.to_csv(f'{input_dir}deliveryNYC_ODRM_extrapolated.csv', index=False)

    ### now get NJ diversions. for time periods we dont have historical record, extrapolate by seasonal relationship to flow.
    nj_diversion = extrapolate_NYC_NJ_diversions('nj')
    nj_diversion.to_csv(f'{input_dir}deliveryNJ_WEAP_23Aug2022_gridmet_extrapolated.csv', index=False)


    ### get catchment demands based on DRBC data
    sw_demand = disaggregate_DRBC_demands()
    sw_demand.to_csv(f'{input_dir}sw_avg_wateruse_Pywr-DRB_Catchments.csv', index_label='node')

