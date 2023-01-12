### organize observed records (USGS gages) & modeled estimates (NHM, NWM, WEAP) into appropriate format for Pywr-DRB
### reorganizing/consolidating code from explore_streamflows.ipynb

import numpy as np
import pandas as pd
import glob


fig_dir = 'figs/'
input_dir = 'input_data/'
weap_dir = input_dir + 'WEAP_23Aug2022_gridmet/'

cms_to_mgd = 22.82
cm_to_mg = 264.17/1e6
cfs_to_mgd = 0.0283 * 22824465.32 / 1e6


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

### read in observed, NHM, & NWM data at gages downstream of reservoirs, as well as NWM inflows to lake objects.
### use same set of dates for all.
df_obs = read_modeled_estimates(f'{input_dir}modeled_gages/streamflow_daily_nhmv10.txt',
                                '\t', 'UTC_date', 'site_no', 'q_cms_obs', '1983/10/01', '2016/12/31')
df_nhm = read_modeled_estimates(f'{input_dir}modeled_gages/streamflow_daily_nhmv10.txt',
                                '\t', 'UTC_date', 'site_no', 'q_cms_mod', '1983/10/01', '2016/12/31')
df_nwm = read_modeled_estimates(f'{input_dir}modeled_gages/streamflow_daily_nwmv21.txt',
                                '\t', 'UTC_date', 'site_no', 'q_cms_mod', '1983/10/01', '2016/12/31')
df_nwm_lakes = read_modeled_estimates(f'{input_dir}modeled_gages/lakes_daily_1979_2020_nwmv21.csv',
                                    ',', 'UTC_date', 'feature_id', 'inflow', '1983/10/01', '2016/12/31')
assert ((df_obs.index == df_nhm.index).mean() == 1) and ((df_nhm.index == df_nwm.index).mean() == 1) and ((df_nhm.index == df_nwm_lakes.index).mean() == 1)

def match_gages(df, dataset_label):
    '''Matches USGS gage sites to nodes in Pywr-DRB.
    For reservoirs, the matched gages are actually downstream, but assume this flows into reservoir from upstream catchment.
    For mainstem nodes, upstream reservoir inflows are subtracted from the flow at mainstem USGS gage.
    Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.'''

    ### dictionary matching gages to reservoir catchments
    site_matches_reservoir = {'cannonsville': '01425000',
                              'pepacton': '01417000',
                              'neversink': '01436000',
                              'wallenpaupack': '01429000', ## Note, wanted to do 01431500 minus 01432110, but don't have latter from Aubrey, so use Prompton for now
                              'prompton': '01429000',
                              'shoholaMarsh': '01429000', ## Note, should have 01432495, but not didnt get from Aubrey, so use Prompton for now
                              'mongaupeCombined': '01433500',
                              'beltzvilleCombined': '01449800',
                              'fewalter': '01447800',
                              'merrillCreek': '01459500', ## Merrill Creek doesnt have gage - use Nockamixon nearby to get flow shape
                              'hopatcong': '01455500',
                              'nockamixon': '01459500',
                              'assunpink': '01463620',
                              'ontelaunee': '01470960', ## Note, should have 01470761, but didnt get from Aubrey, so use Blue Marsh for now
                              'stillCreek': '01469500',
                              'blueMarsh': '01470960',
                              'greenLane': '01473000',
                              'marshCreek': '01480685'
                              }
    ### list of lists, containing mainstem nodes, matching USGS gages, and upstream nodes to subtract
    site_matches_link = [['delLordville', ['01427207'], ['cannonsville', 'pepacton']],
                         ['delMontague', ['01438500'], ['prompton', 'wallenpaupack', 'delLordville', 'shoholaMarsh', 'mongaupeCombined', 'neversink']],
                         ['delTrenton', ['01463500'], ['delMontague', 'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon']],
                         ['outletAssunpink', ['01463620'], ['assunpink']], ## note, should get downstream junction, just using reservoir-adjacent gage for now
                         ['outletSchuylkill', ['01474500'], ['ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane']],
                         ['outletChristina', ['01480685'], ['marshCreek']] ## note, should use ['01481500, 01480015, 01479000, 01478000'], but dont have yet. so use marsh creek gage for now.
                         ]

    ### first match inflows for reservoirs
    for reservoir, site in site_matches_reservoir.items():
        if reservoir == 'cannonsville':
            inflow = pd.DataFrame(df.loc[:, site])
            inflow.columns = [reservoir]
            ## reset date column to be 'datetime'
            inflow['datetime'] = inflow.index
            inflow.index = inflow['datetime']
            inflow = inflow.iloc[:, :-1]
        else:
            inflow[reservoir] = df[site]

    ## now setup inflows for mainstem nodes
    for node, sites, upstreams in site_matches_link:
        inflow[node] = df.loc[:, sites].sum(axis=1)

    ## save full flow version of data to csv -> for downstream nodes, this represents the full flow for results comparison
    inflow.to_csv(f'{input_dir}gage_flow_{dataset_label}.csv')

    ## now for mainstem nodes, subtract flows into upstream nodes so that this represents only the catchment inflows
    for node, sites, upstreams in site_matches_link:
        inflow[node] -= inflow.loc[:, upstreams].sum(axis=1)
        ## make sure no flows are negative after subtraction
        inflow[node].loc[inflow[node] < 0] = 0

    ## save catchment inflow version of data to csv -> for downstream nodes, this represents the catchment inflow with upstream node inflows subtracted
    inflow.to_csv(f'{input_dir}catchment_inflow_{dataset_label}.csv')

    return inflow

### match USGS gage sites to Pywr-DRB model nodes & save inflows to csv file in format expected by Pywr-DRB
df_obs = match_gages(df_obs, 'obs')
df_nhm = match_gages(df_nhm, 'nhmv10')
df_nwm = match_gages(df_nwm, 'nwmv21')



### match NWM lake objects to Pywr-DRB model nodes. For nodes with no NWM lake object, default to NWM downstream flow (df_nwm)
def match_nwm_lakes(df_nwm_USGSflow, df_nwm_lakeflow, dataset_label):
    '''Matches NWM lakes to nodes in Pywr-DRB, based on nwm_feature_id. Where no lake object exists, use NWM downstream flow.
    Saves csv file, & returns dataframe whose columns are names of Pywr-DRB nodes.'''

    ### dictionary matching gages to reservoir catchments
    site_matches_reservoir = {'cannonsville': '2613174',
                              'pepacton': '1748473',
                              'neversink': '4146742',
                              'wallenpaupack': '2741600',
                              'prompton': '2739068',
                              'shoholaMarsh': '120052035',
                              'mongaupeCombined': '4148582',
                              'beltzvilleCombined': '4186689',
                              'fewalter': '4185065',
                              'merrillCreek': 'none',
                              'hopatcong': '2585287',
                              'nockamixon': 'none',
                              'assunpink': '2589015',
                              'ontelaunee': '4779981',
                              'stillCreek': '4778721',
                              'blueMarsh': '4782813',
                              'greenLane': '4780087',
                              'marshCreek': 'none',
                              'delLordville': 'none',
                              'delMontague': 'none',
                              'delTrenton': 'none',
                              'outletAssunpink': 'none',
                              'outletSchuylkill': 'none',
                              'outletChristina': 'none'
                              }
    df_matched = df_nwm_USGSflow.copy()
    for c in df_matched.columns:
        k = site_matches_reservoir[c]
        if k != 'none':
            df_matched[c] = df_nwm_lakeflow[int(k)]

    ## save catchment inflow version of data to csv -> for downstream nodes, this represents the catchment inflow with upstream node inflows subtracted
    df_matched.to_csv(f'{input_dir}catchment_inflow_{dataset_label}.csv')

    return df_matched

df_nwm_withLakes = match_nwm_lakes(df_nwm, df_nwm_lakes, 'nwmv21_withLakes')
# print(df_nwm_withLakes.mean(axis=0)/df_nwm.mean(axis=0))



### organize WEAP results to use in Pywr-DRB
reservoirs = ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'prompton', 'mongaupeCombined',
              'beltzvilleCombined', 'blueMarsh', 'nockamixon', 'ontelaunee', 'assunpink']


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


### first get reservoir inflows and outflows from WEAP
for reservoir in reservoirs:
    filename = f'{weap_dir}WEAP_DRB_23Aug2022_gridmet_ResInOut_{reservoir}.csv'
    df = get_WEAP_df(filename, 'Sources and Destinations')

    if reservoir == 'cannonsville':
        inflows = pd.DataFrame({reservoir: df['Inflow from Upstream']})
        releases = pd.DataFrame({reservoir: np.abs(df['Outflow to Downstream'])})
    else:
        inflows[reservoir] = df['Inflow from Upstream']
        releases[reservoir] = np.abs(df['Outflow to Downstream'])
### convert cubic meter to MG
inflows *= cm_to_mg
releases *= cm_to_mg
### save
inflows.to_csv(f'{input_dir}catchment_inflow_WEAP_23Aug2022_gridmet.csv')
releases.to_csv(f'{input_dir}releases_WEAP_23Aug2022_gridmet.csv')

### now get modeled reservoir storages
for reservoir in reservoirs:
    filename = f'{weap_dir}/WEAP_DRB_23Aug2022_gridmet_ResStoreZones_{reservoir}.csv'
    df = get_WEAP_df(filename, 'Variable')

    if reservoir == 'cannonsville':
        storage = pd.DataFrame({reservoir: df['Storage Volume']})
    else:
        storage[reservoir] = df['Storage Volume']
### convert cubic meter to MG
storage *= cm_to_mg
### save
storage.to_csv(f'{input_dir}storages_WEAP_23Aug2022_gridmet.csv')

### now get observed reservoir storages
for reservoir in reservoirs:
    try:
        filename = f'{weap_dir}/WEAP_DRB_23Aug2022_gridmet_ResStorGage_{reservoir}.csv'
        df = get_WEAP_df(filename, 'Variable')

        if reservoir == 'cannonsville':
            storageObs = pd.DataFrame({reservoir: df['Observed']})
        else:
            storageObs[reservoir] = df['Observed']
    except:
        print('no observed storage for ', reservoir)

### convert cubic meter to MG
storageObs *= cm_to_mg
### save
storageObs.to_csv(f'{input_dir}storageObs_WEAP_23Aug2022_gridmet.csv')

### now get flow gages
filenames = glob.glob(f'{weap_dir}/*flowGage*')
for reservoir in reservoirs:
    try:
        filename = [f for f in filenames if reservoir in f][0]
        df = get_WEAP_df(filename, 'Statistic')

        if reservoir == 'cannonsville':
            flow = pd.DataFrame({reservoir: df['Modeled']})
        else:
            flow[reservoir] = df['Modeled']
    except:
        print('no streamflow gage data for ', reservoir)

### convert cubic meter to MG
flow *= cm_to_mg
### save
flow.to_csv(f'{input_dir}gage_flow_WEAP_23Aug2022_gridmet.csv')

### now get NYC demands/deliveries
filenames = glob.glob(f'{weap_dir}/*delawareTunnelNYC*')
for reservoir in ['cannonsville', 'pepacton', 'neversink']:
    try:
        filename = [f for f in filenames if reservoir in f][0]
        df = get_WEAP_df(filename, 'Scenario')

        if reservoir == 'cannonsville':
            deliveryNYC = pd.DataFrame({reservoir: df['GridMet']})
        else:
            deliveryNYC[reservoir] = df['GridMet']
    except:
        print('no streamflow gage data for ', reservoir)

### convert cubic meter to MG
deliveryNYC *= cm_to_mg
### aggregate demand from 3 reservoirs
deliveryNYC['aggregate'] = deliveryNYC.sum(axis=1)
### save
deliveryNYC.to_csv(f'{input_dir}deliveryNYC_WEAP_23Aug2022_gridmet.csv')

### now get NJ demands/deliveries
filename = glob.glob(f'{weap_dir}/*Raritan*')[0]

deliveryNJ = pd.read_csv(filename, skiprows=1, header=None)
deliveryNJ.columns = ['datetime', 'D_R_Canal', 'dum']
deliveryNJ.index = pd.DatetimeIndex(deliveryNJ['datetime'])
deliveryNJ = deliveryNJ[['D_R_Canal']]

### convert cfs to mgd
deliveryNJ *= cfs_to_mgd

### flow becomes negative sometimes, presumably due to storms and/or drought reversing flow. dont count as deliveries
deliveryNJ[deliveryNJ < 0] = 0

### save
deliveryNJ.to_csv(f'{input_dir}deliveryNJ_WEAP_23Aug2022_gridmet.csv')

