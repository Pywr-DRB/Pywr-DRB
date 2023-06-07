import numpy as np
import pandas as pd
import glob
import statsmodels.api as sm

from pywr_drb_node_data import upstream_nodes_dict
from utils.constants import cfs_to_mgd, cms_to_mgd, cm_to_mg
from utils.directories import input_dir

# Directories
weap_dir = input_dir + 'WEAP_23Aug2022_gridmet/'

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
