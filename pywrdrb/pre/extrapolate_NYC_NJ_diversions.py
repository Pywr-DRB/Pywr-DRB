import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pygeohydro import NWIS
import datetime

from pywrdrb.pywr_drb_node_data import upstream_nodes_dict
from pywrdrb.utils.constants import cfs_to_mgd, cms_to_mgd, cm_to_mg
from pywrdrb.utils.directories import input_dir, fig_dir


def download_USGS_data_NYC_NJ_diversions():
    ### NYC reservoir upstream gages are selected based on historical data. all of these have complete data back to 1952.
    dates = ('1952-01-01', '2022-12-31')
    nwis = NWIS()
    gages = ['01460440', '01463500', '01423000', '01415000', '01413500', '01414500', '01435000']
    labels = ['D_R_Canal', 'delTrenton','cannonsville1', 'pepacton1', 'pepacton2', 'pepacton3', 'neversink1']
    hist_flows = nwis.get_streamflow(gages, dates)
    hist_flows['datetime'] = pd.to_datetime(hist_flows.index.date)
    labels.append('datetime')
    hist_flows.reset_index(inplace=True, drop=True)
    hist_flows = hist_flows[['USGS-'+g for g in gages] + ['datetime']]
    hist_flows.columns = labels

    for reservoir in ['cannonsville','pepacton','neversink']:
        hist_flows[reservoir] = hist_flows[[c for c in labels if reservoir in c]].sum(axis=1)
    hist_flows['NYC_inflow'] = hist_flows[['cannonsville','pepacton','neversink']].sum(axis=1)
    hist_flows = hist_flows[['datetime','D_R_Canal', 'delTrenton', 'NYC_inflow']]
    hist_flows.to_csv(input_dir + '/usgs_gages/streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv', index=False)
    return hist_flows


def extrapolate_NYC_NJ_diversions(loc, make_figs):
    """
    Function for retrieving NYC and NJ historical diversions and extrapolating them into time periods
    where we don't have data based on seasonal flow regressions.

    Args:
        loc (str): The location to extrapolate. Can be either "nyc" or "nj".

    Returns:
        pd.DataFrame: The dataframe containing the extrapolated diversions.
    """

    ### set seed for consistent results
    np.random.seed(1)

    ### get historical diversion data
    if loc == 'nyc':
        diversion = pd.read_excel(f'{input_dir}historic_NYC/Pep_Can_Nev_diversions_daily_2000-2021.xlsx', index_col=0)
        diversion = diversion.iloc[:, :3]
        diversion.index = pd.to_datetime(diversion.index)
        diversion['aggregate'] = diversion.sum(axis=1)
        diversion = diversion.loc[np.logical_not(np.isnan(diversion['aggregate']))]
        ### convert CFS to MGD
        diversion *= cfs_to_mgd
    elif loc == 'nj':
        ### now get NJ demands/deliveries
        ### The gage for D_R_Canal starts 1989-10-23, but lots of NA's early on. Pretty good after 1991-01-01, but a few remaining to clean up.
        start_date = (1991, 1, 1)
        diversion = pd.read_csv(input_dir + '/usgs_gages/streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv')
        diversion.index = pd.DatetimeIndex(diversion['datetime'])
        diversion = diversion[['D_R_Canal']]
        diversion = diversion.loc[diversion.index >= datetime.datetime(*start_date)]

        ### infill NA values with previous day's flow
        for i in range(1, diversion.shape[0]):
            if np.isnan(diversion['D_R_Canal'].iloc[i]):
                diversion['D_R_Canal'].iloc[i] = diversion['D_R_Canal'].iloc[i-1]

        ### convert cms to mgd
        diversion *= cms_to_mgd
        ### flow becomes negative sometimes, presumably due to storms and/or drought reversing flow. dont count as deliveries
        diversion[diversion < 0] = 0

    ### get historical flows
    flow = pd.read_csv(input_dir + '/usgs_gages/streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv')
    flow.index = pd.to_datetime(flow['datetime'])

    ### get maximum overlapping timespan for diversions and flow
    flow = flow.loc[np.logical_and(flow.index >= diversion.index.min(), flow.index <= diversion.index.max())]
    diversion = diversion.loc[np.logical_and(diversion.index >= flow.index.min(), diversion.index <= flow.index.max())]
    assert np.all(flow.index == diversion.index)

    ### dataframe of daily states
    if loc == 'nyc':
        df = pd.DataFrame({'diversion':diversion['aggregate'],
                           'flow_log':np.log(flow['NYC_inflow']),
                           'm': diversion.index.month,
                           'y': diversion.index.year})
    elif loc == 'nj':
        df = pd.DataFrame({'diversion': diversion['D_R_Canal'],
                           'flow_log': np.log(flow['delTrenton']),
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



    ### dataframe of monthly mean states
    df_m = df.resample('m').mean()

    quarters = ('DJF','MAM','JJA','SON')
    df['quarter'] = [get_quarter(m) for m in df['m']]
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
    flow = pd.read_csv(input_dir + '/usgs_gages/streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv')
    flow.index = pd.to_datetime(flow['datetime'])


    if loc == 'nyc':
        df_long = pd.DataFrame({'flow_log': np.log(flow['NYC_inflow']),
                                'm': flow.index.month,
                                'y': flow.index.year})
    elif loc == 'nj':
        df_long = pd.DataFrame({'flow_log': np.log(flow['delTrenton']),
                                'm': flow.index.month,
                                'y': flow.index.year})

    df_long_m = df_long.resample('m').mean()

    df_long['quarter'] = [get_quarter(m) for m in df_long['m']]
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



    ### plot regressions
    if make_figs == True:
        fig, axs = plt.subplots(2, 2, figsize=(8,8), gridspec_kw={'hspace':0.2, 'wspace':0.2})
        for i, q in enumerate(quarters):
            if i >= 2:
                row = 1
            else:
                row = 0
            if i % 2 == 1:
                col = 1
            else:
                col = 0
            ax = axs[row, col]

            ### first plot observed data
            data = df_m.loc[df_m['quarter'] == q].copy()
            ax.scatter(data['flow_log'], data['diversion'], zorder=2, alpha=0.7, color='cornflowerblue',
                       label='Observed')
            ### now plot sampled data during observed period
            data = df_long_m.loc[df_m.index].copy()
            data = data.loc[data['quarter'] == q]
            ax.scatter(data['flow_log'], data['diversion_pred'], zorder=1, alpha=0.7, color='firebrick',
                       label='Extrapolated over\nobserved period')
            ### now plot sampled data during unobserved period
            data = df_long_m.loc[[i not in df_m.index for i in df_long_m.index]].copy()
            ax.scatter(data['flow_log'], data['diversion_pred'], zorder=0, alpha=0.7, color='darkgoldenrod',
                       label = 'Extrapolated over\nunobserved period')

            ### plot regression line
            xlim = ax.get_xlim()
            ax.plot(xlim, [lrrs[q].params[0] + lrrs[q].params[1] * x for x in xlim], color='k', label='Regression')

            ### legend
            if row == 1 and col == 1:
                ax.legend(loc='center left', bbox_to_anchor=(1.0, 1.1), frameon=False)

            ### clean up
            ax.set_title(q)
            if row == 1:
                ax.set_xlabel('Log inflow (log MGD)')
            if loc == 'nyc':
                ax.set_ylim([0, ax.get_ylim()[1]])
            if loc == 'nyc' and col == 0:
                ax.set_ylabel('Monthly NYC diversion (MGD)')
            elif loc == 'nj' and col == 0:
                ax.set_ylabel('Transformed monthly NJ diversion')

        plt.savefig(f'{fig_dir}/extrapolation_{loc}_pt1.png', dpi=400, bbox_inches='tight')


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


    ### plot timeseries
    if make_figs == True:
        fig, ax = plt.subplots(1,1, figsize=(5,3), gridspec_kw={'hspace':0.2, 'wspace':0.2})

        ### plot observed diversion daily timeseries
        ax.plot(df['diversion'], color='cornflowerblue', label='Observed', zorder=2, lw=0.5, alpha=0.7)
        ### plot extrapolated
        ax.plot(df_long['diversion_pred'], color='darkgoldenrod', label='Extrapolated', zorder=1, lw=0.5, alpha=0.7)

        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)

        ### clean up
        ax.set_ylim([0, ax.get_ylim()[1]])
        if loc == 'nyc':
            ax.set_ylabel('Daily NYC diversion (MGD)')
        elif loc == 'nj':
            ax.set_ylabel('Daily NJ diversion (MGD)')

        plt.savefig(f'{fig_dir}/extrapolation_{loc}_pt2.png', dpi=400, bbox_inches='tight')


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
        diversion = pd.concat([diversion, pd.DataFrame({'aggregate': df_long['diversion_pred']})])
        diversion = diversion.sort_index()
        diversion['datetime'] = diversion.index
        diversion.columns = ['pepacton','cannonsville','neversink','aggregate', 'datetime']
        diversion = diversion.iloc[:, [-1,1,0,2,3]]

        diversion.to_csv(f'{input_dir}deliveryNYC_ODRM_extrapolated.csv', index=False)


    elif loc == 'nj':
        ### now get NJ demands/deliveries
        ### The gage for D_R_Canal starts 1989-10-23, but lots of NA's early on. Pretty good after 1991-01-01, but a few remaining to clean up.
        start_date = (1991, 1, 1)
        diversion = pd.read_csv(input_dir + '/usgs_gages/streamflow_daily_usgs_1950_2022_cms_for_NYC_NJ_diversions.csv')
        diversion.index = pd.DatetimeIndex(diversion['datetime'])
        diversion = diversion[['D_R_Canal']]
        diversion = diversion.loc[diversion.index >= datetime.datetime(*start_date)]

        ### infill NA values with previous day's flow
        for i in range(1, diversion.shape[0]):
            if np.isnan(diversion['D_R_Canal'].iloc[i]):
                diversion['D_R_Canal'].iloc[i] = diversion['D_R_Canal'].iloc[i - 1]

        ### convert cms to mgd
        diversion *= cms_to_mgd
        ### flow becomes negative sometimes, presumably due to storms and/or drought reversing flow. dont count as deliveries
        diversion[diversion < 0] = 0

        ### format & save to csv for use in Pywr-DRB
        df_long = df_long.loc[np.logical_or(df_long.index < diversion.index.min(), df_long.index > diversion.index.max())]
        diversion = pd.concat([diversion, pd.DataFrame({'D_R_Canal': df_long['diversion_pred']})])
        diversion = diversion.sort_index()
        diversion['datetime'] = diversion.index
        diversion = diversion.iloc[:, [-1, 0]]

        diversion.to_csv(f'{input_dir}deliveryNJ_DRCanal_extrapolated.csv', index=False)




