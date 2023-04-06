import numpy as np
import pandas as pd
import h5py
import hydroeval as he
from scipy import stats
import sys

from plotting.plotting_functions import plot_3part_flows, plot_weekly_flow_distributions
from plotting.plotting_functions import plot_radial_error_metrics, plot_rrv_metrics, plot_flow_contributions
from plotting.plotting_functions import compare_inflow_data

### I was having trouble with interactive console plotting in Pycharm for some reason - comment this out if you want to use that and not having issues
#mpl.use('TkAgg')

### directories
output_dir = 'output_data/'
input_dir = 'input_data/'
fig_dir = 'figs/'

# Constants
cms_to_mgd = 22.82
cm_to_mg = 264.17/1e6
cfs_to_mgd = 0.0283 * 22824465.32 / 1e6


### list of reservoirs and major flow points to compare across models
reservoir_list = ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'prompton', 'shoholaMarsh', \
                   'mongaupeCombined', 'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', \
                   'assunpink', 'ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', 'marshCreek']

majorflow_list = ['delLordville', 'delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill', 'outletChristina',
                  '01425000', '01417000', '01436000', '01433500', '01449800',
                  '01447800', '01463620', '01470960']

reservoir_link_pairs = {'cannonsville': '01425000',
                           'pepacton': '01417000',
                           'neversink': '01436000',
                           'mongaupeCombined': '01433500',
                           'beltzvilleCombined': '01449800',
                           'fewalter': '01447800',
                           'assunpink': '01463620',
                           'blueMarsh': '01470960'}


def get_pywr_results(output_dir, model, results_set='all', scenario = 0):
    '''
    Gathers simulation results from Pywr model run and returns a pd.DataFrame. 
    
    :param output_dir:
    :param model:
    :param results_set: can be "all" to return all results,
                            "res_release" to return reservoir releases (downstream gage comparison),
                            "res_stroage" to return resrvoir storages,
                            "major_flow" to return flow at major flow points of interest,
                            "inflow" to return the inflow at each catchment.
    :return:
    '''
    with h5py.File(f'{output_dir}drb_output_{model}.hdf5', 'r') as f:
        keys = list(f.keys())
        first = 0
        results = pd.DataFrame()
        for k in keys:
            if results_set == 'all':
                results[k] = f[k][:,scenario]
            elif results_set == 'res_release':
                if k.split('_')[0] == 'outflow' and k.split('_')[1] in reservoir_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'res_storage':
                if k.split('_')[0] == 'volume' and k.split('_')[1] in reservoir_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'major_flow':
                if k.split('_')[0] == 'link' and k.split('_')[1] in majorflow_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'inflow':
                if k.split('_')[0] == 'catchment':
                    results[k.split('_')[1]] = f[k][:, scenario]

        day = [f['time'][i][0] for i in range(len(f['time']))]
        month = [f['time'][i][2] for i in range(len(f['time']))]
        year = [f['time'][i][3] for i in range(len(f['time']))]
        date = [f'{y}-{m}-{d}' for y,m,d in zip(year, month, day)]
        date = pd.to_datetime(date)
        results.index = date
        return results


### load other flow estimates. each column represents modeled flow at USGS gage downstream of reservoir or gage on mainstem
def get_base_results(input_dir, model, datetime_index, results_set='all'):
    '''
    function for retreiving & organizing results from non-pywr results (NHM, NWM, WEAP)
    :param input_dir:
    :param model:
    :param datetime_index:
    :param results_set: can be "all" to return all results,
                            "res_release" to return reservoir releases (downstream gage comparison),
                            "major_flow" to return flow at major flow points of interest
    :return:
    '''
    gage_flow = pd.read_csv(f'{input_dir}gage_flow_{model}.csv')
    gage_flow.index = pd.DatetimeIndex(gage_flow['datetime'])
    gage_flow = gage_flow.drop('datetime', axis=1)
    if results_set == 'res_release':
        available_release_data = gage_flow.columns.intersection(reservoir_link_pairs.values())
        #reservoirs_with_data = [[k for k,v in reservoir_link_pairs.items() if v == site][0] for site in available_release_data]
        reservoirs_with_data = [list(filter(lambda x: reservoir_link_pairs[x] == site, reservoir_link_pairs))[0] for site in available_release_data]
        gage_flow = gage_flow.loc[:, available_release_data]
        gage_flow.columns = reservoirs_with_data
    elif results_set == 'major_flow':
        for c in gage_flow.columns:
            if c not in majorflow_list:
                gage_flow = gage_flow.drop(c, axis=1)
    gage_flow = gage_flow.loc[datetime_index,:]
    return gage_flow


def get_error_metrics(results, models, nodes):
    """Generate error metrics (NSE, KGE, correlation, bias, etc.) for a specific model and node.

    Args:
        results (dict): Dictionary containing dataframes of results.
        models (list): List of model names (str).
        nodes (list): List of node names (str).

    Returns:
        pd.DataFrame: Dataframe containing all error metrics.
    """
    ### compile error across models/nodes/metrics
    for j, node in enumerate(nodes):
        obs = results['obs'][node]
        for i, m in enumerate(models):
            # print(node, m)
            # print(results[m])
            modeled = results[m][node]

            ### only do models with nonzero entries (eg remove some weap)
            if np.sum(modeled) > 0:
                ### get kge & nse
                kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs)
                nse = he.evaluator(he.nse, modeled, obs)
                logkge, logr, logalpha, logbeta = he.evaluator(he.kge, modeled, obs, transform='log')
                lognse = he.evaluator(he.nse, modeled, obs, transform='log')

                ### get Kolmogorov-Smirnov Statistic, & metric is 1 minus this (1 in ideal case, 0 in worst case)
                kss, _ = stats.ks_2samp(modeled, obs)
                kss = 1 - kss

                resultsdict = {'nse': nse[0], 'kge': kge[0], 'r': r[0], 'alpha': alpha[0], 'beta': beta[0],
                               'lognse': lognse[0], 'logkge': logkge[0], 'logr': logr[0], 'logalpha': logalpha[0],
                               'logbeta': logbeta[0], 'kss': kss}

                resultsdict['node'] = node
                resultsdict['model'] = m
                if i == 0 and j == 0:
                    results_metrics = pd.DataFrame(resultsdict, index=[0])
                else:
                    results_metrics = results_metrics.append(pd.DataFrame(resultsdict, index=[0]))

    results_metrics.reset_index(inplace=True, drop=True)
    return results_metrics



### get measures of reliability, resilience, and vulnerability from Hashimoto et al 1982, WRR
def get_RRV_metrics(results, models, nodes):
    thresholds = {'delMontague':1131.05, 'delTrenton':1938.950669} ### FFMP flow targets (MGD)
    eps = 1e-9
    thresholds = {k:v-eps for k,v in thresholds.items()}
    for j, node in enumerate(nodes):
        for i, m in enumerate(models):
            modeled = results[m][node]

            ### only do models with nonzero entries (eg remove some weap)
            if np.sum(modeled) > 0:

                ### reliability is the fraction of time steps above threshold
                reliability = (modeled > thresholds[node]).mean()
                ### resiliency is the probability of recovering to above threshold if currently under threshold
                if reliability < 1- eps:
                    resiliency = np.logical_and((modeled.iloc[:-1] < thresholds[node]).reset_index(drop=True), \
                                                (modeled.iloc[1:] >= thresholds[node]).reset_index(drop=True)).mean() / \
                                 (1 - reliability)
                else:
                    resiliency = np.nan
                ### vulnerability is the expected maximum severity of a failure event
                if reliability > eps:
                    max_shortfalls = []
                    max_shortfall = 0
                    in_event = False
                    for i in range(len(modeled)):
                        v = modeled.iloc[i]
                        if v < thresholds[node]:
                            in_event = True
                            s = thresholds[node] - v
                            max_shortfall = max(max_shortfall, s)
                        else:
                            if in_event:
                                max_shortfalls.append(max_shortfall)
                                in_event = False
                    vulnerability = np.mean(max_shortfalls)
                else:
                    vulnerability = np.nan

                resultsdict = {'reliability': reliability, 'resiliency': resiliency, 'vulnerability': vulnerability}

                resultsdict['node'] = node
                resultsdict['model'] = m
                try:
                    rrv_metrics = rrv_metrics.append(pd.DataFrame(resultsdict, index=[0]))
                except:
                    rrv_metrics = pd.DataFrame(resultsdict, index=[0])

    rrv_metrics.reset_index(inplace=True, drop=True)
    return rrv_metrics




## Execution - Generate all figures
if __name__ == "__main__":

    ## System inputs
    rerun_all = True
    # User-specified date range, or default to full simulation period
    start_date = sys.argv[1] if len(sys.argv) > 1 else '1999-06-01' 
    end_date = sys.argv[2] if len(sys.argv) > 2 else '2010-05-31'

    ## Load data    
    # Load Pywr-DRB simulation models
    print('Retrieving simulation data.')
    pywr_models = ['obs_pub', 'nhmv10', 'nwmv21', 'nwmv21_withLakes', 'WEAP_23Aug2022_gridmet_nhmv10']
    res_releases = {}
    major_flows = {}
    
    for model in pywr_models:
        res_releases[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'res_release').loc[start_date:end_date,:]
        major_flows[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'major_flow').loc[start_date:end_date,:]
    pywr_models = [f'pywr_{m}' for m in pywr_models]

    # Load base (non-pywr) models
    base_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet']
    datetime_index = list(res_releases.values())[0].index
    for model in base_models:
        res_releases[model] = get_base_results(input_dir, model, datetime_index, 'res_release').loc[start_date:end_date,:]
        major_flows[model] = get_base_results(input_dir, model, datetime_index, 'major_flow').loc[start_date:end_date,:]

    # Verify that all datasets have same datetime index
    for r in res_releases.values():
        assert ((r.index == datetime_index).mean() == 1)
    for r in major_flows.values():
        assert ((r.index == datetime_index).mean() == 1)
    print(f'Successfully loaded {len(base_models)} base model results & {len(pywr_models)} pywr model results')

    ## 3-part flow figures with releases
    if rerun_all:
        print('Plotting 3-part flows at reservoirs.')
        ### nhm only - slides 36-39 in 10/24/2022 presentation
        plot_3part_flows(res_releases, ['nhmv10'], 'pepacton')
        ### nhm vs nwm - slides 40-42 in 10/24/2022 presentation
        plot_3part_flows(res_releases, ['nhmv10', 'nwmv21'], 'pepacton')
        ### nhm vs weap (with nhm backup) - slides 60-62 in 10/24/2022 presentation
        plot_3part_flows(res_releases, ['nhmv10', 'WEAP_23Aug2022_gridmet'], 'pepacton')
        ### nhm vs pywr-nhm - slides 60-62 in 10/24/2022 presentation
        plot_3part_flows(res_releases, ['nhmv10', 'pywr_nhmv10'], 'pepacton')
        ## nwm vs pywr-nwm
        plot_3part_flows(res_releases, ['nwmv21', 'pywr_nwmv21'], 'pepacton')
        ## pywr-nwm vs pywr-nwm_withLakes
        plot_3part_flows(res_releases, ['pywr_nwmv21', 'pywr_nwmv21_withLakes'], 'pepacton')
        ## obs-pub only
        plot_3part_flows(res_releases, ['obs_pub'], 'pepacton')
        plot_3part_flows(res_releases, ['pywr_obs_pub'], 'pepacton')
        plot_3part_flows(res_releases, ['obs_pub', 'nhmv10'], 'pepacton')
        plot_3part_flows(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'pepacton')

        plot_3part_flows(res_releases, ['obs_pub', 'nhmv10'], 'cannonsville')
        plot_3part_flows(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'cannonsville')
        plot_3part_flows(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'neversink')


    if rerun_all:
        print('Plotting weekly flow distributions at reservoirs.')
        ### nhm only - slides 36-39 in 10/24/2022 presentation
        plot_weekly_flow_distributions(res_releases, ['nhmv10'], 'pepacton')
        ### nhm vs nwm - slides 35-37 in 10/24/2022 presentation
        plot_weekly_flow_distributions(res_releases, ['nhmv10', 'nwmv21'], 'pepacton')
        ### nhm vs weap (with nhm backup) - slides 68 in 10/24/2022 presentation
        plot_weekly_flow_distributions(res_releases, ['nhmv10', 'WEAP_23Aug2022_gridmet'], 'pepacton')
        ### nhm vs pywr-nhm - slides 68 in 10/24/2022 presentation
        plot_weekly_flow_distributions(res_releases, ['nhmv10', 'pywr_nhmv10'], 'pepacton')
        ## nwm vs pywr-nwm
        plot_weekly_flow_distributions(res_releases, ['nwmv21', 'pywr_nwmv21'], 'pepacton')
        ## pywr-nwm vs pywr-nwm_withLakes
        plot_weekly_flow_distributions(res_releases, ['pywr_nwmv21', 'pywr_nwmv21_withLakes'], 'pepacton')
        ## obs_pub
        plot_weekly_flow_distributions(res_releases, ['obs_pub'], 'pepacton')
        plot_weekly_flow_distributions(res_releases, ['nhmv10', 'obs_pub'], 'pepacton')
        plot_weekly_flow_distributions(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'pepacton')
        plot_weekly_flow_distributions(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'cannonsville')
        plot_weekly_flow_distributions(res_releases, ['pywr_obs_pub', 'pywr_nhmv10'], 'neversink')
            
        

    ### compile error metrics across models/nodes/metrics
    nodes = ['cannonsville', 'pepacton', 'neversink', 'assunpink', 'beltzvilleCombined', 'blueMarsh']
    radial_models = ['nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
    radial_models = radial_models[::-1]

    if rerun_all:
        print('Plotting radial figures for reservoir releases')

        res_release_metrics = get_error_metrics(res_releases, radial_models, nodes)        
        ### nhm vs nwm only, pepacton only - slides 48-54 in 10/24/2022 presentation
        #plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = False, usepywr = False)
        ### nhm vs nwm only, all reservoirs - slides 55-58 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = False)
        ### nhm vs nwm vs weap only, pepaction only - slides 69 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = False)
        ### nhm vs nwm vs weap only, all reservoirs - slides 70 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = False)
        ### all models, pepaction only - slides 72-73 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = True)
        ### all models, all reservoirs - slides 74-75 in 10/24/2022 presentation
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True)

        ### all models, but using nwm_withLakes for pywr comparison. all reservoirs - slides 74-75 in 10/24/2022 presentation
        radial_models = ['nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_nhmv10', 'pywr_nwmv21_withLakes', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
        radial_models = radial_models[::-1]
        res_release_metrics = get_error_metrics(res_releases, radial_models, nodes)
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True)

        ## obs_pub
        radial_models = ['obs_pub', 'nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21_withLakes', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
        radial_models = radial_models[::-1]
        res_release_metrics = get_error_metrics(res_releases, radial_models, nodes)
        plot_radial_error_metrics(res_release_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True)


    ### now do figs for major flow locations
    nodes = ['delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill']#, 'outletChristina', 'delLordville']
    if rerun_all:
        print('Plotting radial error metrics for major flows.')
        major_flow_metrics = get_error_metrics(major_flows, radial_models, nodes)
        plot_radial_error_metrics(major_flow_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True, usemajorflows=True)


    ### flow comparisons for major flow nodes
    if rerun_all:
        print('Plotting 3-part flows at major nodes.')
        plot_3part_flows(major_flows, ['nhmv10', 'nwmv21'], 'delMontague')
        plot_3part_flows(major_flows, ['nhmv10', 'nwmv21'], 'delTrenton')
        plot_3part_flows(major_flows, ['nhmv10', 'nwmv21'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delMontague')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delTrenton')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_nhmv10'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21_withLakes'], 'delMontague')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21_withLakes'], 'delTrenton')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21_withLakes'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delMontague')
        plot_3part_flows(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delTrenton')
        plot_3part_flows(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delMontague')
        plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delTrenton')
        plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'outletSchuylkill')
        plot_3part_flows(major_flows, ['pywr_obs_pub', 'pywr_nhmv10'], 'delTrenton')
        plot_3part_flows(major_flows, ['pywr_obs_pub', 'pywr_nhmv10'], 'delMontague')
        plot_3part_flows(major_flows, ['nhmv10', 'pywr_obs_pub'], 'delMontague')

        ### weekly flow comparison for major flow nodes
        print('Plotting weekly flow distributions at major nodes.')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'nwmv21'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'nwmv21'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'nwmv21'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'pywr_nhmv10'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['nhmv10', 'pywr_nhmv10'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21_withLakes'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21_withLakes'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21_withLakes'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['WEAP_23Aug2022_gridmet', 'pywr_WEAP_23Aug2022_gridmet_nhmv10'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['pywr_obs_pub','pywr_nhmv10'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['pywr_obs_pub','pywr_nhmv10'], 'delTrenton')

    ## RRV metrics
    if rerun_all:
        print('Plotting RRV metrics.')
        rrv_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'WEAP_23Aug2022_gridmet', 'pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21_withLakes', 'pywr_WEAP_23Aug2022_gridmet_nhmv10']
        nodes = ['delMontague','delTrenton']
        rrv_metrics = get_RRV_metrics(major_flows, rrv_models, nodes)
        plot_rrv_metrics(rrv_metrics, rrv_models, nodes)

    ## Plot flow contributions at Trenton
    if rerun_all:
        print('Plotting flow contributions at major nodes.')
        
        node = 'delTrenton'
        models = ['pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21']
        for model in models:  
            plot_flow_contributions(res_releases, major_flows, model, node,
                                    separate_pub_contributions = False,
                                    percentage_flow = True,
                                    plot_target = False)
            plot_flow_contributions(res_releases, major_flows, model, node,
                                    separate_pub_contributions = False,
                                    percentage_flow = False,
                                    plot_target = True)
    ## Plot inflow comparison
    inflows = {}
    inflow_comparison_models = ['obs_pub', 'nhmv10', 'nwmv21']
    for model in inflow_comparison_models:
        inflows[model] = get_pywr_results(output_dir, model, results_set='inflow')
    compare_inflow_data(inflows, nodes = reservoir_list)
        
    print(f'Done! Check the {fig_dir} folder.')