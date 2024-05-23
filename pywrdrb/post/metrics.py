"""
Contains functions for calculating metrics.

Includes:
- calculate_reliability
- calculate_vulnerability
"""
import numpy as np
import pandas as pd
from pywrdrb.pywr_drb_node_data import downstream_node_lags, immediate_downstream_nodes_dict
from pywrdrb.utils.lists import majorflow_list
from pywrdrb.utils.timeseries import subset_timeseries

def calculate_reliability(flow, target):
    """
    Function used to get reliability of streamflows 
    at a specific node given MRF target and simulated flows. 

    Reliability is the fraction of time that the simulated flow
    is above the MRF target.
    
    Args:
        flow (pd.Series or np.array): simulated flows
        target (pd.Series or np.array): MRF target flows
        
    Returns:
        reliability (float): fraction of time flow is above target
    """
    reliability = np.sum(flow > target) / len(flow)
    return reliability


def calculate_vulnerability(flow, target):
    """
    Function used to get vulnerability of streamflows 
    at a specific node given MRF target and simulated flows. 

    Vulnerability is the magnitude of the largest flow deviation below the MRF target.
    
    Args:
        flow (pd.Series or np.array): simulated flows
        target (pd.Series or np.array): MRF target flows
        
    Returns:
        vulnerability (float): magnitude of largest shortage
    """
    vulnerability = np.max(target - flow)
    vulnerability = np.max([0, vulnerability])
    vulnerability = np.abs(vulnerability)
    return vulnerability


def get_lagged_lower_basin_contributions(lower_basin_mrf_contributions, 
                                         model, realization, 
                                         start_date=None, end_date=None,
                                         downstream_node_lags=downstream_node_lags, 
                                         immediate_downstream_nodes_dict=immediate_downstream_nodes_dict):
                    
    if start_date and end_date:
        lower_basin_mrf = subset_timeseries(lower_basin_mrf_contributions[model][realization],
                                                        start_date, end_date)
    else:
        lower_basin_mrf = lower_basin_mrf_contributions[model][realization]
    lower_basin_mrf.columns = [c.split('_')[-1] for c in lower_basin_mrf.columns]
    
    # acct for lag at blue marsh so it can be added to trenton equiv flow.
    for c in lower_basin_mrf.columns:
        lag = downstream_node_lags[c]
        downstream_node = immediate_downstream_nodes_dict[c]
        while downstream_node != 'output_del':
            lag += downstream_node_lags[downstream_node]
            downstream_node = immediate_downstream_nodes_dict[downstream_node]
        if lag > 0:
            lag_start = lower_basin_mrf.index[lag]
            lag_end = lower_basin_mrf.index[-lag]
            lower_basin_mrf.loc[lag_start:, c] = lower_basin_mrf.loc[:lag_end, c]
    lagged_lower_basin_mrf_contributions = lower_basin_mrf
    return lagged_lower_basin_mrf_contributions


def add_blueMarsh_mrf_contribution_to_delTrenton(major_flows, lower_basin_mrf_contributions,
                                                 model, realization, 
                                                 immediate_downstream_nodes_dict=immediate_downstream_nodes_dict,
                                                 downstream_node_lags=downstream_node_lags):
    """
    Adds Blue Marsh extra-release contributions to the flow at delTrenton; 
    used to calculate MRF flow violations.
    
    Args:
        major_flows (dict): dictionary of major flows for each realization
        lower_basin_mrf_contributions (dict): dictionary of lower_basin_mrf_constributions for each realization
        node (str): node of interest
        immediate_downstream_nodes_dict (dict): dictionary of immediate downstream nodes
        downstream_node_lags (dict): dictionary of lags for downstream nodes
        
    Returns:
    """
    trenton_flow = major_flows[model][realization]['delTrenton'].copy()
    lagged_lower_basin_contributions = get_lagged_lower_basin_contributions(lower_basin_mrf_contributions, model, realization)
    trenton_flow += lagged_lower_basin_contributions['blueMarsh']
    return trenton_flow


def get_ensemble_hashimoto_metrics(major_flows, mrf_targets, 
                                   model,node,
                                   lower_basin_mrf_contributions=None):
    """
    Function used to calculate reliability and vulnerability for each realization
    in the ensemble. 
    """
    if node == 'delTrenton':
        err_msg = 'Need to consider blueMarsh excess releases for system performance at Trenton.'
        err_msg += '\nCannot have lower_basin_mrf_contributions=None.'
        assert(lower_basin_mrf_contributions is not None), err_msg
    
    reliability = []
    vulnerability = []
    for realization in major_flows[model].keys():
        if node == 'delTrenton':
            sim_flow = add_blueMarsh_mrf_contribution_to_delTrenton(major_flows, 
                                                                    lower_basin_mrf_contributions,
                                                                    model, realization)
        else:
            sim_flow = major_flows[model][realization][node]
            
        target_label = f'mrf_target_{node}' if 'mrf_target' not in node else node
        sim_target = mrf_targets[model][realization][target_label]
        reliability.append(calculate_reliability(sim_flow, sim_target))
        vulnerability.append(calculate_vulnerability(sim_flow, sim_target))
    
    return np.array(reliability), np.array(vulnerability)



def change_results_dict_to_ensemble(results_dict):
    """
    Takes a dict of structure results_dict[model] = pd.DataFrame and 
    changes it to have structure results_dict[model]['0'] = pd.DataFrame. 
    """
    models = results_dict.keys()
    new_results_dict = {}
    for m in models:
        # If results_dict is non-ensemble; add ensemble with realization 0
        if (type(results_dict[m]) == pd.DataFrame):
            model_realization_data = results_dict[m]
            new_results_dict[m] = {}
            new_results_dict[m][0] = model_realization_data
        else:
            new_results_dict[m] = results_dict[m]
    return new_results_dict


def get_shortfall_metrics(major_flows, lower_basin_mrf_contributions, 
                          mrf_targets, 
                          ibt_demands, ibt_diversions, 
                          models_mrf, models_ibt, nodes,
                          shortfall_threshold=0.95, shortfall_break_length=7, units='MG',
                          start_date=None, end_date=None):
    """

    """
    units_daily = 'BGD' if units == 'MG' else 'MCM/D' 
    eps = 1e-9

    # Check and modify all results dicts to have ensemble structure
    # results_dict[model][realization] = pd.DataFrame
    major_flows = change_results_dict_to_ensemble(major_flows)
    lower_basin_mrf_contributions = change_results_dict_to_ensemble(lower_basin_mrf_contributions)
    mrf_targets = change_results_dict_to_ensemble(mrf_targets)
    ibt_demands = change_results_dict_to_ensemble(ibt_demands)
    ibt_diversions = change_results_dict_to_ensemble(ibt_diversions)

    resultsdict = {}
    for j, node in enumerate(nodes):
        resultsdict[node] = {}
        models = models_mrf if j<2 else models_ibt
        for i, m in enumerate(models):
            resultsdict[node][m] = {}
            
            ### Storage lists for model-specific shortfalls
            # Contains individual events & get event-specific metrics
            ensemble_reliabilities = []
            ensemble_resiliencies = []
            durations = []          # length of each event
            intensities = []        # intensity of each event = avg deficit within event
            severities = []         # severity = duration * intensity
            vulnerabilities = []    # vulnerability = max daily deficit within event
            magnitudes = []         # magnitude of each event = total accumulated deficit within event
            event_starts = []       # define event to start with nonzero shortfall
            event_ends = []         # end with the next shortfall date that preceeds shortfall_break_length non-shortfall dates.
            realization_ids = []    # realization number or key
            
            # Loop through realizations (all models have atleast realization '0')    
            realization_keys = list(major_flows[m].keys())
            n_realizations = len(realization_keys)
            for r in realization_keys:
                if node in majorflow_list:
                    flows = subset_timeseries(major_flows[m][r][node], start_date, end_date)

                    ### for Trenton & Pywr models
                    # include BLue Marsh FFMP releases in Trenton Equiv flow
                    if 'pywr' in m:
                        lagged_lower_basin_contributions = get_lagged_lower_basin_contributions(lower_basin_mrf_contributions,
                                                                                    model=m,
                                                                                    realization=r,
                                                                                    start_date=start_date,
                                                                                    end_date=end_date)
                        flows += lagged_lower_basin_contributions['blueMarsh']
                    
                    dates = flows.index
                    flows = flows.values
                    target_label = f'mrf_target_{node}' if 'mrf_target' not in node else node
                    thresholds = np.ones(len(flows)) * mrf_targets[models_ibt[0]][r][target_label].max() * \
                                shortfall_threshold - eps
                    # print(f'{node} normal minimum flow target: {thresholds[0]} {units_daily}')
                else:
                    flows = subset_timeseries(ibt_diversions[m][r][f'delivery_{node}'], start_date, end_date)
                    dates = flows.index
                    flows = flows.values
                    thresholds = subset_timeseries(ibt_demands[m][r][f'demand_{node}'], start_date, end_date).values * \
                                shortfall_threshold - eps

                ### reliability is the fraction of time steps above threshold
                reliability = (flows > thresholds).mean()
                ### resiliency is the probability of recovering to above threshold if currently under threshold
                if reliability < 1 - eps:
                    resiliency = np.logical_and(flows[:-1] < thresholds[:-1], \
                                                (flows[1:] >= thresholds[1:])).mean() / (1 - reliability)
                else:
                    resiliency = np.nan

                ### convert to percents
                if n_realizations == 1: 
                    # This maintains old behavior (for diagnostics paper)
                    resultsdict[node][m]['reliability'] = reliability * 100
                    resultsdict[node][m]['resiliency'] = resiliency * 100
                
                ensemble_reliabilities.append(reliability * 100)
                ensemble_resiliencies.append(resiliency * 100)
                
                if reliability > eps and reliability < 1 - eps:
                    duration = 0
                    severity = 0
                    vulnerability = 0
                    magnitude = 0
                    in_event = False
                    for i in range(len(flows)):
                        v = flows[i]
                        t = thresholds[i]
                        d = dates[i]
                        if in_event or v < t:
                            ### is this the start of a new event?
                            if not in_event:
                                event_starts.append(d)
                            ### if this is part of event, we add to metrics whether today is deficit or not
                            duration += 1
                            s = max(t - v, 0)
                            severity += s
                            vulnerability = max(vulnerability, s)
                            ### now check if next shortfall_break_length days include any deficits. if not, end event.
                            in_event = np.any(flows[i+1: i+1+shortfall_break_length] < \
                                            thresholds[i+1: i+1+shortfall_break_length])
                            if not in_event:
                                event_ends.append(dates[min(i+1, len(dates)-1)])
                                durations.append(duration)
                                severities.append(severity)
                                intensities.append(severity / duration)
                                realization_ids.append(r)
                                vulnerabilities.append(vulnerability)
                                in_event = False
                                duration = 0
                                severity = 0
                                vulnerability = 0
            
            resultsdict[node][m]['event_starts'] = event_starts
            resultsdict[node][m]['event_ends'] = event_ends
            resultsdict[node][m]['durations'] = durations
            resultsdict[node][m]['severities'] = severities
            resultsdict[node][m]['intensities'] = intensities
            resultsdict[node][m]['vulnerabilities'] = vulnerabilities
            resultsdict[node][m]['realization_ids'] = realization_ids
            
            if n_realizations > 1:
                resultsdict[node][m]['reliability'] = ensemble_reliabilities
                resultsdict[node][m]['resiliency'] = ensemble_resiliencies
    return resultsdict



# def get_shortfall_metrics(major_flows, lower_basin_mrf_contributions, 
#                           mrf_targets, 
#                           ibt_demands, ibt_diversions, 
#                           models_mrf, models_ibt, nodes,
#                           shortfall_threshold=0.95, shortfall_break_length=7, units='MG',
#                           start_date=None, end_date=None):
#     """

#     """
#     units_daily = 'BGD' if units == 'MG' else 'MCM/D' 
#     eps = 1e-9

#     # Check and modify all results dicts to have ensemble structure
#     # results_dict[model][realization] = pd.DataFrame
#     major_flows = change_results_dict_to_ensemble(major_flows)
#     lower_basin_mrf_contributions = change_results_dict_to_ensemble(lower_basin_mrf_contributions)
#     mrf_targets = change_results_dict_to_ensemble(mrf_targets)
#     ibt_demands = change_results_dict_to_ensemble(ibt_demands)
#     ibt_diversions = change_results_dict_to_ensemble(ibt_diversions)

#     resultsdict = {}
#     for j, node in enumerate(nodes):
#         resultsdict[node] = {}
#         models = models_mrf if j<2 else models_ibt
#         for i, m in enumerate(models):
#             resultsdict[node][m] = {}
            
#             ### Storage lists for model-specific shortfalls
#             # Contains individual events & get event-specific metrics
#             ensemble_reliabilities = []
#             ensemble_resiliencies = []
#             durations = []          # length of each event
#             intensities = []        # intensity of each event = avg deficit within event
#             severities = []         # severity = duration * intensity
#             vulnerabilities = []    # vulnerability = max daily deficit within event
#             event_starts = []       # define event to start with nonzero shortfall
#             event_ends = []         # end with the next shortfall date that preceeds shortfall_break_length non-shortfall dates.
#             realization_ids = []    # realization number or key
            
#             # Loop through realizations (all models have atleast realization '0')    
#             realization_keys = list(major_flows[m].keys())
#             n_realizations = len(realization_keys)
#             for r in realization_keys:
#                 if node in majorflow_list:
#                     flows = subset_timeseries(major_flows[m][r][node], start_date, end_date)

#                     ### for Trenton & Pywr models
#                     # include BLue Marsh FFMP releases in Trenton Equiv flow
#                     if 'pywr' in m:
#                         lagged_lower_basin_contributions = get_lagged_lower_basin_contributions(lower_basin_mrf_contributions,
#                                                                                     model=m,
#                                                                                     realization=r,
#                                                                                     start_date=start_date,
#                                                                                     end_date=end_date)
#                         flows += lagged_lower_basin_contributions['blueMarsh']
                    
#                     dates = flows.index
#                     flows = flows.values
#                     target_label = f'mrf_target_{node}' if 'mrf_target' not in node else node
#                     thresholds = np.ones(len(flows)) * mrf_targets[models_ibt[0]][r][target_label].max() * \
#                                 shortfall_threshold - eps
#                     # print(f'{node} normal minimum flow target: {thresholds[0]} {units_daily}')
#                 else:
#                     flows = subset_timeseries(ibt_diversions[m][r][f'delivery_{node}'], start_date, end_date)
#                     dates = flows.index
#                     flows = flows.values
#                     thresholds = subset_timeseries(ibt_demands[m][r][f'demand_{node}'], start_date, end_date).values * \
#                                 shortfall_threshold - eps

#                 ### reliability is the fraction of time steps above threshold
#                 reliability = (flows > thresholds).mean()
#                 ### resiliency is the probability of recovering to above threshold if currently under threshold
#                 if reliability < 1 - eps:
#                     resiliency = np.logical_and(flows[:-1] < thresholds[:-1], \
#                                                 (flows[1:] >= thresholds[1:])).mean() / (1 - reliability)
#                 else:
#                     resiliency = np.nan

#                 ### convert to percents
#                 if n_realizations == 1: 
#                     # This maintains old behavior (for diagnostics paper)
#                     resultsdict[node][m]['reliability'] = reliability * 100
#                     resultsdict[node][m]['resiliency'] = resiliency * 100
                
#                 ensemble_reliabilities.append(reliability * 100)
#                 ensemble_resiliencies.append(resiliency * 100)
                
#                 if reliability > eps and reliability < 1 - eps:
#                     duration = 0
#                     severity = 0
#                     vulnerability = 0
#                     in_event = False
#                     for i in range(len(flows)):
#                         v = flows[i]
#                         t = thresholds[i]
#                         d = dates[i]
#                         if in_event or v < t:
#                             ### is this the start of a new event?
#                             if not in_event:
#                                 event_starts.append(d)
#                             ### if this is part of event, we add to metrics whether today is deficit or not
#                             duration += 1
#                             s = max(t - v, 0)
#                             severity += s
#                             vulnerability = max(vulnerability, s)
#                             ### now check if next shortfall_break_length days include any deficits. if not, end event.
#                             in_event = np.any(flows[i+1: i+1+shortfall_break_length] < \
#                                             thresholds[i+1: i+1+shortfall_break_length])
#                             if not in_event:
#                                 event_ends.append(dates[min(i+1, len(dates)-1)])
#                                 durations.append(duration)
#                                 severities.append(severity)
#                                 intensities.append(severity / duration)
#                                 realization_ids.append(r)
#                                 vulnerabilities.append(vulnerability)
#                                 in_event = False
#                                 duration = 0
#                                 severity = 0
#                                 vulnerability = 0
            
#             resultsdict[node][m]['event_starts'] = event_starts
#             resultsdict[node][m]['event_ends'] = event_ends
#             resultsdict[node][m]['durations'] = durations
#             resultsdict[node][m]['severities'] = severities
#             resultsdict[node][m]['intensities'] = intensities
#             resultsdict[node][m]['vulnerabilities'] = vulnerabilities
#             resultsdict[node][m]['realization_ids'] = realization_ids
            
#             if n_realizations > 1:
#                 resultsdict[node][m]['reliability'] = ensemble_reliabilities
#                 resultsdict[node][m]['resiliency'] = ensemble_resiliencies
#     return resultsdict
