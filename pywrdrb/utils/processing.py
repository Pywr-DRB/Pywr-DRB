"""
Contains functions used to process Pywr-DRB data.  

Functions include:
- get_pywr_results()
- get_base_results()


"""

import numpy as np
import pandas as pd

import h5py

from .lists import reservoir_list, majorflow_list, reservoir_link_pairs
from .constants import cms_to_mgd, cfs_to_mgd, cm_to_mg


def get_pywr_results(output_dir, model, results_set='all', scenario=0):
    '''
    Gathers simulation results from Pywr model run and returns a pd.DataFrame.

    :param output_dir:
    :param model:
    :param results_set: can be "all" to return all results,
                            "res_release" to return reservoir releases (downstream gage comparison),
                            "res_storage" to return resrvoir storages,
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
                results[k] = f[k][:, scenario]
            elif results_set == 'res_release':
                ## Need to pull flow data for link_ downstream of reservoirs instead of simulated outflows
                if k.split('_')[0] == 'link' and k.split('_')[1] in reservoir_link_pairs.values():
                    res_name = [res for res, link in reservoir_link_pairs.items() if link == k.split('_')[1]][0]
                    results[res_name] = f[k][:, scenario]
                # Now pull simulated relases from un-observed reservoirs
                elif k.split('_')[0] == 'outflow' and k.split('_')[1] in reservoir_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'res_storage':
                if k.split('_')[0] == 'reservoir' and k.split('_')[1] in reservoir_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'major_flow':
                if k.split('_')[0] == 'link' and k.split('_')[1] in majorflow_list:
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'inflow':
                if k.split('_')[0] == 'catchment':
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'withdrawal':
                if k.split('_')[0] == 'catchmentWithdrawal':
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set == 'consumption':
                if k.split('_')[0] == 'catchmentConsumption':
                    results[k.split('_')[1]] = f[k][:, scenario]
            elif results_set in ('prev_flow_catchmentWithdrawal', 'max_flow_catchmentWithdrawal', 'max_flow_catchmentConsumption'):
                if results_set in k:
                    results[k.split('_')[-1]] = f[k][:, scenario]
            elif results_set in ('res_level'):
                if 'drought_level' in k:
                    results[k.split('_')[-1]] = f[k][:, scenario]
        
        # Format datetime index
        day = [f['time'][i][0] for i in range(len(f['time']))]
        month = [f['time'][i][2] for i in range(len(f['time']))]
        year = [f['time'][i][3] for i in range(len(f['time']))]
        date = [f'{y}-{m}-{d}' for y, m, d in zip(year, month, day)]
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
        reservoirs_with_data = [list(filter(lambda x: reservoir_link_pairs[x] == site, reservoir_link_pairs))[0] for
                                site in available_release_data]
        gage_flow = gage_flow.loc[:, available_release_data]
        gage_flow.columns = reservoirs_with_data
    elif results_set == 'major_flow':
        for c in gage_flow.columns:
            if c not in majorflow_list:
                gage_flow = gage_flow.drop(c, axis=1)
    gage_flow = gage_flow.loc[datetime_index, :]
    return gage_flow


