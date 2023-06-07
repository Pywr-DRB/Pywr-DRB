import numpy as np
import pandas as pd
import sys

from plotting.plotting_functions import *
from utils.lists import reservoir_list, majorflow_list, reservoir_link_pairs
from utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from utils.directories import input_dir, output_dir, fig_dir

from data_processing.get_results import get_base_results, get_pywr_results

fig_dir = fig_dir+'full_reconstruction/'

## Execution - Generate all figures
if __name__ == "__main__":

    ## System inputs
    start_date = '1955-01-01'
    end_date = '2022-12-31'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    ## Load data    
    # Load Pywr-DRB simulation models
    print(f'Retrieving simulation data from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}.')
    pywr_models = ['obs_pub']

    reservoir_downstream_gages = {}
    major_flows = {}
    storages = {}
    reservoir_releases = {}

    for model in pywr_models:
        reservoir_downstream_gages[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'reservoir_downstream_gage').loc[start_date:end_date,:]
        major_flows[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'major_flow').loc[start_date:end_date,:]
        storages[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'res_storage')
        reservoir_releases[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'res_releases').loc[start_date:end_date,:]
    pywr_models = [f'pywr_{m}' for m in pywr_models]

    # Load base (non-pywr) models
    base_models = ['obs_pub', 'obs']

    datetime_index = pd.to_datetime(reservoir_downstream_gages['pywr_obs_pub'].index)
    for model in base_models:
        reservoir_downstream_gages[model] = get_base_results(input_dir, model, datetime_index, 'reservoir_downstream_gage') #.loc[start_date:end_date,:]
        major_flows[model] = get_base_results(input_dir, model, datetime_index, 'major_flow').loc[start_date:end_date,:]
        print(f"loaded {model} base data")

    # Verify that all datasets have same datetime index
    for r in reservoir_downstream_gages.values():
        # print(f'len r: {len(r.index)} and dt: {len(datetime_index)}')
        assert ((r.index == datetime_index).mean() == 1)
    for r in major_flows.values():
        assert ((r.index == datetime_index).mean() == 1)
    print(f'Successfully loaded {len(base_models)} base model results & {len(pywr_models)} pywr model results')

    ## 3-part flow figures with releases
    print('Plotting 3-part flows at reservoirs.')

    plot_3part_flows(reservoir_downstream_gages, ['obs_pub'], 'pepacton', fig_dir = fig_dir)
    plot_3part_flows(reservoir_downstream_gages, ['pywr_obs_pub'], 'pepacton', fig_dir = fig_dir)
    plot_3part_flows(reservoir_downstream_gages, ['obs_pub', 'pywr_obs_pub'], 'pepacton', fig_dir = fig_dir)

    plot_3part_flows(reservoir_downstream_gages, ['obs_pub', 'pywr_obs_pub'], 'cannonsville', fig_dir = fig_dir)
    plot_3part_flows(reservoir_downstream_gages, ['obs_pub', 'pywr_obs_pub'], 'neversink', fig_dir = fig_dir)


    print('Plotting weekly flow distributions at reservoirs.')
    plot_weekly_flow_distributions(reservoir_downstream_gages, ['obs_pub', 'pywr_obs_pub'], 'pepacton', fig_dir = fig_dir)
    plot_weekly_flow_distributions(reservoir_downstream_gages, ['obs_pub', 'pywr_obs_pub'], 'cannonsville', fig_dir = fig_dir)
    plot_weekly_flow_distributions(reservoir_downstream_gages, ['obs_pub', 'pywr_obs_pub'], 'neversink', fig_dir = fig_dir)


    nodes = ['cannonsville', 'pepacton', 'neversink', 'fewalter', 'beltzvilleCombined', 'blueMarsh']
    radial_models = ['obs_pub', 'pywr_obs_pub']
    radial_models = radial_models[::-1]

    ### compile error metrics across models/nodes/metrics
    print('Plotting radial figures for reservoir releases')

    reservoir_downstream_gage_metrics = get_error_metrics(reservoir_downstream_gages, radial_models, nodes)
    plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = True, fig_dir = fig_dir)


    ### now do figs for major flow locations
    print('Plotting radial error metrics for major flows.')
    nodes = ['delMontague', 'delTrenton', 'outletSchuylkill']  # , 'outletChristina', 'delLordville']
    major_flow_metrics = get_error_metrics(major_flows, radial_models, nodes)
    plot_radial_error_metrics(major_flow_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = True, usemajorflows=True, fig_dir = fig_dir)


    ### flow comparisons for major flow nodes
    plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delMontague', fig_dir = fig_dir)
    plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delTrenton', fig_dir = fig_dir)
    plot_3part_flows(major_flows, ['obs_pub', 'pywr_obs_pub'], 'outletSchuylkill', fig_dir = fig_dir)
    plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delMontague', fig_dir = fig_dir)
    plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delTrenton', fig_dir = fig_dir)
    plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'outletSchuylkill', fig_dir = fig_dir)

    ## RRV metrics
    print('Plotting RRV metrics.')
    rrv_models = ['obs', 'obs_pub', 'pywr_obs_pub']
    nodes = ['delMontague','delTrenton']
    rrv_metrics = get_RRV_metrics(major_flows, rrv_models, nodes)
    plot_rrv_metrics(rrv_metrics, rrv_models, nodes, fig_dir = fig_dir)

    ## Plot flow contributions at Trenton
    print('Plotting NYC reservoir operations')
    plot_combined_nyc_storage(storages, reservoir_downstream_gages, pywr_models, start_date=start_date, end_date=end_date, fig_dir = fig_dir)
    plot_combined_nyc_storage(storages, reservoir_downstream_gages, pywr_models, start_date='1962-01-01', end_date='1970-01-01', fig_dir = fig_dir)

    print(f'Done! Check the {fig_dir} folder.')