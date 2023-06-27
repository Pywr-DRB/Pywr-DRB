import numpy as np
import pandas as pd
import sys

from plotting.plotting_functions import *
from utils.lists import reservoir_list, majorflow_list, reservoir_link_pairs
from utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from utils.directories import input_dir, output_dir, fig_dir

from data_processing.get_results import get_base_results, get_pywr_results

### I was having trouble with interactive console plotting in Pycharm for some reason - comment this out if you want to use that and not having issues
#mpl.use('TkAgg')



## Execution - Generate all figures
if __name__ == "__main__":

    ## System inputs
    rerun_all = False
    use_WEAP = False

    ### User-specified date range, or default to minimum overlapping period across models
    if use_WEAP:
        start_date = sys.argv[1] if len(sys.argv) > 1 else '1999-06-01'
        end_date = sys.argv[2] if len(sys.argv) > 2 else '2010-05-31'
    else:
        start_date = sys.argv[1] if len(sys.argv) > 1 else '1984-01-01'
        end_date = sys.argv[2] if len(sys.argv) > 2 else '2017-01-01'

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    ## Load data    
    # Load Pywr-DRB simulation models
    print(f'Retrieving simulation data from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}.')
    if use_WEAP:
        pywr_models = ['obs_pub', 'nhmv10', 'nwmv21', 'WEAP_24Apr2023_gridmet']
    else:
        pywr_models = ['obs_pub', 'nhmv10', 'nwmv21']

    reservoir_downstream_gages = {}
    major_flows = {}
    storages = {}
    reservoir_releases = {}

    for model in pywr_models:
        reservoir_downstream_gages[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'reservoir_downstream_gage').loc[start_date:end_date,:]
        major_flows[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'major_flow').loc[start_date:end_date,:]
        storages[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'res_storage')
        reservoir_releases[f'pywr_{model}'] = get_pywr_results(output_dir, model, 'res_release').loc[start_date:end_date,:]
    pywr_models = [f'pywr_{m}' for m in pywr_models]


    # Load base (non-pywr) models
    if use_WEAP:
        base_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'WEAP_24Apr2023_gridmet']
    else:
        base_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21']

    datetime_index = list(reservoir_downstream_gages.values())[0].index
    for model in base_models:
        reservoir_downstream_gages[model] = get_base_results(input_dir, model, datetime_index, 'reservoir_downstream_gage').loc[start_date:end_date,:]
        major_flows[model] = get_base_results(input_dir, model, datetime_index, 'major_flow').loc[start_date:end_date,:]

    # Verify that all datasets have same datetime index
    for r in reservoir_downstream_gages.values():
        # print(f'len r: {len(r.index)} and dt: {len(datetime_index)}')
        assert ((r.index == datetime_index).mean() == 1)
    for r in major_flows.values():
        # print(f'len r: {len(r.index)} and dt: {len(datetime_index)}')
        assert ((r.index == datetime_index).mean() == 1)
    print(f'Successfully loaded {len(base_models)} base model results & {len(pywr_models)} pywr model results')

    ## 3-part flow figures with releases
    if rerun_all:
        print('Plotting 3-part flows at reservoirs.')
        ### nhm only - slides 36-39 in 10/24/2022 presentation
        plot_3part_flows(reservoir_downstream_gages, ['nhmv10'], 'pepacton')
        ### nhm vs nwm - slides 40-42 in 10/24/2022 presentation
        plot_3part_flows(reservoir_downstream_gages, ['nhmv10', 'nwmv21'], 'pepacton')
        if use_WEAP:
            ### nhm vs weap (with nhm backup) - slides 60-62 in 10/24/2022 presentation
            plot_3part_flows(reservoir_downstream_gages, ['nhmv10', 'WEAP_24Apr2023_gridmet'], 'pepacton')
        ### nhm vs pywr-nhm - slides 60-62 in 10/24/2022 presentation
        plot_3part_flows(reservoir_downstream_gages, ['nhmv10', 'pywr_nhmv10'], 'pepacton')
        ## obs-pub only
        plot_3part_flows(reservoir_downstream_gages, ['obs_pub'], 'pepacton')
        plot_3part_flows(reservoir_downstream_gages, ['pywr_obs_pub'], 'pepacton')
        plot_3part_flows(reservoir_downstream_gages, ['obs_pub', 'nhmv10'], 'pepacton')
        plot_3part_flows(reservoir_downstream_gages, ['pywr_obs_pub', 'pywr_nhmv10'], 'pepacton')

        plot_3part_flows(reservoir_downstream_gages, ['obs_pub', 'nhmv10'], 'cannonsville')
        plot_3part_flows(reservoir_downstream_gages, ['pywr_obs_pub', 'pywr_nhmv10'], 'cannonsville')
        plot_3part_flows(reservoir_downstream_gages, ['pywr_obs_pub', 'pywr_nhmv10'], 'neversink')


    if rerun_all:
        print('Plotting weekly flow distributions at reservoirs.')
        ### nhm only - slides 36-39 in 10/24/2022 presentation
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['nhmv10'], 'pepacton')
        ### nhm vs nwm - slides 35-37 in 10/24/2022 presentation
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['nhmv10', 'nwmv21'], 'pepacton')
        if use_WEAP:
            ### nhm vs weap (with nhm backup) - slides 68 in 10/24/2022 presentation
            plot_weekly_flow_distributions(reservoir_downstream_gages, ['nhmv10', 'WEAP_24Apr2023_gridmet'], 'pepacton')
        ### nhm vs pywr-nhm - slides 68 in 10/24/2022 presentation
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['nhmv10', 'pywr_nhmv10'], 'pepacton')

        ## obs_pub
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['obs_pub'], 'pepacton')
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['nhmv10', 'obs_pub'], 'pepacton')
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['pywr_obs_pub', 'pywr_nhmv10'], 'pepacton')
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['pywr_obs_pub', 'pywr_nhmv10'], 'cannonsville')
        plot_weekly_flow_distributions(reservoir_downstream_gages, ['pywr_obs_pub', 'pywr_nhmv10'], 'neversink')

        


    nodes = ['cannonsville', 'pepacton', 'neversink', 'fewalter', 'beltzvilleCombined', 'blueMarsh']
    if use_WEAP:
        radial_models = ['nhmv10', 'nwmv21', 'WEAP_24Apr2023_gridmet', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_WEAP_24Apr2023_gridmet_nhmv10']
    else:
        radial_models = ['nhmv10', 'nwmv21', 'obs_pub', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_obs_pub']
    radial_models = radial_models[::-1]

    ### compile error metrics across models/nodes/metrics
    if rerun_all:

        print('Plotting radial figures for reservoir releases')

        reservoir_downstream_gage_metrics = get_error_metrics(reservoir_downstream_gages, radial_models, nodes)
        ### nhm vs nwm only, pepacton only - slides 48-54 in 10/24/2022 presentation
        #plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = False, useweap = False, usepywr = False)
        ### nhm vs nwm only, all reservoirs - slides 55-58 in 10/24/2022 presentation
        plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = False)
        ### nhm vs nwm vs weap only, pepaction only - slides 69 in 10/24/2022 presentation
        plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = False)
        ### nhm vs nwm vs weap only, all reservoirs - slides 70 in 10/24/2022 presentation
        plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = False)
        ### all models, pepaction only - slides 72-73 in 10/24/2022 presentation
        plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = False, useweap = True, usepywr = True)
        ### all models, all reservoirs - slides 74-75 in 10/24/2022 presentation
        plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = True, useweap = True, usepywr = True)



    ### now do figs for major flow locations
    if rerun_all:
        print('Plotting radial error metrics for major flows.')
        nodes = ['delMontague', 'delTrenton', 'outletSchuylkill']  #  'delLordville']
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
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delMontague')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delTrenton')
        plot_3part_flows(major_flows, ['nwmv21', 'pywr_nwmv21'], 'outletSchuylkill')
        if use_WEAP:
            plot_3part_flows(major_flows, ['WEAP_24Apr2023_gridmet', 'pywr_WEAP_24Apr2023_gridmet'], 'delMontague')
            plot_3part_flows(major_flows, ['WEAP_24Apr2023_gridmet', 'pywr_WEAP_24Apr2023_gridmet'], 'delTrenton')
            plot_3part_flows(major_flows, ['WEAP_24Apr2023_gridmet', 'pywr_WEAP_24Apr2023_gridmet'], 'outletSchuylkill')
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
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['nwmv21', 'pywr_nwmv21'], 'outletSchuylkill')
        if use_WEAP:
            plot_weekly_flow_distributions(major_flows, ['WEAP_24Apr2023_gridmet', 'pywr_WEAP_24Apr2023_gridmet'], 'delMontague')
            plot_weekly_flow_distributions(major_flows, ['WEAP_24Apr2023_gridmet', 'pywr_WEAP_24Apr2023_gridmet'], 'delTrenton')
            plot_weekly_flow_distributions(major_flows, ['WEAP_24Apr2023_gridmet', 'pywr_WEAP_24Apr2023_gridmet'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'delTrenton')
        plot_weekly_flow_distributions(major_flows, ['obs_pub', 'pywr_obs_pub'], 'outletSchuylkill')
        plot_weekly_flow_distributions(major_flows, ['pywr_obs_pub','pywr_nhmv10'], 'delMontague')
        plot_weekly_flow_distributions(major_flows, ['pywr_obs_pub','pywr_nhmv10'], 'delTrenton')

    ## RRV metrics
    if rerun_all:
        print('Plotting RRV metrics.')
        if use_WEAP:
            rrv_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'WEAP_24Apr2023_gridmet', 'pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21', 'pywr_WEAP_24Apr2023_gridmet']
        else:
            rrv_models = ['obs', 'obs_pub', 'nhmv10', 'nwmv21', 'pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21']

        nodes = ['delMontague','delTrenton']
        rrv_metrics = get_RRV_metrics(major_flows, rrv_models, nodes)
        plot_rrv_metrics(rrv_metrics, rrv_models, nodes)

    ## Plot flow contributions at Trenton
    if rerun_all:
        print('Plotting flow contributions at major nodes.')

        node = 'delTrenton'
        models = ['pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21']
        for model in models:  
            plot_flow_contributions(reservoir_releases, major_flows, model, node,
                                    start_date= '2000-01-01',
                                    end_date= '2004-01-01',
                                    percentage_flow = False,
                                    plot_target = True)

        # Only plot percentage for obs-pub
        plot_flow_contributions(reservoir_releases, major_flows, 'pywr_obs_pub', node,
                                start_date= '2000-01-01',
                                end_date= '2004-01-01',
                                percentage_flow = True,
                                plot_target = False)
    node = 'delTrenton'
    models = ['pywr_obs_pub', 'pywr_nhmv10', 'pywr_nwmv21', 'obs']
    for model in models:      
        plot_flow_contributions(reservoir_downstream_gages, major_flows, model, node,
                                start_date= '2000-01-01',
                                end_date= '2004-01-01',
                                percentage_flow = False,
                                plot_target = False)

        
        plot_flow_contributions(reservoir_downstream_gages, major_flows, model, node,
                                start_date= '2000-01-01',
                                end_date= '2004-01-01',
                                percentage_flow = True,
                                plot_target = False)

    ## Plot inflow comparison
    if rerun_all:
        inflows = {}
        inflow_comparison_models = ['obs_pub', 'nhmv10', 'nwmv21']
        for model in inflow_comparison_models:
            inflows[model] = get_pywr_results(output_dir, model, results_set='inflow')
        compare_inflow_data(inflows, nodes = reservoir_list)


    ### plot NYC reservoir comparison
    if rerun_all:
        print('Plotting NYC reservoir operations')
        plot_combined_nyc_storage(storages, reservoir_downstream_gages, pywr_models, start_date='2000-01-01', end_date='2004-01-01')
        plot_combined_nyc_storage(storages, reservoir_downstream_gages, pywr_models, start_date='2000-01-01', end_date='2010-01-01')
        plot_combined_nyc_storage(storages, reservoir_downstream_gages, pywr_models, start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))

    print(f'Done! Check the {fig_dir} folder.')