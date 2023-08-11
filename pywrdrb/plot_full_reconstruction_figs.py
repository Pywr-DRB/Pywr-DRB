import numpy as np
import pandas as pd
import sys

from plotting.plotting_functions import *
from utils.lists import reservoir_list, majorflow_list, reservoir_link_pairs
from utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from utils.directories import input_dir, output_dir, fig_dir

from post.get_results import get_base_results, get_pywr_results

fig_dir = fig_dir+'full_reconstruction/'

## Execution - Generate all figures
if __name__ == "__main__":

    ## System inputs
    start_date = '1950-01-01'
    end_date = '2022-12-31'
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    
    ## Load data    
    # Load Pywr-DRB simulation models
    print(f'Retrieving simulation data from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}.')
    
    pywr_models = ['obs_pub_nhmv10', 'obs_pub_nwmv21',
                   'obs_pub_nhmv10_NYCScaled', 'obs_pub_nwmv21_NYCScaled'] #, 'obs_pub_nhmv10_NYCScaled_ensemble', 'obs_pub_nwmv21_NYCScaled_ensemble',]
    
    base_models = ['obs', 
                   'obs_pub_nhmv10', 'obs_pub_nwmv21',
                   'obs_pub_nhmv10_NYCScaled', 'obs_pub_nwmv21_NYCScaled']
    
    rrv_models = ['obs', 
                  'obs_pub_nhmv10', 'pywr_obs_pub_nhmv10',
                  'obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled']
    
    reservoir_downstream_gages = {}
    major_flows = {}
    storages = {}
    inflows={}
    datetime_index = None

    for model in pywr_models:
        print(f'Retrieving {model} PywrDRB data')
        if 'ensemble' in model:
            reservoir_downstream_gages[f'pywr_{model}'] = {}
            major_flows[f'pywr_{model}'] = {}
            storages[f'pywr_{model}']={}
            n_realizations = 30 if model == 'obs_pub_nhmv10_NYCScaled_ensemble' else 20

            for i in range(n_realizations):
                reservoir_downstream_gages[f'pywr_{model}'][f'realization_{i}'], datetime_index = get_pywr_results(output_dir, model, 'reservoir_downstream_gage', datetime_index=datetime_index, scenario=i)
                reservoir_downstream_gages[f'pywr_{model}'][f'realization_{i}']= reservoir_downstream_gages[f'pywr_{model}'][f'realization_{i}'].loc[start_date:end_date,:]
                
                major_flows[f'pywr_{model}'][f'realization_{i}'], datetime_index = get_pywr_results(output_dir, model, 'major_flow', datetime_index=datetime_index, scenario=i)
                major_flows[f'pywr_{model}'][f'realization_{i}']= major_flows[f'pywr_{model}'][f'realization_{i}'].loc[start_date:end_date,:]
                
                storages[f'pywr_{model}'][f'realization_{i}'], datetime_index = get_pywr_results(output_dir, model, 'res_storage', datetime_index=datetime_index, scenario=i)
        else:
            reservoir_downstream_gages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'reservoir_downstream_gage', datetime_index=datetime_index)
            reservoir_downstream_gages[f'pywr_{model}'] = reservoir_downstream_gages[f'pywr_{model}'].loc[start_date:end_date,:]
            
            major_flows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'major_flow', datetime_index=datetime_index)
            major_flows[f'pywr_{model}'] = major_flows[f'pywr_{model}'].loc[start_date:end_date,:]
            
            storages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'res_storage', datetime_index=datetime_index)
            inflows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'inflow', datetime_index=datetime_index)
    pywr_models = [f'pywr_{m}' for m in pywr_models]
    
    
    # Load base (non-pywr) models
    datetime_index = list(reservoir_downstream_gages.values())[0].index
    for model in base_models:
        print(model)
        reservoir_downstream_gages[model], datetime_index = get_base_results(input_dir, model, datetime_index, 'reservoir_downstream_gage')
        major_flows[model], datetime_index = get_base_results(input_dir, model, datetime_index, 'major_flow')
    

    # Verify that all datasets have same datetime index
    for m, r_df in major_flows.items():
        # assert ((r.index == datetime_index).mean() == 1)
        # Debug
        if r_df.index.equals(datetime_index) == False:
            print(f'Error with datetime index for model {m}')
    print(f'Successfully loaded {len(base_models)} base model results & {len(pywr_models)} pywr model results')

    ## 3-part flow figures with releases
    print('Plotting 3-part flows at reservoirs.')

    # plot_3part_flows(reservoir_downstream_gages, ['obs_pub_nhmv10_NYCScaled'], 'pepacton', fig_dir = fig_dir)
    # plot_3part_flows(reservoir_downstream_gages, ['pywr_obs_pub_nhmv10_NYCScaled'], 'pepacton', fig_dir = fig_dir)
    # plot_3part_flows(reservoir_downstream_gages, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'pepacton', fig_dir = fig_dir)

    # plot_3part_flows(reservoir_downstream_gages, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'cannonsville', fig_dir = fig_dir)
    # plot_3part_flows(reservoir_downstream_gages, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'neversink', fig_dir = fig_dir)


    print('Plotting weekly flow distributions at reservoirs.')
    # plot_weekly_flow_distributions(reservoir_downstream_gages, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'pepacton', fig_dir = fig_dir)
    # plot_weekly_flow_distributions(reservoir_downstream_gages, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'cannonsville', fig_dir = fig_dir)
    # plot_weekly_flow_distributions(reservoir_downstream_gages, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'neversink', fig_dir = fig_dir)

    ### now do figs for major flow locations


    ### flow comparisons for major flow nodes
    # plot_3part_flows(major_flows, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'delMontague', fig_dir = fig_dir)
    # plot_3part_flows(major_flows, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'delTrenton', fig_dir = fig_dir)
    # plot_3part_flows(major_flows, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'outletSchuylkill', fig_dir = fig_dir)
    # plot_weekly_flow_distributions(major_flows, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'delMontague', fig_dir = fig_dir)
    # plot_weekly_flow_distributions(major_flows, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'delTrenton', fig_dir = fig_dir)
    # plot_weekly_flow_distributions(major_flows, ['obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled'], 'outletSchuylkill', fig_dir = fig_dir)

    # ## RRV metrics
    # print('Plotting RRV metrics.')
    # nodes = ['delMontague','delTrenton']
    # rrv_metrics = get_RRV_metrics(major_flows, rrv_models, nodes)
    # plot_rrv_metrics(rrv_metrics, rrv_models, nodes, fig_dir = fig_dir)

    # ## Plot flow contributions at Trenton
    # print('Plotting NYC reservoir operations')
    plot_combined_nyc_storage(storages, reservoir_downstream_gages, 
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date=start_date, end_date=end_date, fig_dir = fig_dir,
                              add_ffmp_levels=False)
    
    ## 1960s drought
    plot_combined_nyc_storage(storages, reservoir_downstream_gages, 
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='1962-01-01', end_date='1970-01-01', fig_dir = fig_dir)

    ## 2002 drought
    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2000-01-01', end_date='2003-12-31', fig_dir = fig_dir,
                              add_ffmp_levels=False, plot_observed=False, plot_sim=False, filename_addon='_part1')
    
    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2000-01-01', end_date='2003-12-31', fig_dir = fig_dir,
                              add_ffmp_levels=True, plot_observed=False, plot_sim=False, filename_addon='_part2')

    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2000-01-01', end_date='2003-12-31', fig_dir = fig_dir,
                              add_ffmp_levels=True, plot_observed=True, plot_sim=False, filename_addon='_part3')

    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2000-01-01', end_date='2003-12-31', fig_dir = fig_dir,
                              add_ffmp_levels=True, plot_observed=True, plot_sim=True, filename_addon='_part4')

    ## FFMP Period
    p_start='2018-07-01'
    p_end='2020-07-01'
    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date=p_start, end_date=p_end, fig_dir = fig_dir,
                              add_ffmp_levels=False, plot_observed=False, plot_sim=False, filename_addon='_part1')
    
    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date=p_start, end_date=p_end, fig_dir = fig_dir,
                              add_ffmp_levels=True, plot_observed=False, plot_sim=False, filename_addon='_part2')

    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date=p_start, end_date=p_end, fig_dir = fig_dir,
                              add_ffmp_levels=True, plot_observed=True, plot_sim=False, filename_addon='_part3')

    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date=p_start, end_date=p_end, fig_dir = fig_dir,
                              add_ffmp_levels=True, plot_observed=True, plot_sim=True, filename_addon='_part4')


    
    # 2016 drought warning 
    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2016-01-01', end_date='2018-12-31', fig_dir = fig_dir)

    plot_combined_nyc_storage(storages, reservoir_downstream_gages,
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date= '2014-09-17', end_date= '2017-05-17', fig_dir = fig_dir)

                        
    # ## Post-2017
    # plot_combined_nyc_storage(storages, reservoir_downstream_gages, 
    #                           ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2019-01-01', end_date='2020-12-31', fig_dir = fig_dir)
    plot_combined_nyc_storage(storages, reservoir_downstream_gages, 
                              ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2018-01-01', end_date='2020-12-31', fig_dir = fig_dir)

    plot_combined_nyc_storage(storages, reservoir_downstream_gages, 
                                ['pywr_obs_pub_nhmv10'], start_date='2018-01-01', end_date='2020-12-31', 
                                fig_dir = fig_dir, filename_addon='_unscaled')

    plot_combined_nyc_storage(storages, reservoir_downstream_gages, 
                                ['pywr_obs_pub_nhmv10_NYCScaled'], start_date='2000-01-01', end_date='2020-12-31', 
                                fig_dir = fig_dir, filename_addon='_unscaled')

    # ## Radial metric plots
    # nodes = ['cannonsville', 'pepacton', 'neversink', 'fewalter', 'beltzvilleCombined', 'blueMarsh']
    # radial_models = ['obs', 'obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10_NYCScaled', 
    #                  'obs_pub_nwmv21_NYCScaled', 'pywr_obs_pub_nwmv21_NYCScaled']
    # radial_models = radial_models[::-1]
    
    # print('Plotting radial error metrics for major flows.')
    # nodes = ['delMontague', 'delTrenton', 'outletSchuylkill']  # , 'outletChristina', 'delLordville']
    # major_flow_metrics = get_error_metrics(major_flows, radial_models, nodes)
    # plot_radial_error_metrics(major_flow_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = True, usemajorflows=True, fig_dir = fig_dir)

    # ### compile error metrics across models/nodes/metrics
    # print('Plotting radial figures for reservoir releases')
    # nodes = ['cannonsville', 'pepacton', 'neversink', 'fewalter', 'beltzvilleCombined', 'blueMarsh']
    # reservoir_downstream_gage_metrics = get_error_metrics(reservoir_downstream_gages, radial_models, nodes)
    # plot_radial_error_metrics(reservoir_downstream_gage_metrics, radial_models, nodes, useNonPep = True, useweap = False, usepywr = True, fig_dir = fig_dir)


    ## Plot flow contributions at Trenton

    print('Plotting flow contributions at major nodes.')

    node = 'delTrenton'
    models = ['pywr_obs_pub_nhmv10_NYCScaled', 'pywr_obs_pub_nhmv10']
    for model in models:  
        plot_flow_contributions(reservoir_downstream_gages, major_flows, inflows, 
                                model, node, start_date=p_start, end_date=p_end,
                                log_flows=True,
                                fig_dir=fig_dir)
        
        plot_flow_contributions(reservoir_downstream_gages, major_flows, inflows, 
                        model, node,
                        start_date= '2001-01-01',
                        end_date= '2003-12-31',
                        fig_dir=fig_dir)

        plot_flow_contributions(reservoir_downstream_gages, major_flows, inflows, 
                        model, node,
                        start_date= '2017-07-01',
                        end_date= '2022-12-31',
                        fig_dir=fig_dir)

    print(f'Done! Check the {fig_dir} folder.')