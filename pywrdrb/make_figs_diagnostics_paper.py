import numpy as np
import pandas as pd
import sys

sys.path.append('./')
sys.path.append('../')

from pywrdrb.plotting.plotting_functions import *
from pywrdrb.plotting.styles import *
from pywrdrb.utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list, majorflow_list_figs, reservoir_link_pairs
from pywrdrb.utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from pywrdrb.utils.directories import input_dir, output_dir, fig_dir
from pywrdrb.post.get_results import get_base_results, get_pywr_results

### I was having trouble with interactive console plotting in Pycharm for some reason - comment this out if you want to use that and not having issues
# mpl.use('TkAgg')




## Execution - Generate all figures
if __name__ == "__main__":

    rerun_all = True

    ## Load data    
    # Load Pywr-DRB simulation models
    print(f'Retrieving simulation data.')
    pywr_models = ['nhmv10', 'nwmv21', 'nhmv10_withObsScaled', 'nwmv21_withObsScaled']

    reservoir_downstream_gages = {}
    major_flows = {}
    storages = {}
    reservoir_releases = {}
    all_drought_levels = {}
    inflows = {}
    nyc_release_components = {}
    diversions = {}
    consumptions = {}

    datetime_index = None
    for model in pywr_models:
        print(f'pywr_{model}')
        reservoir_downstream_gages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        major_flows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='major_flow', datetime_index=datetime_index)
        storages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_storage', datetime_index=datetime_index)
        reservoir_releases[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_release', datetime_index=datetime_index)
        all_drought_levels[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_level', datetime_index=datetime_index)
        inflows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'inflow', datetime_index=datetime_index)
        nyc_release_components[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'nyc_release_components', datetime_index=datetime_index)
        diversions[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'diversions', datetime_index=datetime_index)
        consumptions[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'consumption', datetime_index=datetime_index)

    pywr_models = [f'pywr_{m}' for m in pywr_models]

    ### Load base (non-pywr) models
    base_models = ['obs', 'nhmv10', 'nwmv21', 'nhmv10_withObsScaled', 'nwmv21_withObsScaled']

    datetime_index = list(reservoir_downstream_gages.values())[0].index
    for model in base_models:
        print(model)
        reservoir_downstream_gages[model], datetime_index = get_base_results(input_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        major_flows[model], datetime_index = get_base_results(input_dir, model, results_set='major_flow', datetime_index=datetime_index)


    start_date = pd.to_datetime('2008-01-01')
    end_date = pd.to_datetime('2017-01-01')
    ## 3-part flow figures with releases
    if rerun_all:
        uselog=True
        print('Plotting 3-part flows at nodes.')
        for node in reservoir_list_nyc:
            for model in pywr_models:
                plot_3part_flows(reservoir_downstream_gages, [model.replace('pywr_',''), model], node, uselog=uselog,
                                 colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_3part_flows(reservoir_downstream_gages, [pywr_models[0], pywr_models[2]], node, uselog=uselog,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_3part_flows(reservoir_downstream_gages, [pywr_models[1], pywr_models[3]], node, uselog=uselog,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
        for node in majorflow_list_figs:
            for model in pywr_models:
                plot_3part_flows(major_flows, [model.replace('pywr_',''), model], node, uselog=uselog,
                                 colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_3part_flows(major_flows, [pywr_models[0], pywr_models[2]], node, uselog=uselog,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_3part_flows(major_flows, [pywr_models[1], pywr_models[3]], node, uselog=uselog,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)



    if rerun_all:
        print('Plotting weekly flow distributions at nodes.')
        for node in reservoir_list_nyc:
            for model in pywr_models:
                plot_weekly_flow_distributions(reservoir_downstream_gages, [model.replace('pywr_',''), model], node,
                                 colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_weekly_flow_distributions(reservoir_downstream_gages, [pywr_models[0], pywr_models[2]], node,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_weekly_flow_distributions(reservoir_downstream_gages, [pywr_models[1], pywr_models[3]], node,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
        for node in majorflow_list_figs:
            for model in pywr_models:
                plot_weekly_flow_distributions(major_flows, [model.replace('pywr_',''), model], node,
                                 colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_weekly_flow_distributions(major_flows, [pywr_models[0], pywr_models[2]], node,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)
            plot_weekly_flow_distributions(major_flows, [pywr_models[1], pywr_models[3]], node,
                             colordict=model_colors_diagnostics_paper, start_date=start_date, end_date=end_date)





    radial_models = base_models[1:] + pywr_models
    radial_models = radial_models[::-1]

    ### radial error metrics for NYC reservoirs + major flows
    if rerun_all:
        print('Plotting radial error metrics for reservoirs + major flows.')
        nodes = reservoir_list_nyc[::-1] + ['beltzvilleCombined', 'blueMarsh','fewalter']
        node_metrics = get_error_metrics(reservoir_downstream_gages, major_flows, radial_models, nodes,
                                         start_date=start_date, end_date=end_date)
        plot_radial_error_metrics(node_metrics, radial_models, nodes, usemajorflows=False,
                                  colordict=model_colors_diagnostics_paper)

        nodes = reservoir_list_nyc[::-1] + majorflow_list_figs[::-1]
        node_metrics = get_error_metrics(reservoir_downstream_gages, major_flows, radial_models, nodes,
                                         start_date=start_date, end_date=end_date)
        plot_radial_error_metrics(node_metrics, radial_models, nodes, usemajorflows=True,
                                  colordict=model_colors_diagnostics_paper)




    ## RRV metrics
    if rerun_all:
        print('Plotting RRV metrics.')
        rrv_models = base_models + pywr_models

        nodes = ['delMontague','delTrenton']
        rrv_metrics = get_RRV_metrics(major_flows, rrv_models, nodes, start_date=start_date, end_date=end_date)
        plot_rrv_metrics(rrv_metrics, rrv_models, nodes, colordict=model_colors_diagnostics_paper)



    ## Plot NYC storage dynamics
    if rerun_all:
        print('Plotting NYC reservoir storages & releases')
        for reservoir in ['agg']+reservoir_list_nyc:
            plot_combined_nyc_storage(storages, reservoir_releases, all_drought_levels, pywr_models,
                                      start_date=start_date, end_date=end_date, reservoir=reservoir, fig_dir=fig_dir,
                                      colordict=model_colors_diagnostics_paper,
                                      add_ffmp_levels=True, plot_observed=True, plot_sim=True)


    # ### flow contributions plot
    # if rerun_all:
    #     print('Plotting flow contributions at major nodes.')
    #     for node in ['delMontague', 'delTrenton']:
    #         for model in pywr_models:
    #             plot_flow_contributions(reservoir_releases, major_flows, inflows, model, node,
    #                                     start_date= start_date, end_date= end_date, log_flows = True, fig_dir = fig_dir)



    # ## Plot inflow comparison
    # if rerun_all:
    #     print('Plotting inflow data boxplots')
    #     compare_inflow_data(inflows, reservoir_list, pywr_models,
    #                         start_date=start_date, end_date=end_date, fig_dir=fig_dir)

    ### xQn grid low flow comparison figure
    if rerun_all:
        print('Plotting low flow grid.')
        plot_xQn_grid(reservoir_downstream_gages, major_flows,  base_models + pywr_models,
                      reservoir_list_nyc + majorflow_list_figs, xlist = [1,7,30,90, 365], nlist = [5, 10, 20, 30],
                      start_date=start_date, end_date=end_date, fig_dir=fig_dir)

    ### plot comparing flow series with overlapping boxplots & FDCs
    if rerun_all:
        print('Plotting monthly boxplot/FDC figures')
        for node in reservoir_list_nyc + majorflow_list_figs:
            plot_monthly_boxplot_fdc_combined(reservoir_downstream_gages, major_flows, base_models, pywr_models, node,
                                              colordict=model_colors_diagnostics_paper, start_date=start_date,
                                              end_date=end_date, fig_dir=fig_dir)


    ### plot breaking down NYC flows into components
    # if rerun_all:
    #     print('Plotting NYC releases by components')
    #     for model in pywr_models:
    #         plot_NYC_release_components_indiv(nyc_release_components, reservoir_releases, model,
    #                                             use_proportional=True, use_log=True,
    #                                             start_date=start_date, end_date=end_date, fig_dir=fig_dir)
    #         plot_NYC_release_components_indiv(nyc_release_components, reservoir_releases, model,
    #                                             use_proportional=True, use_log=True,
    #                                             start_date='2011-01-01', end_date='2013-01-01', fig_dir=fig_dir)

    if rerun_all:
        print('Plotting NYC releases by components, combined with downstream flow components')
        for model in pywr_models:
            for node in ['delMontague','delTrenton']:
                plot_NYC_release_components_combined(nyc_release_components, reservoir_releases, major_flows, inflows,
                                                     diversions, consumptions, model, node, use_proportional=True, use_log=True,
                                                     start_date=start_date, end_date=end_date, fig_dir=fig_dir)

                plot_NYC_release_components_combined(nyc_release_components, reservoir_releases, major_flows, inflows,
                                                     diversions, consumptions, model, node, use_proportional=True, use_log=True,
                                                     start_date='2011-01-01', end_date='2013-01-01', fig_dir=fig_dir)


    print(f'Done! Check the {fig_dir} folder.')