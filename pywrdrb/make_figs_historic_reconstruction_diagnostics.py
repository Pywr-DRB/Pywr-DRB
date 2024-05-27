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

fig_dir += 'full_reconstruction_diagnostics/'

## Execution - Generate all figures
if __name__ == "__main__":

    rerun_all = True
    model_colordict = model_colors_historic_reconstruction

    ## Load data    
    # Load Pywr-DRB simulation models
    print(f'Retrieving simulation data.')
    pywr_models = ['nhmv10', 'nwmv21', 'obs_pub_nhmv10_ObsScaled', 'obs_pub_nwmv21_ObsScaled']
    base_models = ['obs', 'nhmv10', 'nwmv21', 'nhmv10_withObsScaled', 'nwmv21_withObsScaled']
    
    reservoir_downstream_gages = {}
    major_flows = {}
    storages = {}
    reservoir_releases = {}
    ffmp_levels = {}
    ffmp_level_boundaries = {}
    inflows = {}
    nyc_release_components = {}
    ibt_demands = {}
    ibt_diversions = {}
    catchment_consumptions = {}
    mrf_targets = {}
    downstream_release_targets = {}

    datetime_index = None
    for model in pywr_models:
        print(f'pywr_{model}')
        reservoir_downstream_gages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        reservoir_downstream_gages[f'pywr_{model}']['NYCAgg'] = reservoir_downstream_gages[f'pywr_{model}'][reservoir_list_nyc].sum(axis=1)
        major_flows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='major_flow', datetime_index=datetime_index)
        storages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_storage', datetime_index=datetime_index)
        reservoir_releases[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_release', datetime_index=datetime_index)
        ffmp_levels[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_level', datetime_index=datetime_index)
        inflows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'inflow', datetime_index=datetime_index)
        nyc_release_components[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'nyc_release_components', datetime_index=datetime_index)
        ibt_demands[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'ibt_demands', datetime_index=datetime_index)
        ibt_diversions[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'ibt_diversions', datetime_index=datetime_index)
        catchment_consumptions[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'catchment_consumption', datetime_index=datetime_index)
        mrf_targets[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'mrf_targets', datetime_index=datetime_index)
        downstream_release_targets[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'downstream_release_target', datetime_index=datetime_index)

    ffmp_level_boundaries, datetime_index = get_pywr_results(output_dir, model, results_set='ffmp_level_boundaries', datetime_index=datetime_index)

    pywr_models = [f'pywr_{m}' for m in pywr_models]

    ### Load base (non-pywr) models
    datetime_index = list(reservoir_downstream_gages.values())[0].index
    for model in base_models:
        print(model)
        reservoir_downstream_gages[model], datetime_index = get_base_results(input_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        reservoir_downstream_gages[model]['NYCAgg'] = reservoir_downstream_gages[model][reservoir_list_nyc].sum(axis=1)
        major_flows[model], datetime_index = get_base_results(input_dir, model, results_set='major_flow', datetime_index=datetime_index)

    ### different time periods for different type of figures.
    start_date_obs = pd.to_datetime('2008-01-01')   ### comparison to observed record, error metrics
    end_date_obs = pd.to_datetime('2017-01-01')
    start_date_short_obs = pd.to_datetime('2011-01-01')   ### comparison to observed record, error metrics
    end_date_short_obs = pd.to_datetime('2013-01-01')
    start_date_full = pd.to_datetime('1984-01-01')  ### full NHM/NWM time series
    end_date_full = pd.to_datetime('2017-01-01')
    start_date_medium_preobs = pd.to_datetime('1990-01-01') ### 10-year period in pre-observed record for zoomed dynamics
    end_date_medium_preobs = pd.to_datetime('2000-01-01')
    start_date_short_preobs = pd.to_datetime('1993-01-01') ### short 2-year period pre observed record for zoomed dynamics
    end_date_short_preobs = pd.to_datetime('1996-01-01')



    ### first set of subplots showing comparison of modeled & observed flows at 3 locations, top to bottom of basin
    if rerun_all:
        print('Plotting 3x3 flow comparison figure')
        nodes = ['cannonsville','NYCAgg','delTrenton']
        for model in pywr_models:
            ### first with full observational record
            plot_3part_flows_hier(reservoir_downstream_gages, major_flows, [model.replace('pywr_',''), model], uselog=True,
                                  colordict=model_colors_historic_reconstruction, start_date=start_date_obs, end_date=end_date_obs)
            ### now zoomed in 2 year period where it is easier to see time series
            plot_3part_flows_hier(reservoir_downstream_gages, major_flows, [model.replace('pywr_', ''), model], uselog=True,
                                  colordict=model_colors_historic_reconstruction, start_date=start_date_short_obs, end_date=end_date_short_obs)


    ### compare modeled vs observed NYC storages
    if rerun_all:
        print('Plotting new NYC storage figure')
        plot_combined_nyc_storage_new(storages, ffmp_level_boundaries, pywr_models, colordict = model_colors_historic_reconstruction,
                                      start_date=start_date_obs, end_date=end_date_obs, fig_dir=fig_dir)




    ### gridded error metrics for NYC reservoirs + major flows
    if rerun_all:
        print('Plotting gridded error metrics')
        ### first do smaller subset of models & locations for main text
        error_models = base_models[1:-1] + pywr_models[:-1]
        nodes = ['cannonsville', 'NYCAgg', 'delTrenton']
        node_metrics = get_error_metrics(reservoir_downstream_gages, major_flows, error_models, nodes,
                                         start_date=start_date_obs, end_date=end_date_obs)
        plot_gridded_error_metrics(node_metrics, error_models, nodes,
                                   start_date=start_date_obs, end_date=end_date_obs, figstage=0)
        ### now full set of models & locations, for SI
        error_models = base_models[1:] + pywr_models
        nodes = reservoir_list_nyc + ['NYCAgg','delMontague','delTrenton']
        node_metrics = get_error_metrics(reservoir_downstream_gages, major_flows, error_models, nodes,
                                         start_date=start_date_obs, end_date=end_date_obs)
        plot_gridded_error_metrics(node_metrics, error_models, nodes,
                                   start_date=start_date_obs, end_date=end_date_obs, figstage=1)
        ### repeat full figure but with monthly time step
        plot_gridded_error_metrics(node_metrics, error_models, nodes,
                                   start_date=start_date_obs, end_date=end_date_obs, figstage=2)




    ### gridded low flow exceedances based on full record
    if rerun_all:
        print('Plotting low flow return period figs')
        models = base_models[1:] + pywr_models
        nodes = ['cannonsville', 'NYCAgg', 'delTrenton']
        plot_lowflow_exceedances(reservoir_downstream_gages, major_flows, models, nodes,
                                 start_date=start_date_full, end_date=end_date_full,
                                 colordict=model_colors_historic_reconstruction)




    ### plot breaking down NYC flows & Trenton flows into components
    if rerun_all:
        print('Plotting NYC releases by components, combined with downstream flow components')
        for model in pywr_models:
            for node in ['delMontague','delTrenton']:
                plot_NYC_release_components_combined(nyc_release_components, reservoir_releases, major_flows, inflows,
                                                     ibt_diversions, catchment_consumptions, model, node,
                                                     use_proportional=True, use_log=True,
                                                     start_date=start_date_short_preobs, end_date=end_date_short_preobs,
                                                     fig_dir=fig_dir)






    ### plot shortfall event metrics (reliability/duration/intensity/vulnerability) distributions for min flows and ibt diversions
    if rerun_all:
        print('Plotting shortfall event metrics for min flows and ibt diversions')
        models_mrf = base_models[1:] + pywr_models
        models_ibt = pywr_models
        nodes = ['delMontague', 'delTrenton', 'nyc', 'nj']
        shortfall_type = 'absolute' ### 'absolute' or 'percent'
        shortfall_metrics = get_shortfall_metrics(major_flows, mrf_targets, ibt_demands, ibt_diversions, models_mrf,
                                                  models_ibt, nodes,
                                                  shortfall_type=shortfall_type,shortfall_threshold=0.99,  shortfall_break_length=30,
                                                  start_date=start_date_full, end_date=end_date_full)
        plot_shortfall_metrics(shortfall_metrics, models_mrf, models_ibt, nodes,
                               colordict=model_colors_historic_reconstruction, shortfall_type=shortfall_type)





    ### show NYC storage vs min flow satisfaction dynamics
    if rerun_all:
        print('Plotting NYC storages vs Montague/Trenton min flow targets')
        for base_model in base_models[1:]:  ### should be a base model, and we will compare base vs pywr version
            plot_combined_nyc_storage_vs_minflows(storages, ffmp_level_boundaries, major_flows, mrf_targets,
                                                  reservoir_releases, downstream_release_targets, base_model,
                                                  shortfall_metrics,
                                                  colordict=model_colors_historic_reconstruction,
                                                   start_date=start_date_medium_preobs, end_date=end_date_medium_preobs,
                                                   fig_dir=fig_dir)
            plot_combined_nyc_storage_vs_minflows(storages, ffmp_level_boundaries, major_flows, mrf_targets,
                                                  reservoir_releases, downstream_release_targets, base_model,
                                                  shortfall_metrics,
                                                   colordict=model_colors_historic_reconstruction,
                                                   start_date=start_date_full, end_date=end_date_full,
                                                   fig_dir=fig_dir)

    # ### show NYC storage vs diversion dynamics
    if rerun_all:
        print('Plotting NYC storages vs NYC/NJ diversions')
        for customer in ['nyc','nj']:
            plot_combined_nyc_storage_vs_diversion(storages, ffmp_level_boundaries, ibt_demands, ibt_diversions,
                                                   pywr_models, customer, shortfall_metrics,
                                                   colordict=model_colors_historic_reconstruction,
                                                   start_date=start_date_medium_preobs, end_date=end_date_medium_preobs,
                                                   fig_dir=fig_dir)
            plot_combined_nyc_storage_vs_diversion(storages, ffmp_level_boundaries, ibt_demands, ibt_diversions,
                                                   pywr_models, customer, shortfall_metrics,
                                                   colordict = model_colors_historic_reconstruction,
                                                   start_date=start_date_full, end_date=end_date_full,
                                                   fig_dir=fig_dir)




    print(f'Done! Check the {fig_dir} folder.')

