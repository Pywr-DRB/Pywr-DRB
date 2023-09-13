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
    pywr_models = ['nhmv10', 'nwmv21', 'obs_pub_nwmv21_NYCScaled']

    reservoir_downstream_gages = {}
    major_flows = {}
    storages = {}
    reservoir_releases = {}
    all_drought_levels = {}
    inflows = {}

    datetime_index = None
    for model in pywr_models:
        print(f'pywr_{model}')
        reservoir_downstream_gages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        major_flows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='major_flow', datetime_index=datetime_index)
        storages[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_storage', datetime_index=datetime_index)
        reservoir_releases[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_release', datetime_index=datetime_index)
        all_drought_levels[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, results_set='res_level', datetime_index=datetime_index)
        inflows[f'pywr_{model}'], datetime_index = get_pywr_results(output_dir, model, 'inflow', datetime_index=datetime_index)

    pywr_models = [f'pywr_{m}' for m in pywr_models]

    ### Load base (non-pywr) models
    base_models = ['obs', 'nhmv10', 'nwmv21', 'obs_pub_nwmv21_NYCScaled']

    datetime_index = list(reservoir_downstream_gages.values())[0].index
    for model in base_models:
        print(model)
        reservoir_downstream_gages[model], datetime_index = get_base_results(input_dir, model, results_set='reservoir_downstream_gage', datetime_index=datetime_index)
        major_flows[model], datetime_index = get_base_results(input_dir, model, results_set='major_flow', datetime_index=datetime_index)


    start_date = pd.to_datetime('1984-01-01')
    end_date = pd.to_datetime('2000-10-01')


    ## Plot NYC storage dynamics
    if rerun_all:
        plot_combined_nyc_storage(storages, reservoir_releases, all_drought_levels, pywr_models,
                                  start_date=start_date, end_date=end_date, fig_dir=fig_dir,
                                  colordict=model_colors_diagnostics_paper,
                                  add_ffmp_levels=True, plot_observed=True, plot_sim=True, filename_addon='_part4')

    ### flow contributions plot
    if rerun_all:
        print('Plotting flow contributions at major nodes.')
        for node in ['delMontague', 'delTrenton']:
            for model in pywr_models:
                plot_flow_contributions(reservoir_releases, major_flows, inflows, model, node,
                                        start_date= start_date, end_date= end_date, log_flows = True, fig_dir = fig_dir)



    ## Plot inflow comparison
    if rerun_all:
        compare_inflow_data(inflows, reservoir_list, pywr_models,
                            start_date=start_date, end_date=end_date, fig_dir=fig_dir)


    print(f'Done! Check the {fig_dir} folder.')