import pandas as pd
import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../pywrdrb/")


from pywrdrb.post.get_results import (
    get_base_results,
    get_all_historic_reconstruction_pywr_results,
)
from pywrdrb.utils.directories import input_dir, fig_dir, output_dir


from pywrdrb.plotting.styles import model_colors_historic_reconstruction
from pywrdrb.plotting.ensemble_plots import plot_NYC_release_components_combined
from pywrdrb.utils.lists import reservoir_list_nyc

# Redirect figures
fig_dir = fig_dir + "historic_reconstruction/"

### Alternative windows of focus
start_date = pd.to_datetime("1945-01-01")
end_date = pd.to_datetime("2022-12-31")

start_1960s_drought = pd.to_datetime("1964-01-01")
end_1960s_drought = pd.to_datetime("1967-01-01")

start_1980s_drought = pd.to_datetime("1980-06-01")
end_1980s_drought = pd.to_datetime("1982-10-31")

start_ffmp = pd.to_datetime("2017-10-01")
end_ffmp = pd.to_datetime("2022-12-31")


date_ranges = {
    "1960s_drought": (start_1960s_drought, end_1960s_drought),
    "1980s_drought": (start_1980s_drought, end_1980s_drought),
    "post_ffmp": (start_ffmp, end_ffmp),
    "full": (start_date, end_date),
}

model_labels = {
    "obs": "Observed",
    "nhmv10": "NHMv1.0",
    "nwmv21": "NWMv2.1",
    "obs_pub_nhmv10": "PUB-NHM",
    "obs_pub_nhmv10_ensemble": "PUB-NHM Ensemble",
    "obs_pub_nwmv21": "PUB-NWM",
    "obs_pub_nwmv21_ensemble": "PUB-NWM Ensemble",
    "obs_pub_nhmv10_ObsScaled": "PUB-NHM",
    "obs_pub_nhmv10_ObsScaled_ensemble": "PUB-NHM Ensemble",
    "obs_pub_nwmv21_ObsScaled": "PUB-NWM",
    "obs_pub_nwmv21_ObsScaled_ensemble": "PUB-NWM Ensemble",
}


## Execution - Generate all figures
if __name__ == "__main__":
    rerun_all = True
    start_date = "1945-01-01"
    end_date = "2022-12-31"

    model_list = [
        "obs_pub_nhmv10_ObsScaled",
        "obs_pub_nwmv21_ObsScaled",
        "obs_pub_nhmv10_ObsScaled_ensemble",
        "obs_pub_nwmv21_ObsScaled_ensemble",
    ]

    ensemble_models = [
        "obs_pub_nhmv10_ObsScaled_ensemble",
        "obs_pub_nwmv21_ObsScaled_ensemble",
    ]

    pywr_ensemble_models = ["pywr_" + m for m in ensemble_models]
    pywr_models = [f"pywr_{m}" for m in model_list]
    base_models = ["obs", "nhmv10", "nwmv21"] + model_list[:2]

    #################################
    ### Load data ###################
    #################################
    print("Loading Pywr-DRB simulation ensemble results...")
    major_flows = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="major_flow",
        start_date=start_date,
        end_date=end_date,
    )
    reservoir_downstream_gages = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="reservoir_downstream_gage",
        start_date=start_date,
        end_date=end_date,
    )
    lower_basin_mrf_contributions = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="lower_basin_mrf_contributions",
        start_date=start_date,
        end_date=end_date,
    )
    ibt_diversions = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="ibt_diversions",
        start_date=start_date,
        end_date=end_date,
    )
    catchment_consumptions = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="catchment_consumption",
        start_date=start_date,
        end_date=end_date,
    )

    storages = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="res_storage",
        start_date=start_date,
        end_date=end_date,
    )
    ffmp_levels = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="ffmp_level_boundaries",
        start_date=start_date,
        end_date=end_date,
    )

    reservoir_inflows = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="inflow",
        start_date=start_date,
        end_date=end_date,
    )
    # targets = get_all_historic_reconstruction_pywr_results(output_dir=output_dir, model_list=model_list,
    #                                                                 results_set='mrf_target',
    #                                                                 start_date=start_date, end_date=end_date)
    nyc_releases = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="nyc_release_components",
        start_date=start_date,
        end_date=end_date,
    )

    reservoir_releases = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="res_release",
        start_date=start_date,
        end_date=end_date,
    )
    reservoir_inflows = get_all_historic_reconstruction_pywr_results(
        output_dir=output_dir,
        model_list=model_list,
        results_set="inflow",
        start_date=start_date,
        end_date=end_date,
    )

    # Get datetime index
    datetime_index = major_flows["pywr_obs_pub_nhmv10_ObsScaled"].index

    ### Load base (non-pywr) models
    print('Loading "base" models, pre-pywrdrb data...')
    for model in base_models:
        reservoir_downstream_gages[model], datetime_index = get_base_results(
            input_dir, model, datetime_index, "reservoir_downstream_gage"
        )
        major_flows[model], datetime_index = get_base_results(
            input_dir, model, datetime_index, "major_flow"
        )
        reservoir_inflows[model], datetime_index = get_base_results(
            input_dir, model, datetime_index, "inflow"
        )

    ### Get aggregate NYC data
    for model in pywr_models:
        for real in reservoir_releases[model].keys():
            reservoir_downstream_gages[model][real][
                "NYCAgg"
            ] = reservoir_downstream_gages[model][real][reservoir_list_nyc].sum(axis=1)

        if "ensemble" in model:
            for real in reservoir_releases[model].keys():
                reservoir_downstream_gages[model][real][
                    "NYCAgg"
                ] = reservoir_downstream_gages[model][real][reservoir_list_nyc].sum(
                    axis=1
                )
        else:
            reservoir_downstream_gages[model]["NYCAgg"] = reservoir_downstream_gages[
                model
            ][reservoir_list_nyc].sum(axis=1)

    for model in base_models:
        reservoir_downstream_gages[model]["NYCAgg"] = reservoir_downstream_gages[model][
            reservoir_list_nyc
        ].sum(axis=1)

    #################################
    ### Plotting ####################
    #################################

    ### NYC reservoir operations and downstream flow contributions
    for model in pywr_ensemble_models:
        for node in ["delMontague", "delTrenton"]:
            for dates in date_ranges.keys():
                if dates == "full":
                    continue

                plot_obs = True if dates == "post_ffmp" else False

                start = date_ranges[dates][0]
                end = date_ranges[dates][1]
                plot_NYC_release_components_combined(
                    storages,
                    ffmp_level_boundaries=ffmp_levels["pywr_obs_pub_nwmv21_ObsScaled"],
                    model=model,
                    node=node,
                    nyc_release_components=nyc_releases,
                    lower_basin_mrf_contributions=lower_basin_mrf_contributions,
                    reservoir_releases=reservoir_releases,
                    reservoir_downstream_gages=reservoir_downstream_gages,
                    major_flows=major_flows,
                    inflows=reservoir_inflows,
                    diversions=ibt_diversions,
                    consumptions=catchment_consumptions,
                    colordict=model_colors_historic_reconstruction,
                    start_date=start,
                    end_date=end,
                    plot_observed=plot_obs,
                    percentile_cmap=True,
                    contribution_fill_alpha=0.9,
                    plot_flow_target=True,
                    use_log=False,
                    q_lower_bound=0.01,
                    q_upper_bound=0.99,
                    smoothing_window=7,
                    fig_dir=fig_dir,
                )

    ### OLD SCRIPTS: Outdated ###

    ### NYC Storage across ensembles
    # print('Plotting NYC Storage figure...')
    # for dates in date_ranges.keys():
    #     start = date_ranges[dates][0]
    #     end = date_ranges[dates][1]
    #     plot_ensemble_nyc_storage(storages,
    #                               ffmp_levels['pywr_obs_pub_nhmv10_ObsScaled'],
    #                               pywr_models,
    #                               model_colors_historic_reconstruction,
    #                               start_date=start,
    #                               end_date = end,
    #                               plot_observed=False,
    #                               fig_dir=fig_dir,
    #                               fill_ffmp_levels=False,
    #                               plot_ensemble_mean=False,
    #                               ax=None)

    # ### NYC Storage, Montauge Deficit, and Trenton Deficit
    # print('Plotting NYC Storage, Montauge Deficit, and Trenton Deficit figure...')
    # for dates in date_ranges.keys():
    #     start = date_ranges[dates][0]
    #     end = date_ranges[dates][1]
    #     plot_ensemble_nyc_storage_and_deficit(storages,
    #                                             major_flows,
    #                                             ffmp_levels['pywr_obs_pub_nhmv10_ObsScaled'],
    #                                             pywr_models,
    #                                             start_date=start,
    #                                             end_date=end,
    #                                             plot_observed=True,
    #                                             plot_ensemble_mean=False,
    #                                             percentiles_cmap=False,
    #                                             fill_ffmp_levels=False,
    #                                             ensemble_fill_alpha=0.75,
    #                                             fig_dir=fig_dir,
    #                                             dpi=200)

    # ### NYC Storage, Target Deficit, and Total Flow
    # print('Plotting NYC Storage, Target Deficit, and Total Flow figure...')
    # for dates in date_ranges.keys():
    #     start = date_ranges[dates][0]
    #     end = date_ranges[dates][1]
    #     for node in ['delTrenton', 'delMontague']:
    #         plot_ensemble_nyc_storage_flow_deficit(storages,
    #                                             major_flows,
    #                                             ffmp_levels['pywr_obs_pub_nhmv10_ObsScaled'],
    #                                             pywr_models,
    #                                             node=node,
    #                                             start_date=start,
    #                                             end_date=end,
    #                                             plot_observed=True,
    #                                             plot_ensemble_mean=False,
    #                                             percentiles_cmap=False,
    #                                             fill_ffmp_levels=False,
    #                                             ensemble_fill_alpha=0.75,
    #                                             fig_dir=fig_dir,
    #                                             dpi=200)
