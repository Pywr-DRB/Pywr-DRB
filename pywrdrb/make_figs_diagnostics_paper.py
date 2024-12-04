import numpy as np
import pandas as pd
import sys

sys.path.append("./")
sys.path.append("../")

from pywrdrb.plotting.plotting_functions import *
from pywrdrb.plotting.styles import *
from pywrdrb.utils.lists import (
    reservoir_list,
    reservoir_list_nyc,
    majorflow_list,
    majorflow_list_figs,
    reservoir_link_pairs,
)
from pywrdrb.utils.constants import cms_to_mgd, cm_to_mg, cfs_to_mgd
from pywrdrb.utils.directories import input_dir, output_dir, fig_dir
from pywrdrb.post.get_results import get_base_results, get_pywr_results
from pywrdrb.post.metrics import get_shortfall_metrics


## Execution - Generate all figures
if __name__ == "__main__":
    fig_dir = fig_dir + "diagnostics_paper/"

    rerun_all = True
    remake_map = False
    units = "MCM"
    assert units in ["MCM", "MG"]

    ## Load data
    # Load Pywr-DRB simulation models
    print(f"Retrieving simulation data.")
    pywr_models = ["nhmv10", "nwmv21", "nhmv10_withObsScaled", "nwmv21_withObsScaled"]

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
    lower_basin_mrf_contributions = {}

    datetime_index = None
    for model in pywr_models:
        print(f"pywr_{model}")
        reservoir_downstream_gages[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            results_set="reservoir_downstream_gage",
            datetime_index=datetime_index,
            units=units,
        )
        reservoir_downstream_gages[f"pywr_{model}"][
            "NYCAgg"
        ] = reservoir_downstream_gages[f"pywr_{model}"][reservoir_list_nyc].sum(axis=1)
        major_flows[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            results_set="major_flow",
            datetime_index=datetime_index,
            units=units,
        )
        storages[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            results_set="res_storage",
            datetime_index=datetime_index,
            units=units,
        )
        reservoir_releases[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            results_set="res_release",
            datetime_index=datetime_index,
            units=units,
        )
        ffmp_levels[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir, model, results_set="res_level", datetime_index=datetime_index
        )
        inflows[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir, model, "inflow", datetime_index=datetime_index, units=units
        )
        nyc_release_components[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            "nyc_release_components",
            datetime_index=datetime_index,
            units=units,
        )
        (
            lower_basin_mrf_contributions[f"pywr_{model}"],
            datetime_index,
        ) = get_pywr_results(
            output_dir,
            model,
            "lower_basin_mrf_contributions",
            datetime_index=datetime_index,
            units=units,
        )
        ibt_demands[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir, model, "ibt_demands", datetime_index=datetime_index, units=units
        )
        ibt_diversions[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            "ibt_diversions",
            datetime_index=datetime_index,
            units=units,
        )
        catchment_consumptions[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            "catchment_consumption",
            datetime_index=datetime_index,
            units=units,
        )
        mrf_targets[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir, model, "mrf_targets", datetime_index=datetime_index, units=units
        )
        downstream_release_targets[f"pywr_{model}"], datetime_index = get_pywr_results(
            output_dir,
            model,
            "downstream_release_target",
            datetime_index=datetime_index,
            units=units,
        )

    ffmp_level_boundaries, datetime_index = get_pywr_results(
        output_dir,
        model,
        results_set="ffmp_level_boundaries",
        datetime_index=datetime_index,
    )

    pywr_models = [f"pywr_{m}" for m in pywr_models]

    ### Load base (non-pywr) models
    base_models = [
        "obs",
        "nhmv10",
        "nwmv21",
        "nhmv10_withObsScaled",
        "nwmv21_withObsScaled",
    ]

    datetime_index = list(reservoir_downstream_gages.values())[0].index
    for model in base_models:
        print(model)
        reservoir_downstream_gages[model], datetime_index = get_base_results(
            input_dir,
            model,
            results_set="reservoir_downstream_gage",
            datetime_index=datetime_index,
            units=units,
        )
        reservoir_downstream_gages[model] = reservoir_downstream_gages[model][0]
        reservoir_downstream_gages[model]["NYCAgg"] = reservoir_downstream_gages[model][
            reservoir_list_nyc
        ].sum(axis=1)

        major_flows[model], datetime_index = get_base_results(
            input_dir,
            model,
            results_set="major_flow",
            datetime_index=datetime_index,
            units=units,
        )
        major_flows[model] = major_flows[model][0]

    ### different time periods for different type of figures.
    start_date_obs = pd.to_datetime(
        "2008-01-01"
    )  ### comparison to observed record, error metrics
    end_date_obs = pd.to_datetime("2017-01-01")
    start_date_short_obs = pd.to_datetime(
        "2010-01-01"
    )  ### comparison to observed record, error metrics
    end_date_short_obs = pd.to_datetime("2013-01-01")
    start_date_full = pd.to_datetime("1984-01-01")  ### full NHM/NWM time series
    end_date_full = pd.to_datetime("2017-01-01")
    start_date_medium_preobs = pd.to_datetime(
        "1990-01-01"
    )  ### 10-year period in pre-observed record for zoomed dynamics
    end_date_medium_preobs = pd.to_datetime("2000-01-01")
    start_date_short_preobs = pd.to_datetime(
        "2001-01-01"
    )  ### short 2-year period pre observed record for zoomed dynamics
    end_date_short_preobs = pd.to_datetime("2004-01-01")
    start_date_storage_obs = pd.to_datetime("2000-01-01")  ### full NHM/NWM time series

    ### first set of subplots showing comparison of modeled & observed flows at 3 locations, top to bottom of basin
    if rerun_all:
        print("\nPlotting 3x3 flow comparison figure\n")
        nodes = ["cannonsville", "NYCAgg", "delMontague"]
        for model in pywr_models:
            # ### first with full observational record
            plot_3part_flows_hier(
                reservoir_downstream_gages,
                major_flows,
                nodes,
                [model.replace("pywr_", ""), model],
                uselog=True,
                units=units,
                colordict=model_colors_diagnostics_paper2,
                start_date=start_date_obs,
                end_date=end_date_obs,
            )
            ### now zoomed in 2 year period where it is easier to see time series
            plot_3part_flows_hier(
                reservoir_downstream_gages,
                major_flows,
                nodes,
                [model.replace("pywr_", ""), model],
                uselog=True,
                units=units,
                colordict=model_colors_diagnostics_paper2,
                start_date=start_date_short_obs,
                end_date=end_date_short_obs,
            )

    ### compare modeled vs observed NYC storages
    if rerun_all:
        print("\nPlotting NYC storage modeled vs observed\n")
        plot_combined_nyc_storage(
            storages,
            ffmp_level_boundaries,
            pywr_models,
            colordict=model_colors_diagnostics_paper3,
            start_date=start_date_obs,
            end_date=end_date_obs,
            fig_dir=fig_dir,
            units=units,
        )
        plot_combined_nyc_storage(
            storages,
            ffmp_level_boundaries,
            pywr_models,
            colordict=model_colors_diagnostics_paper3,
            start_date=start_date_storage_obs,
            end_date=end_date_full,
            fig_dir=fig_dir,
            units=units,
        )
        plot_combined_nyc_storage(
            storages,
            ffmp_level_boundaries,
            pywr_models,
            colordict=model_colors_diagnostics_paper3,
            start_date=start_date_full,
            end_date=end_date_full,
            fig_dir=fig_dir,
            units=units,
        )

    ### gridded error metrics for NYC reservoirs + major flows
    if rerun_all:
        print("\nPlotting gridded error metrics\n")
        ### first do smaller subset of models & locations for main text
        nodes = ["cannonsville", "NYCAgg", "delMontague"]
        error_models = base_models[1:] + pywr_models
        node_metrics = get_error_metrics(
            reservoir_downstream_gages,
            major_flows,
            error_models,
            nodes,
            start_date=start_date_obs,
            end_date=end_date_obs,
        )
        plot_gridded_error_metrics(
            node_metrics,
            error_models,
            nodes,
            start_date=start_date_obs,
            end_date=end_date_obs,
            figstage=0,
        )
        ### now full set of models & locations, for SI
        nodes = reservoir_list_nyc + ["NYCAgg", "delMontague", "delTrenton"]
        node_metrics = get_error_metrics(
            reservoir_downstream_gages,
            major_flows,
            error_models,
            nodes,
            start_date=start_date_obs,
            end_date=end_date_obs,
        )
        plot_gridded_error_metrics(
            node_metrics,
            error_models,
            nodes,
            start_date=start_date_obs,
            end_date=end_date_obs,
            figstage=1,
        )
        ### repeat full figure but with monthly time step
        plot_gridded_error_metrics(
            node_metrics,
            error_models,
            nodes,
            start_date=start_date_obs,
            end_date=end_date_obs,
            figstage=2,
        )

    ### gridded low flow exceedances based on full record
    if rerun_all:
        print("\nPlotting low flow return period figs\n")
        models = base_models[1:] + pywr_models
        nodes = ["NYCAgg", "delMontague"]
        plot_lowflow_exceedances(
            reservoir_downstream_gages,
            major_flows,
            lower_basin_mrf_contributions,
            models,
            nodes,
            start_date=start_date_full,
            end_date=end_date_full,
            colordict=model_colors_diagnostics_paper3,
            figstage=0,
            units=units,
        )
        nodes = ["cannonsville", "NYCAgg", "delMontague", "delTrenton"]
        plot_lowflow_exceedances(
            reservoir_downstream_gages,
            major_flows,
            lower_basin_mrf_contributions,
            models,
            nodes,
            start_date=start_date_full,
            end_date=end_date_full,
            colordict=model_colors_diagnostics_paper3,
            figstage=1,
            units=units,
        )

    ### plot breaking down NYC flows & Trenton flows into components
    if rerun_all:
        print(
            "\nPlotting NYC releases by components, combined with downstream flow components\n"
        )
        model = "nwmv21_withObsScaled"
        plot_NYC_release_components_combined(
            storages,
            ffmp_level_boundaries,
            nyc_release_components,
            lower_basin_mrf_contributions,
            reservoir_releases,
            reservoir_downstream_gages,
            major_flows,
            inflows,
            ibt_diversions,
            catchment_consumptions,
            model,
            figstage=0,
            colordict=model_colors_diagnostics_paper3,
            use_log=True,
            use_observed=False,
            start_date=start_date_short_obs,
            end_date=end_date_short_obs,
            fig_dir=fig_dir,
            units=units,
        )
        for model in base_models[1:]:
            plot_NYC_release_components_combined(
                storages,
                ffmp_level_boundaries,
                nyc_release_components,
                lower_basin_mrf_contributions,
                reservoir_releases,
                reservoir_downstream_gages,
                major_flows,
                inflows,
                ibt_diversions,
                catchment_consumptions,
                model,
                figstage=1,
                colordict=model_colors_diagnostics_paper3,
                use_log=True,
                use_observed=False,
                start_date=start_date_short_obs,
                end_date=end_date_short_obs,
                fig_dir=fig_dir,
                units=units,
            )

            plot_NYC_release_components_combined(
                storages,
                ffmp_level_boundaries,
                nyc_release_components,
                lower_basin_mrf_contributions,
                reservoir_releases,
                reservoir_downstream_gages,
                major_flows,
                inflows,
                ibt_diversions,
                catchment_consumptions,
                model,
                figstage=1,
                colordict=model_colors_diagnostics_paper3,
                use_log=True,
                use_observed=False,
                start_date=start_date_full,
                end_date=end_date_full,
                fig_dir=fig_dir,
                units=units,
            )

    ### companion figure to the flow component figure to show timeseries of ratio between unmanaged & managed flows
    if rerun_all:
        plot_ratio_managed_unmanaged(
            lower_basin_mrf_contributions,
            reservoir_releases,
            reservoir_downstream_gages,
            major_flows,
            inflows,
            ibt_diversions,
            catchment_consumptions,
            model,
            colordict=model_colors_diagnostics_paper3,
            use_log=True,
            start_date=start_date_short_obs,
            end_date=end_date_short_obs,
            fig_dir=fig_dir,
            units=units,
        )

    ### plot shortfall event metrics (reliability/duration/intensity/vulnerability) distributions for min flows and ibt diversions
    if rerun_all:
        print(
            "\nPlotting shortfall event metrics for min flows and NYC/NJ diversions\n"
        )
        models_mrf = base_models[1:] + pywr_models
        models_ibt = pywr_models
        nodes = ["delMontague", "delTrenton", "nyc", "nj"]
        shortfall_type = "absolute"

        shortfall_metrics = get_shortfall_metrics(
            major_flows,
            lower_basin_mrf_contributions,
            mrf_targets,
            ibt_demands,
            ibt_diversions,
            models_mrf,
            models_ibt,
            nodes,
            shortfall_threshold=0.99,
            shortfall_break_length=30,
            units=units,
            start_date=start_date_medium_preobs,
            end_date=end_date_medium_preobs,
        )
        plot_shortfall_metrics(
            shortfall_metrics,
            models_mrf,
            models_ibt,
            nodes,
            colordict=model_colors_diagnostics_paper3,
            units=units,
            print_reliabilities=False,
            print_events=True,
        )
        shortfall_metrics = get_shortfall_metrics(
            major_flows,
            lower_basin_mrf_contributions,
            mrf_targets,
            ibt_demands,
            ibt_diversions,
            models_mrf,
            models_ibt,
            nodes,
            shortfall_threshold=0.99,
            shortfall_break_length=30,
            units=units,
            start_date=start_date_full,
            end_date=end_date_full,
        )
        plot_shortfall_metrics(
            shortfall_metrics,
            models_mrf,
            models_ibt,
            nodes,
            colordict=model_colors_diagnostics_paper3,
            units=units,
            print_reliabilities=True,
            print_events=False,
        )

    ### show NYC storage vs min flow satisfaction dynamics
    if rerun_all:
        print("\nPlotting NYC storages vs Montague/Trenton min flow targets\n")
        base_model = "nwmv21_withObsScaled"
        plot_combined_nyc_storage_vs_minflows(
            storages,
            ffmp_level_boundaries,
            major_flows,
            lower_basin_mrf_contributions,
            mrf_targets,
            reservoir_releases,
            downstream_release_targets,
            base_model,
            shortfall_metrics,
            figstage=0,
            colordict=model_colors_diagnostics_paper3,
            units=units,
            start_date=start_date_medium_preobs,
            end_date=end_date_medium_preobs,
            fig_dir=fig_dir,
        )
        for base_model in base_models[
            1:
        ]:  ### should be a base model, and we will compare base vs pywr version
            plot_combined_nyc_storage_vs_minflows(
                storages,
                ffmp_level_boundaries,
                major_flows,
                lower_basin_mrf_contributions,
                mrf_targets,
                reservoir_releases,
                downstream_release_targets,
                base_model,
                shortfall_metrics,
                figstage=1,
                colordict=model_colors_diagnostics_paper3,
                units=units,
                start_date=start_date_full,
                end_date=end_date_full,
                fig_dir=fig_dir,
            )

    # ### show NYC storage vs diversion dynamics
    if rerun_all:
        print("\nPlotting NYC storages vs NYC/NJ diversions\n")
        customer = "nyc"
        models = pywr_models[-2:]
        plot_combined_nyc_storage_vs_diversion(
            storages,
            ffmp_level_boundaries,
            ibt_demands,
            ibt_diversions,
            models,
            customer,
            shortfall_metrics,
            colordict=model_colors_diagnostics_paper3,
            start_date=start_date_medium_preobs,
            end_date=end_date_medium_preobs,
            fig_dir=fig_dir,
            figstage=0,
            units=units,
        )
        for customer in ["nyc", "nj"]:
            plot_combined_nyc_storage_vs_diversion(
                storages,
                ffmp_level_boundaries,
                ibt_demands,
                ibt_diversions,
                pywr_models,
                customer,
                shortfall_metrics,
                colordict=model_colors_diagnostics_paper3,
                start_date=start_date_full,
                end_date=end_date_full,
                fig_dir=fig_dir,
                figstage=1,
                units=units,
            )

    ### Make DRB map
    if remake_map:
        print("\nMaking DRB map\n")
        make_DRB_map(fig_dir=fig_dir, units=units)

    print(f"Done! Check the {fig_dir} folder.")
