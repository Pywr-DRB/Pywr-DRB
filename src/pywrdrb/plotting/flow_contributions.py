"""
Contains functions for plotting PywrDRB ensemble results.

Includes:
- plot_ensemble_nyc_storage

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

from pywrdrb.pywr_drb_node_data import (
    upstream_nodes_dict,
    downstream_node_lags,
    immediate_downstream_nodes_dict,
)

from pywrdrb.post.ensemble_metrics import ensemble_mean
from pywrdrb.plotting.styles import (
    model_label_dict,
    model_colors_historic_reconstruction,
)
from pywrdrb.plotting.styles import base_model_colors, get_model_color
from pywrdrb.plotting.styles import model_colors_historic_reconstruction
from pywrdrb.plotting.ensembles import plot_ensemble_percentile_cmap
from pywrdrb.plotting.nyc_storage import plot_nyc_storage

from pywrdrb.utils.constants import delTrenton_target, delMontague_target
from pywrdrb.utils.lists import (
    reservoir_list,
    reservoir_list_nyc,
    majorflow_list,
    drbc_lower_basin_reservoirs,
)
from pywrdrb.utils.directories import input_dir, fig_dir
from pywrdrb.utils.reservoir_data import get_reservoir_capacity
from pywrdrb.utils.timeseries import subset_timeseries


####################################################################


def plot_ensemble_NYC_release_contributions(
    model,
    nyc_release_components,
    reservoir_releases,
    reservoir_downstream_gages,
    colordict=model_colors_historic_reconstruction,
    plot_observed=True,
    plot_ensemble_mean=False,
    percentile_cmap=False,
    start_date=None,
    end_date=None,
    fig_dpi=200,
    fig_dir=fig_dir,
    fontsize=10,
    use_log=False,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    smoothing_window=1,
    ensemble_fill_alpha=1,
    contribution_fill_alpha=0.9,
    model_label_dict=model_label_dict,
    ax=None,
):
    # Calculate aggregate NYC downstream flow, if not already done
    if "NYCAgg" not in reservoir_downstream_gages[model][0].columns:
        ### Get aggregate NYC data
        for real in reservoir_downstream_gages[model].keys():
            reservoir_downstream_gages[model][real][
                "NYCAgg"
            ] = reservoir_downstream_gages[model][real][reservoir_list_nyc].sum(axis=1)

    model_is_ensemble = True if len(reservoir_releases[model].keys()) > 1 else False

    if model_is_ensemble:
        use_contribution_model = model + "_mean"
        colordict[use_contribution_model] = colordict[model]
        model_label_dict[use_contribution_model] = model_label_dict[model] + "Mean"

        # get ensemble mean for each dataset
        nyc_release_components[use_contribution_model] = {}
        reservoir_releases[use_contribution_model] = {}
        reservoir_downstream_gages[use_contribution_model] = {}

        nyc_release_components[use_contribution_model][0] = ensemble_mean(
            nyc_release_components[model]
        )
        reservoir_releases[use_contribution_model][0] = ensemble_mean(
            reservoir_releases[model]
        )
        reservoir_downstream_gages[use_contribution_model][0] = ensemble_mean(
            reservoir_downstream_gages[model]
        )
    else:
        use_contribution_model = model

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=fig_dpi)
        is_subplot = False
    else:
        is_subplot = True
    release_total = subset_timeseries(
        reservoir_releases[use_contribution_model][0][reservoir_list_nyc],
        start_date,
        end_date,
    ).sum(axis=1)

    # Handle when mrf requirements are greater than release
    total_flow_released = release_total.copy()
    for c in ["mrf_target_individual", "mrf_montagueTrenton", "flood_release", "spill"]:
        for r in reservoir_list_nyc:
            nyc_release_components[use_contribution_model][0][
                f"{c}_{r}"
            ] = nyc_release_components[use_contribution_model][0][f"{c}_{r}"].clip(
                lower=0, upper=total_flow_released
            )
            total_flow_released -= nyc_release_components[use_contribution_model][0][
                f"{c}_{r}"
            ]
    deficit = release_total - total_flow_released

    if np.isnan(release_total).any():
        print("Warning: NaNs in release_total.")

    x = release_total.index
    downstream_gage_pywr = subset_timeseries(
        reservoir_downstream_gages[use_contribution_model][0]["NYCAgg"],
        start_date,
        end_date,
    )
    if np.isnan(downstream_gage_pywr).any():
        print("Warning: NaNs in downstream_gage_pywr.")
        print(f"downstream_gage_pywr: {downstream_gage_pywr}")
    downstream_uncontrolled_pywr = downstream_gage_pywr - release_total

    realizations = list(reservoir_downstream_gages[model].keys())
    for i, real in enumerate(realizations):
        realization_downstream_gage_pywr = subset_timeseries(
            reservoir_downstream_gages[model][real][reservoir_list_nyc],
            start_date,
            end_date,
        ).sum(axis=1)
        if i == 0:
            ensemble_downstream_gage_pywr = pd.DataFrame(
                realization_downstream_gage_pywr,
                columns=[real],
                index=release_total.index,
            )
        else:
            ensemble_downstream_gage_pywr[real] = realization_downstream_gage_pywr
            # df = pd.DataFrame(realization_downstream_gage_pywr,
            #                   columns=[real],
            #                   index=release_total.index)
            # ensemble_downstream_gage_pywr = pd.concat((ensemble_downstream_gage_pywr, df), axis=1)

    ensemble_downstream_gage_pywr = ensemble_downstream_gage_pywr.rolling(
        smoothing_window, center=True
    ).mean()

    # If ensemble, use a fill plot
    if model_is_ensemble:
        if percentile_cmap:
            ax = plot_ensemble_percentile_cmap(
                ensemble_downstream_gage_pywr,
                model,
                ax,
                q_lower_bound=q_lower_bound,
                q_upper_bound=q_upper_bound,
                alpha=ensemble_fill_alpha,
                zorder=2,
            )
        else:
            ax.fill_between(
                ensemble_downstream_gage_pywr.index,
                ensemble_downstream_gage_pywr.quantile(q_lower_bound, axis=1),
                ensemble_downstream_gage_pywr.quantile(q_upper_bound, axis=1),
                color=colordict[model],
                alpha=0.85,
                zorder=4,
                lw=0.0,
                label=model_label_dict[model],
            )

    # If not ensemble, use a lineplot
    else:
        ax.plot(
            ensemble_downstream_gage_pywr,
            color=colordict[model],
            lw=2,
            zorder=4,
            label=model_label_dict[model],
        )

    if plot_observed:
        downstream_gage_obs = subset_timeseries(
            reservoir_downstream_gages["obs"]["NYCAgg"], start_date, end_date
        )
        downstream_gage_obs = downstream_gage_obs.rolling(
            smoothing_window, center=True
        ).mean()

        if len(downstream_gage_obs) > 0:
            ax.plot(
                downstream_gage_obs,
                color="k",
                ls="--",
                lw=1,
                label="Observed Flow",
                zorder=10,
            )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.94, -1.25),
        ncols=1,
        fontsize=fontsize,
    )

    ax.set_xlim([x[0], x[-1]])
    ax_twin = ax.twinx()
    ax_twin.set_ylim([0, 100])

    if use_log:
        ax.semilogy()
        ymax = downstream_gage_pywr.max()
        ymin = downstream_gage_pywr.min()
        if plot_observed:
            ymax = max(ymax, downstream_gage_obs.max())
            ymin = max(ymin, downstream_gage_obs.min())
        for i in range(10):
            if ymin < 10**i:
                ymin = 10 ** (i - 1)
                break
        for i in range(10):
            if ymax < 10**i:
                ymax = 10 ** (i)
                break
        ax.set_ylim([ymin, ymax])
    else:
        pass
        # ax.set_ylim([0, ax.get_ylim()[1]])

    ### colorbrewer brown/teal palette https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=4
    colors = [
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#d1e5f0",
        "#f6e8c3",
        "#dfc27d",
        "#bf812d",
        "#8c510a",
    ]

    release_components_full = subset_timeseries(
        nyc_release_components[use_contribution_model][0], start_date, end_date
    )

    release_types = [
        "mrf_target_individual",
        "mrf_montagueTrenton",
        "flood_release",
        "spill",
    ]
    release_components = pd.DataFrame(
        {
            release_type: release_components_full[
                [c for c in release_components_full.columns if release_type in c]
            ].sum(axis=1)
            for release_type in release_types
        }
    )
    release_components["uncontrolled"] = downstream_uncontrolled_pywr

    release_components = release_components.divide(downstream_gage_pywr, axis=0) * 100
    release_components.fillna(0, inplace=True)

    release_components = release_components.rolling(
        smoothing_window, center=True, axis=0
    ).mean()
    release_components.fillna(0, inplace=True)
    for c in release_types:
        if np.isnan(release_components[c]).any():
            print(
                f"Warning: NaNs in release components for {c} after divide and rolling."
            )
            print(f"downstream_gage_pywr: {downstream_gage_pywr}")

    y1 = np.zeros(len(release_components["uncontrolled"].values))
    y2 = y1 + release_components[f"uncontrolled"].values
    y3 = y2 + release_components[f"mrf_montagueTrenton"].values
    y4 = y3 + release_components[f"mrf_target_individual"].values
    y5 = y4 + release_components[f"flood_release"].values
    y6 = y5 + release_components[f"spill"].values
    for i, y in enumerate([y2, y3, y4, y5, y6]):
        if sum(np.isnan(y)) > 0:
            print(f"Warning: NaNs in release components for y{i+1}")
    # print(f'Max NYC contribution perc: {y6.max()}')
    ax_twin.fill_between(
        x,
        y5,
        y6,
        label="NYC Spill",
        color=colors[0],
        alpha=contribution_fill_alpha,
        lw=0,
        zorder=1,
    )
    ax_twin.fill_between(
        x,
        y4,
        y5,
        label="NYC FFMP Flood",
        color=colors[1],
        alpha=contribution_fill_alpha,
        lw=0,
        zorder=1,
    )
    ax_twin.fill_between(
        x,
        y3,
        y4,
        label="NYC FFMP Individual",
        color=colors[2],
        alpha=contribution_fill_alpha,
        lw=0,
        zorder=1,
    )
    ax_twin.fill_between(
        x,
        y2,
        y3,
        label="NYC FFMP Downstream",
        color=colors[3],
        alpha=contribution_fill_alpha,
        lw=0,
        zorder=1,
    )
    ax_twin.fill_between(
        x,
        y1,
        y2,
        label="Uncontrolled",
        color=colors[4],
        alpha=contribution_fill_alpha,
        lw=0,
        zorder=1,
    )

    ax.set_ylabel("NYC Release (MGD)", fontsize=fontsize)
    ax_twin.set_ylabel("Flow Contribution (%)", fontsize=fontsize)

    ax_twin.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax_twin.set_yticks(
        ax_twin.get_yticks(), ax_twin.get_yticklabels(), fontsize=fontsize
    )

    if is_subplot:
        return ax
    else:
        plt.show()
        return


def plot_ensemble_node_flow_contributions(
    model,
    node,
    major_flows,
    nyc_release_components,
    lower_basin_mrf_contributions,
    reservoir_releases,
    inflows,
    consumptions,
    diversions,
    colordict=model_colors_historic_reconstruction,
    plot_observed=True,
    plot_ensemble_mean=False,
    percentile_cmap=False,
    ensemble_fill_alpha=1,
    contribution_fill_alpha=0.9,
    start_date=None,
    end_date=None,
    fig_dpi=200,
    fig_dir=fig_dir,
    fontsize=10,
    use_log=False,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    smoothing_window=1,
    ax=None,
):
    model_is_ensemble = True if len(major_flows[model].keys()) > 1 else False

    if model_is_ensemble:
        use_contribution_model = model + "_mean"
        colordict[use_contribution_model] = colordict[model]
        model_label_dict[use_contribution_model] = model_label_dict[model] + " Mean"

        # get ensemble mean for each dataset
        major_flows[use_contribution_model] = {}
        nyc_release_components[use_contribution_model] = {}
        lower_basin_mrf_contributions[use_contribution_model] = {}
        reservoir_releases[use_contribution_model] = {}
        inflows[use_contribution_model] = {}
        consumptions[use_contribution_model] = {}
        diversions[use_contribution_model] = {}

        major_flows[use_contribution_model][0] = ensemble_mean(major_flows[model])
        nyc_release_components[use_contribution_model][0] = ensemble_mean(
            nyc_release_components[model]
        )
        lower_basin_mrf_contributions[use_contribution_model][0] = ensemble_mean(
            lower_basin_mrf_contributions[model]
        )
        reservoir_releases[use_contribution_model][0] = ensemble_mean(
            reservoir_releases[model]
        )
        inflows[use_contribution_model][0] = ensemble_mean(inflows[model])
        consumptions[use_contribution_model][0] = ensemble_mean(consumptions[model])
        diversions[use_contribution_model][0] = ensemble_mean(diversions[model])
    else:
        use_contribution_model = model

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=fig_dpi)
        is_subplot = False
    else:
        is_subplot = True

    # Get total sim and obs flow
    realizations = list(major_flows[model].keys())
    for i, real in enumerate(realizations):
        total_sim_node_flow = subset_timeseries(
            major_flows[model][real][node], start_date, end_date
        )

        if i == 0:
            ensemble_sim_node_flow = pd.DataFrame(
                columns=realizations, index=total_sim_node_flow.index
            )

        ### for Trenton, add NJ diversion to simulated flow. also add Blue Marsh MRF contribution for FFMP Trenton equivalent flow
        if node == "delTrenton":
            nj_diversion = subset_timeseries(
                diversions[model][real]["delivery_nj"], start_date, end_date
            )
            total_sim_node_flow += nj_diversion

            ### get drbc contributions from lower basin reservoirs
            realization_lower_basin_mrf_contributions = subset_timeseries(
                lower_basin_mrf_contributions[model][real], start_date, end_date
            )
            realization_lower_basin_mrf_contributions.columns = [
                c.split("_")[-1]
                for c in realization_lower_basin_mrf_contributions.columns
            ]

            # acct for lag at blue marsh so it can be added to trenton equiv flow. other flows lagged below
            if node == "delTrenton":
                for c in ["blueMarsh"]:
                    lag = downstream_node_lags[c]
                    downstream_node = immediate_downstream_nodes_dict[c]
                    while downstream_node != "output_del":
                        lag += downstream_node_lags[downstream_node]
                        downstream_node = immediate_downstream_nodes_dict[
                            downstream_node
                        ]
                    if lag > 0:
                        idx = realization_lower_basin_mrf_contributions.index
                        realization_lower_basin_mrf_contributions.loc[
                            idx[lag:], c
                        ] = realization_lower_basin_mrf_contributions.loc[:, c].shift(
                            lag
                        )
            total_sim_node_flow += realization_lower_basin_mrf_contributions[
                "blueMarsh"
            ]

            ensemble_sim_node_flow.loc[:, real] = total_sim_node_flow.values

    ensemble_sim_node_flow = ensemble_sim_node_flow.copy()
    ensemble_sim_node_flow = ensemble_sim_node_flow.rolling(
        smoothing_window, center=True
    ).mean()

    # If ensemble, use a fill plot
    if model_is_ensemble:
        if percentile_cmap:
            ax = plot_ensemble_percentile_cmap(
                ensemble_sim_node_flow,
                model,
                ax,
                q_lower_bound=q_lower_bound,
                q_upper_bound=q_upper_bound,
                alpha=ensemble_fill_alpha,
                zorder=2,
            )
        else:
            ax.fill_between(
                ensemble_sim_node_flow.index,
                ensemble_sim_node_flow.quantile(q_lower_bound, axis=1),
                ensemble_sim_node_flow.quantile(q_upper_bound, axis=1),
                color=colordict[model],
                alpha=ensemble_fill_alpha,
                zorder=2,
                lw=1.6,
                label=model_label_dict[model],
            )
    # If not ensemble, use a lineplot
    else:
        ax.plot(
            ensemble_sim_node_flow,
            color=colordict[model],
            lw=2,
            zorder=2,
            label=model_label_dict[model],
        )

    # repeat for contribution model to get background fill
    total_sim_node_flow = subset_timeseries(
        major_flows[use_contribution_model][0][node], start_date, end_date
    )

    ### for Trenton
    # add NJ diversion to simulated flow
    # also add Blue Marsh MRF contribution for FFMP Trenton equivalent flow
    if node == "delTrenton":
        nj_diversion = subset_timeseries(
            diversions[use_contribution_model][0]["delivery_nj"], start_date, end_date
        )
        total_sim_node_flow += nj_diversion

        ### get drbc contributions from lower basin reservoirs
        lower_basin_mrf_contributions = subset_timeseries(
            lower_basin_mrf_contributions[use_contribution_model][0],
            start_date,
            end_date,
        )
        lower_basin_mrf_contributions.columns = [
            c.split("_")[-1] for c in lower_basin_mrf_contributions.columns
        ]

        # acct for lag at blue marsh so it can be added to trenton equiv flow. other flows lagged below
        for c in ["blueMarsh"]:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != "output_del":
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = lower_basin_mrf_contributions.index
                lower_basin_mrf_contributions.loc[
                    idx[lag:], c
                ] = lower_basin_mrf_contributions.loc[:, c].shift(lag)
        total_sim_node_flow += lower_basin_mrf_contributions["blueMarsh"]

    if total_sim_node_flow.isna().any():
        print(f"WARNING: toal_sim_node_flow has NAs.")

    nyc_release_components = nyc_release_components.copy()

    for r in reservoir_list_nyc:
        total_release = reservoir_releases[use_contribution_model][0][r].copy()
        for c in [
            "mrf_target_individual",
            "mrf_montagueTrenton",
            "flood_release",
            "spill",
        ]:
            # Getting error: "cannot reindex on an axis with duplicate labels"
            total_release = total_release.loc[
                ~total_release.index.duplicated(keep="first")
            ]
            nyc_res_release = nyc_release_components[use_contribution_model][0][
                f"{c}_{r}"
            ]
            nyc_res_release = nyc_res_release.loc[
                ~nyc_res_release.index.duplicated(keep="first")
            ]
            nyc_res_release = nyc_res_release.reindex(total_release.index, fill_value=0)

            mrf_shortfall = total_release - nyc_res_release
            mrf_shortfall[mrf_shortfall >= 0] = 0
            total_release -= nyc_res_release
            total_release[total_release < 0] = 0

            nyc_release_components[use_contribution_model][0][f"{c}_{r}"] = (
                nyc_release_components[use_contribution_model][0][f"{c}_{r}"]
                + mrf_shortfall
            )

    # Plot observed flow
    if plot_observed:
        total_obs_node_flow = subset_timeseries(
            major_flows["obs"][node], start_date, end_date
        )
        if node == "delTrenton":
            nj_diversion = subset_timeseries(
                diversions[use_contribution_model][0]["delivery_nj"],
                start_date,
                end_date,
            )
            total_obs_node_flow += nj_diversion
        total_obs_node_flow = total_obs_node_flow.rolling(
            smoothing_window, center=True
        ).mean()

        if len(total_obs_node_flow) > 0:
            ax.plot(total_obs_node_flow, color="k", ls="--", lw=1, zorder=10)

    ax_twin = ax.twinx()
    ax_twin.set_ylim([0, 100])
    ax.set_xlim(start_date, end_date)
    if use_log:
        ax.semilogy()
        ymax = total_sim_node_flow.max()
        ymin = total_sim_node_flow.min()
        if plot_observed:
            ymax = max(ymax, total_obs_node_flow.max())
            ymin = max(ymin, total_obs_node_flow.min())
        for i in range(10):
            if ymin < 10**i:
                ymin = 10 ** (i - 1)
                break
        for i in range(10):
            if ymax < 10**i:
                ymax = 10 ** (i)
                break
        ax.set_ylim([ymin, ymax])
    else:
        pass
        # ax.set_ylim([0, ax.get_ylim()[1]])

    ax.set_ylabel(f"Total Flow (MGD)", fontsize=fontsize)
    ax_twin.set_ylabel("Flow Contribution (%)", fontsize=fontsize)

    # Get contributing flows
    contributing = upstream_nodes_dict[node]
    non_nyc_reservoirs = [
        i
        for i in contributing
        if (i in reservoir_list) and (i not in reservoir_list_nyc)
    ]
    non_nyc_release_contributions = reservoir_releases[use_contribution_model][0][
        non_nyc_reservoirs
    ]

    if node == "delTrenton":
        ### subtract lower basin ffmp releases from their non-ffmp releases
        for r in drbc_lower_basin_reservoirs:
            if r != "blueMarsh":
                non_nyc_release_contributions[r] = np.maximum(
                    non_nyc_release_contributions[r] - lower_basin_mrf_contributions[r],
                    0,
                )

    use_inflows = [i for i in contributing if (i in majorflow_list)]
    if node == "delMontague":
        use_inflows.append("delMontague")
    inflow_contributions = (
        inflows[use_contribution_model][0][use_inflows]
        - consumptions[use_contribution_model][0][use_inflows]
    )
    mrf_target_individuals = nyc_release_components[use_contribution_model][0][
        [
            c
            for c in nyc_release_components[use_contribution_model][0].columns
            if "mrf_target_individual" in c
        ]
    ]
    mrf_target_individuals.columns = [
        c.rsplit("_", 1)[1] for c in mrf_target_individuals.columns
    ]
    mrf_montagueTrentons = nyc_release_components[use_contribution_model][0][
        [
            c
            for c in nyc_release_components[use_contribution_model][0].columns
            if "mrf_montagueTrenton" in c
        ]
    ]
    mrf_montagueTrentons.columns = [
        c.rsplit("_", 1)[1] for c in mrf_montagueTrentons.columns
    ]
    flood_releases = nyc_release_components[use_contribution_model][0][
        [
            c
            for c in nyc_release_components[use_contribution_model][0].columns
            if "flood_release" in c
        ]
    ]
    flood_releases.columns = [c.rsplit("_", 1)[1] for c in flood_releases.columns]
    spills = nyc_release_components[use_contribution_model][0][
        [
            c
            for c in nyc_release_components[use_contribution_model][0].columns
            if "spill" in c
        ]
    ]
    spills.columns = [c.rsplit("_", 1)[1] for c in spills.columns]

    # Impose lag
    for c in upstream_nodes_dict[node][::-1]:
        if c in inflow_contributions.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = inflow_contributions.index
                inflow_contributions.loc[idx[lag:], c] = inflow_contributions.loc[
                    :, c
                ].shift(lag)
        elif c in non_nyc_release_contributions.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = non_nyc_release_contributions.index
                non_nyc_release_contributions.loc[
                    idx[lag:], c
                ] = non_nyc_release_contributions.loc[:, c].shift(lag)
                if node == "delTrenton" and c in drbc_lower_basin_reservoirs:
                    idx = lower_basin_mrf_contributions.index
                    lower_basin_mrf_contributions.loc[
                        idx[lag:], c
                    ] = lower_basin_mrf_contributions.loc[:, c].shift(lag)
                ### note: blue marsh lower_basin_mrf_contribution lagged above.
                # It wont show up in upstream_nodes_dict here, so not double lagging.
        elif c in mrf_target_individuals.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                idx = mrf_target_individuals.index
                mrf_target_individuals.loc[idx[lag:], c] = mrf_target_individuals.loc[
                    :, c
                ].shift(lag)
                mrf_montagueTrentons.loc[idx[lag:], c] = mrf_montagueTrentons.loc[
                    :, c
                ].shift(lag)
                flood_releases.loc[idx[lag:], c] = flood_releases.loc[:, c].shift(lag)
                spills.loc[idx[lag:], c] = spills.loc[:, c].shift(lag)

    print(f"Inflows from: {inflow_contributions.columns}")
    print(f"Non-NYC releases from: {non_nyc_release_contributions.columns}")
    if node == "delTrenton":
        print(
            f"Lower basin MRF contributions from: {lower_basin_mrf_contributions.columns}"
        )
    print(f"NYC FFMP target individuals from: {mrf_target_individuals.columns}")
    print(f"NYC FFMP montagueTrentons from: {mrf_montagueTrentons.columns}")
    print(f"NYC FFMP flood releases from: {flood_releases.columns}")
    print(f"NYC spills from: {spills.columns}")

    inflow_contributions = subset_timeseries(
        inflow_contributions, start_date, end_date
    ).sum(axis=1)
    non_nyc_release_contributions = subset_timeseries(
        non_nyc_release_contributions, start_date, end_date
    ).sum(axis=1)
    if node == "delTrenton":
        lower_basin_mrf_contributions = lower_basin_mrf_contributions.sum(axis=1)
    mrf_target_individuals = subset_timeseries(
        mrf_target_individuals, start_date, end_date
    ).sum(axis=1)
    mrf_montagueTrentons = subset_timeseries(
        mrf_montagueTrentons, start_date, end_date
    ).sum(axis=1)
    flood_releases = subset_timeseries(flood_releases, start_date, end_date).sum(axis=1)
    spills = subset_timeseries(spills, start_date, end_date).sum(axis=1)

    inflow_contributions = inflow_contributions.divide(total_sim_node_flow) * 100
    non_nyc_release_contributions = (
        non_nyc_release_contributions.divide(total_sim_node_flow) * 100
    )
    if node == "delTrenton":
        lower_basin_mrf_contributions = (
            lower_basin_mrf_contributions.divide(total_sim_node_flow) * 100
        )
    mrf_target_individuals = mrf_target_individuals.divide(total_sim_node_flow) * 100
    mrf_montagueTrentons = mrf_montagueTrentons.divide(total_sim_node_flow) * 100
    flood_releases = flood_releases.divide(total_sim_node_flow) * 100
    spills = spills.divide(total_sim_node_flow) * 100

    # Apply rolling smooth across dfs
    inflow_contributions = inflow_contributions.rolling(
        smoothing_window, center=True
    ).mean()
    non_nyc_release_contributions = non_nyc_release_contributions.rolling(
        smoothing_window, center=True
    ).mean()
    if node == "delTrenton":
        lower_basin_mrf_contributions = lower_basin_mrf_contributions.rolling(
            smoothing_window, center=True
        ).mean()
    mrf_target_individuals = mrf_target_individuals.rolling(
        smoothing_window, center=True
    ).mean()
    mrf_montagueTrentons = mrf_montagueTrentons.rolling(
        smoothing_window, center=True
    ).mean()
    flood_releases = flood_releases.rolling(smoothing_window, center=True).mean()
    spills = spills.rolling(smoothing_window, center=True).mean()

    colors = [
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#d1e5f0",
        "#f6e8c3",
        "#dfc27d",
        "#bf812d",
        "#8c510a",
    ]

    x = total_sim_node_flow.index
    y1 = 0
    y2 = y1 + inflow_contributions
    y3 = y2 + non_nyc_release_contributions
    if node == "delTrenton":
        y4 = y3 + lower_basin_mrf_contributions
        y5 = y4 + mrf_montagueTrentons
    else:
        y5 = y3 + mrf_montagueTrentons
    y6 = y5 + mrf_target_individuals
    y7 = y6 + flood_releases
    y8 = y7 + spills

    # Finally, plot flow contribution filled bands
    ax_twin.fill_between(
        x,
        y7,
        y8,
        label="NYC Spill",
        color=colors[0],
        alpha=contribution_fill_alpha,
        lw=0,
    )
    ax_twin.fill_between(
        x,
        y6,
        y7,
        label="NYC FFMP Flood",
        color=colors[1],
        alpha=contribution_fill_alpha,
        lw=0,
    )
    ax_twin.fill_between(
        x,
        y5,
        y6,
        label="NYC FFMP Individual",
        color=colors[2],
        alpha=contribution_fill_alpha,
        lw=0,
    )
    if node == "delTrenton":
        ax_twin.fill_between(
            x,
            y4,
            y5,
            label="NYC FFMP Downstream",
            color=colors[3],
            alpha=contribution_fill_alpha,
            lw=0,
        )
        ax_twin.fill_between(
            x,
            y3,
            y4,
            label="Non-NYC FFMP",
            color=colors[6],
            alpha=contribution_fill_alpha,
            lw=0,
        )
    else:
        ax_twin.fill_between(
            x,
            y3,
            y5,
            label="NYC FFMP Downstream",
            color=colors[3],
            alpha=contribution_fill_alpha,
            lw=0,
        )
    ax_twin.fill_between(
        x,
        y2,
        y3,
        label="Non-NYC Other",
        color=colors[5],
        alpha=contribution_fill_alpha,
        lw=0,
    )
    ax_twin.fill_between(
        x,
        y1,
        y2,
        label="Uncontrolled Flow",
        color=colors[4],
        alpha=contribution_fill_alpha,
        lw=0,
    )

    ax_twin.legend(
        frameon=False,
        fontsize=fontsize,
        loc="upper center",
        bbox_to_anchor=(0.37, -0.15),
        ncols=3,
    )

    ax_twin.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    # ax_twin.set_yticks(ax_twin.get_yticks(),
    #                    ax_twin.get_yticklabels(), fontsize=fontsize)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=fontsize)
    return (ax, ax_twin)


def plot_NYC_release_components_combined(
    storages,
    ffmp_level_boundaries,
    model,
    node,
    nyc_release_components,
    lower_basin_mrf_contributions,
    reservoir_releases,
    reservoir_downstream_gages,
    major_flows,
    inflows,
    diversions,
    consumptions,
    colordict=base_model_colors,
    start_date=None,
    end_date=None,
    use_log=False,
    plot_flow_target=False,
    plot_observed=False,
    fill_ffmp_levels=True,
    percentile_cmap=False,
    plot_ensemble_mean=False,
    ensemble_fill_alpha=1,
    contribution_fill_alpha=0.9,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    smoothing_window=1,
    fig_dir=fig_dir,
    fig_dpi=200,
    save_svg=False,
):
    fig, axs = plt.subplots(
        3, 1, figsize=(7, 7), gridspec_kw={"hspace": 0.1}, sharex=True
    )
    fontsize = 8
    labels = ["a)", "b)", "c)"]

    ########################################################
    ### subplot a: Reservoir modeled storages
    ########################################################

    ax1 = axs[0]

    ### subplot a: Reservoir modeled storages
    plot_nyc_storage(
        storages,
        ffmp_level_boundaries,
        models=[model, model.replace("_ensemble", "")],
        colordict=colordict,
        start_date=start_date,
        end_date=end_date,
        fig_dir=fig_dir,
        plot_observed=plot_observed,
        ax=ax1,
        fill_ffmp_levels=fill_ffmp_levels,
        fontsize=fontsize,
        percentile_cmap=percentile_cmap,
        plot_ensemble_mean=plot_ensemble_mean,
        ensemble_fill_alpha=ensemble_fill_alpha,
        smoothing_window=smoothing_window,
        dpi=fig_dpi,
        legend=False,
    )

    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.1,0.5), ncols=1, fontsize=fontsize)
    ax1.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncols=4,
        fontsize=fontsize,
    )
    ax1.annotate(
        labels[0],
        xy=(0.005, 0.975),
        xycoords="axes fraction",
        ha="left",
        va="top",
        weight="bold",
        fontsize=fontsize,
    )

    ########################################################
    # ### subfig b: first split up NYC releases into components
    ########################################################
    ax2 = axs[1]
    plot_ensemble_NYC_release_contributions(
        model=model,
        nyc_release_components=nyc_release_components,
        reservoir_releases=reservoir_releases,
        reservoir_downstream_gages=reservoir_downstream_gages,
        colordict=colordict,
        plot_observed=plot_observed,
        plot_ensemble_mean=plot_ensemble_mean,
        percentile_cmap=percentile_cmap,
        start_date=start_date,
        end_date=end_date,
        fig_dpi=fig_dpi,
        fig_dir=fig_dir,
        fontsize=fontsize,
        use_log=use_log,
        q_lower_bound=q_lower_bound,
        q_upper_bound=q_upper_bound,
        ensemble_fill_alpha=ensemble_fill_alpha,
        contribution_fill_alpha=contribution_fill_alpha,
        smoothing_window=smoothing_window,
        ax=ax2,
    )
    ax2.annotate(
        labels[1],
        xy=(0.005, 0.975),
        xycoords="axes fraction",
        ha="left",
        va="top",
        weight="bold",
        fontsize=fontsize,
    )

    ########################################################
    ### subfig c: split up montague/trenton flow into components
    ########################################################
    ax3 = axs[2]
    ax3, ax3_twin = plot_ensemble_node_flow_contributions(
        model,
        node,
        major_flows,
        nyc_release_components=nyc_release_components,
        lower_basin_mrf_contributions=lower_basin_mrf_contributions,
        reservoir_releases=reservoir_releases,
        inflows=inflows,
        consumptions=consumptions,
        diversions=diversions,
        colordict=colordict,
        plot_observed=plot_observed,
        plot_ensemble_mean=plot_ensemble_mean,
        percentile_cmap=percentile_cmap,
        ensemble_fill_alpha=ensemble_fill_alpha,
        contribution_fill_alpha=contribution_fill_alpha,
        start_date=start_date,
        end_date=end_date,
        fig_dpi=fig_dpi,
        fig_dir=fig_dir,
        fontsize=fontsize,
        use_log=use_log,
        q_lower_bound=q_lower_bound,
        q_upper_bound=q_upper_bound,
        smoothing_window=smoothing_window,
        ax=ax3,
    )
    ax3.annotate(
        labels[2],
        xy=(0.005, 0.975),
        xycoords="axes fraction",
        ha="left",
        va="top",
        weight="bold",
        fontsize=fontsize,
    )

    if plot_flow_target:
        target = delTrenton_target if node == "delTrenton" else delMontague_target
        ax3.hlines(
            target,
            start_date,
            end_date,
            lw=1,
            ls=":",
            color="k",
            label="Min. Flow Target",
        )

    ### Clean up figure
    plt.xlim(start_date, end_date)
    start_year = str(pd.to_datetime(start_date).year)
    end_year = str(pd.to_datetime(end_date).year)
    filename = (
        f"NYC_release_components_combined_{model}_{node}_"
        + f"{start_year}_{end_year}"
        + f'{"logscale" if use_log else ""}'
    )
    if save_svg:
        plt.savefig(f"{fig_dir}/{filename}.svg", bbox_inches="tight", dpi=fig_dpi)

    plt.savefig(f"{fig_dir}/{filename}.png", bbox_inches="tight", dpi=fig_dpi)

    return
