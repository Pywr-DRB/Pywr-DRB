"""
Contains functions for plotting PywrDRB ensemble results.

Includes:
- plot_ensemble_nyc_storage

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, LogNorm, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

import datetime as dt
import datetime


from pywrdrb.utils.lists import reservoir_list_nyc, drbc_lower_basin_reservoirs
from .plotting_functions import subset_timeseries
from pywrdrb.plotting.styles import (
    model_label_dict,
    model_colors_historic_reconstruction,
)

from pywrdrb.pywr_drb_node_data import (
    upstream_nodes_dict,
    downstream_node_lags,
    immediate_downstream_nodes_dict,
)


# Custom modules
from pywrdrb.utils.constants import delTrenton_target, delMontague_target
from pywrdrb.utils.lists import reservoir_list, reservoir_list_nyc, majorflow_list
from pywrdrb.utils.directories import input_dir, fig_dir, output_dir, model_data_dir
from pywrdrb.plotting.styles import base_model_colors, model_label_dict
from pywrdrb.plotting.styles import model_colors_historic_reconstruction
from pywrdrb.utils.reservoir_data import get_reservoir_capacity


def get_subplot_handles_and_labels(axs):
    # Gather legend handles and labels from all subplots and combine into single legend
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    # get only unique handles and labels
    handles, labels = np.array(handles), np.array(labels)
    idx = np.unique(labels, return_index=True)[1]
    handles, labels = handles[idx], labels[idx]
    return handles, labels


def create_mirrored_cmap(cmap_name):
    original_cmap = plt.cm.get_cmap(cmap_name)
    reversed_cmap = original_cmap.reversed()
    combined_colors = np.vstack(
        (original_cmap(np.linspace(0, 1, 128)), reversed_cmap(np.linspace(0, 1, 128)))
    )
    mirrored_cmap = mcolors.LinearSegmentedColormap.from_list(
        "mirrored_" + cmap_name, combined_colors
    )
    return mirrored_cmap


def clean_xtick_labels(
    axes,
    start_date,
    end_date,
    fontsize=10,
    date_format="%Y",
    max_ticks=10,
    rotate_labels=False,
):
    """
    Clean up x-axis tick labels for time series data.
    """
    try:
        start_date = (
            pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        )
        end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        if start_date >= end_date:
            raise ValueError(
                f"Start date must be before end date. Start: {start_date}, End: {end_date}"
            )

        total_days = (end_date - start_date).days

        if total_days <= 30:
            date_format = "%Y-%m-%d"
            tick_spacing = "D"
        elif total_days <= 365 * 2:
            date_format = "%Y-%m"
            tick_spacing = "MS"
        elif total_days <= 365 * 6:
            date_format = "%Y"
            tick_spacing = "1YS"
        elif total_days <= 365 * 10:
            date_format = "%Y"
            tick_spacing = "2YS"
        elif total_days <= 365 * 20:
            # Space every 5 years
            date_format = "%Y"
            tick_spacing = "5YS"
        else:
            # Space every 10 years
            date_format = "%Y"
            tick_spacing = "10YS"

        use_ticks = pd.date_range(start_date, end_date, freq=tick_spacing)
        tick_labels = [t.strftime(date_format) for t in use_ticks]

        for i in range(len(axes)):
            ax = axes[i]
            ax.set_xticks(use_ticks)
            ax.set_xticklabels(
                tick_labels,
                rotation=45 if rotate_labels else 0,
                fontsize=fontsize,
                ha="center",
            )
            ax.tick_params(axis="x", which="minor", length=0)
            ax.xaxis.set_minor_locator(plt.NullLocator())

            # Adjust layout to ensure labels are not cut off
            ax.figure.tight_layout()

    except Exception as e:
        print(f"Error in setting tick labels: {e}")

    return axes


####################################################################


def plot_ensemble_NYC_release_contributions(
    model,
    nyc_release_components,
    reservoir_releases,
    reservoir_downstream_gages,
    colordict=model_colors_historic_reconstruction,
    plot_observed=True,
    plot_ensemble_mean=False,
    start_date=None,
    end_date=None,
    fig_dpi=200,
    fig_dir=fig_dir,
    fontsize=10,
    use_log=False,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    ax=None,
):
    use_contribution_model = (
        model.split("_ensemble")[0] if "ensemble" in model else model
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=fig_dpi)
        is_subplot = False
    else:
        is_subplot = True
    release_total = subset_timeseries(
        reservoir_releases[use_contribution_model][reservoir_list_nyc],
        start_date,
        end_date,
    ).sum(axis=1)
    x = release_total.index
    downstream_gage_pywr = subset_timeseries(
        reservoir_downstream_gages[use_contribution_model]["NYCAgg"],
        start_date,
        end_date,
    )
    downstream_uncontrolled_pywr = downstream_gage_pywr - release_total

    if "ensemble" in model:
        realizations = list(reservoir_releases[model].keys())

        for i, real in enumerate(realizations):
            release_total = subset_timeseries(
                reservoir_releases[model][real][reservoir_list_nyc],
                start_date,
                end_date,
            ).sum(axis=1)
            if i == 0:
                ensemble_downstream_gage_pywr = pd.DataFrame(
                    release_total, columns=[real], index=release_total.index
                )
            else:
                ensemble_downstream_gage_pywr[real] = release_total

        # Fill between quantiles
        ax.fill_between(
            ensemble_downstream_gage_pywr.index,
            ensemble_downstream_gage_pywr.quantile(q_lower_bound, axis=1),
            ensemble_downstream_gage_pywr.quantile(q_upper_bound, axis=1),
            color=colordict[model],
            alpha=0.5,
            zorder=2,
            lw=0.0,
            label=model_label_dict[model],
        )

    ax.plot(downstream_gage_pywr, color="k", lw=1.7, zorder=3)
    ax.plot(
        downstream_gage_pywr,
        color=colordict[use_contribution_model],
        lw=1.4,
        label=f"{model_label_dict[use_contribution_model]} Flow",
        zorder=3.1,
    )

    if plot_observed:
        downstream_gage_obs = subset_timeseries(
            reservoir_downstream_gages["obs"]["NYCAgg"], start_date, end_date
        )

        if len(downstream_gage_obs) > 0:
            ax.plot(
                downstream_gage_obs,
                color="k",
                ls=":",
                lw=1.7,
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
    ax.set_ylim([0, 100])

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
    else:
        ax.set_ylim([0, ax.get_ylim()[1]])

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
    alpha = 1

    release_components_full = subset_timeseries(
        nyc_release_components[use_contribution_model], start_date, end_date
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

    y1 = 0
    y2 = y1 + release_components[f"uncontrolled"].values
    y3 = y2 + release_components[f"mrf_montagueTrenton"].values
    y4 = y3 + release_components[f"mrf_target_individual"].values
    y5 = y4 + release_components[f"flood_release"].values
    y6 = y5 + release_components[f"spill"].values
    ax.fill_between(x, y5, y6, label="NYC Spill", color=colors[0], alpha=alpha, lw=0)
    ax.fill_between(
        x, y4, y5, label="NYC FFMP Flood", color=colors[1], alpha=alpha, lw=0
    )
    ax.fill_between(
        x, y3, y4, label="NYC FFMP Individual", color=colors[2], alpha=alpha, lw=0
    )
    ax.fill_between(
        x, y2, y3, label="NYC FFMP Downstream", color=colors[3], alpha=alpha, lw=0
    )
    ax.fill_between(x, y1, y2, label="Uncontrolled", color=colors[4], alpha=alpha, lw=0)

    ax_twin.set_ylabel("NYC Release (MGD)", fontsize=fontsize)
    ax.set_ylabel("Flow Contribution (%)", fontsize=fontsize)

    ax_twin.set_zorder(1)
    ax_twin.set_zorder(2)
    ax_twin.patch.set_visible(False)

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax_twin.set_yticks(ax_twin.get_yticks(), ax.get_yticklabels(), fontsize=fontsize)
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
    start_date=None,
    end_date=None,
    fig_dpi=200,
    fig_dir=fig_dir,
    fontsize=10,
    use_log=False,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    ax=None,
):
    use_contribution_model = (
        model.split("_ensemble")[0] if "ensemble" in model else model
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 3), dpi=fig_dpi)
        is_subplot = False
    else:
        is_subplot = True

    # Get total sim and obs flow
    total_sim_node_flow = subset_timeseries(
        major_flows[use_contribution_model][node], start_date, end_date
    )
    if plot_observed:
        total_obs_node_flow = subset_timeseries(
            major_flows["obs"][node], start_date, end_date
        )

    ### for Trenton, add NJ diversion to simulated flow. also add Blue Marsh MRF contribution for FFMP Trenton equivalent flow
    if node == "delTrenton":
        nj_diversion = subset_timeseries(
            diversions[use_contribution_model]["delivery_nj"], start_date, end_date
        )
        total_sim_node_flow += nj_diversion
        # total_base_node_flow += nj_diversion
        if plot_observed:
            total_obs_node_flow += nj_diversion

        ### get drbc contributions from lower basin reservoirs
        lower_basin_mrf_contributions = subset_timeseries(
            lower_basin_mrf_contributions[use_contribution_model], start_date, end_date
        )
        lower_basin_mrf_contributions.columns = [
            c.split("_")[-1] for c in lower_basin_mrf_contributions.columns
        ]

        # acct for lag at blue marsh so it can be added to trenton equiv flow. other flows lagged below
        if node == "delTrenton":
            for c in ["blueMarsh"]:
                lag = downstream_node_lags[c]
                downstream_node = immediate_downstream_nodes_dict[c]
                while downstream_node != "output_del":
                    lag += downstream_node_lags[downstream_node]
                    downstream_node = immediate_downstream_nodes_dict[downstream_node]
                if lag > 0:
                    lower_basin_mrf_contributions[c].iloc[
                        lag:
                    ] = lower_basin_mrf_contributions[c].iloc[:-lag]

        total_sim_node_flow += lower_basin_mrf_contributions["blueMarsh"]

    ax.plot(total_sim_node_flow, color="k", lw=1.7)
    ax.plot(total_sim_node_flow, color=colordict[use_contribution_model], lw=1.4)
    if plot_observed:
        if len(total_obs_node_flow) > 0:
            ax.plot(total_obs_node_flow, color="k", ls=":", lw=1.7)
    ax_twin = ax.twinx()
    ax.set_ylim([0, 100])
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
    else:
        ax.set_ylim([0, ax.get_ylim()[1]])

    ax.set_ylabel(f"Total Flow (MGD)", fontsize=fontsize)
    ax_twin.set_ylabel("Flow Contribution (%)", fontsize=fontsize)

    # Get contributing flows
    contributing = upstream_nodes_dict[node]
    non_nyc_reservoirs = [
        i
        for i in contributing
        if (i in reservoir_list) and (i not in reservoir_list_nyc)
    ]
    non_nyc_release_contributions = reservoir_releases[use_contribution_model][
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
        inflows[use_contribution_model][use_inflows]
        - consumptions[use_contribution_model][use_inflows]
    )
    mrf_target_individuals = nyc_release_components[use_contribution_model][
        [
            c
            for c in nyc_release_components[use_contribution_model].columns
            if "mrf_target_individual" in c
        ]
    ]
    mrf_target_individuals.columns = [
        c.rsplit("_", 1)[1] for c in mrf_target_individuals.columns
    ]
    mrf_montagueTrentons = nyc_release_components[use_contribution_model][
        [
            c
            for c in nyc_release_components[use_contribution_model].columns
            if "mrf_montagueTrenton" in c
        ]
    ]
    mrf_montagueTrentons.columns = [
        c.rsplit("_", 1)[1] for c in mrf_montagueTrentons.columns
    ]
    flood_releases = nyc_release_components[use_contribution_model][
        [
            c
            for c in nyc_release_components[use_contribution_model].columns
            if "flood_release" in c
        ]
    ]
    flood_releases.columns = [c.rsplit("_", 1)[1] for c in flood_releases.columns]
    spills = nyc_release_components[use_contribution_model][
        [
            c
            for c in nyc_release_components[use_contribution_model].columns
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
                inflow_contributions[c].iloc[lag:] = inflow_contributions[c].iloc[:-lag]
        elif c in non_nyc_release_contributions.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                non_nyc_release_contributions[c].iloc[
                    lag:
                ] = non_nyc_release_contributions[c].iloc[:-lag]
                if node == "delTrenton" and c in drbc_lower_basin_reservoirs:
                    lower_basin_mrf_contributions[c].iloc[
                        lag:
                    ] = lower_basin_mrf_contributions[c].iloc[:-lag]
                ### note: blue marsh lower_basin_mrf_contribution lagged above.
                # It wont show up in upstream_nodes_dict here, so not double lagging.
        elif c in mrf_target_individuals.columns:
            lag = downstream_node_lags[c]
            downstream_node = immediate_downstream_nodes_dict[c]
            while downstream_node != node:
                lag += downstream_node_lags[downstream_node]
                downstream_node = immediate_downstream_nodes_dict[downstream_node]
            if lag > 0:
                mrf_target_individuals[c].iloc[lag:] = mrf_target_individuals[c].iloc[
                    :-lag
                ]
                mrf_montagueTrentons[c].iloc[lag:] = mrf_montagueTrentons[c].iloc[:-lag]
                flood_releases[c].iloc[lag:] = flood_releases[c].iloc[:-lag]
                spills[c].iloc[lag:] = spills[c].iloc[:-lag]

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
    alpha = 1

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
    ax.fill_between(x, y7, y8, label="NYC Spill", color=colors[0], alpha=alpha, lw=0)
    ax.fill_between(
        x, y6, y7, label="NYC FFMP Flood", color=colors[1], alpha=alpha, lw=0
    )
    ax.fill_between(
        x, y5, y6, label="NYC FFMP Individual", color=colors[2], alpha=alpha, lw=0
    )
    if node == "delTrenton":
        ax.fill_between(
            x, y4, y5, label="NYC FFMP Downstream", color=colors[3], alpha=alpha, lw=0
        )
        ax.fill_between(
            x, y3, y4, label="Non-NYC FFMP", color=colors[6], alpha=alpha, lw=0
        )
    else:
        ax.fill_between(
            x, y3, y5, label="NYC FFMP Downstream", color=colors[3], alpha=alpha, lw=0
        )
    ax.fill_between(
        x, y2, y3, label="Non-NYC Other", color=colors[5], alpha=alpha, lw=0
    )
    ax.fill_between(
        x, y1, y2, label="Uncontrolled Flow", color=colors[4], alpha=alpha, lw=0
    )

    ax.legend(
        frameon=False,
        fontsize=fontsize,
        loc="upper center",
        bbox_to_anchor=(0.37, -0.15),
        ncols=3,
    )

    ax_twin.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax_twin.set_yticks(
        ax_twin.get_yticks(), ax_twin.get_yticklabels(), fontsize=fontsize
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
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
    plot_observed=False,
    fill_ffmp_levels=True,
    percentiles_cmap=False,
    plot_ensemble_mean=False,
    ensemble_fill_alpha=0.8,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    fig_dir=fig_dir,
    fig_dpi=200,
):
    fig, axs = plt.subplots(
        3, 1, figsize=(7, 7), gridspec_kw={"hspace": 0.1}, sharex=True
    )
    fontsize = 8
    labels = ["a)", "b)", "c)"]

    ########################################################
    ### subplot a: Reservoir modeled storages
    ########################################################

    ax = axs[0]

    ### subplot a: Reservoir modeled storages
    plot_ensemble_nyc_storage(
        storages,
        ffmp_level_boundaries,
        models=model,
        colordict=colordict,
        start_date=start_date,
        end_date=end_date,
        fig_dir=fig_dir,
        plot_observed=plot_observed,
        ax=ax,
        fill_ffmp_levels=fill_ffmp_levels,
        fontsize=fontsize,
        percentiles_cmap=percentiles_cmap,
        plot_ensemble_mean=plot_ensemble_mean,
        ensemble_fill_alpha=ensemble_fill_alpha,
        dpi=fig_dpi,
        legend=False,
    )

    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.1,0.5), ncols=1, fontsize=fontsize)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncols=4,
        fontsize=fontsize,
    )
    ax.annotate(
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

    plot_ensemble_NYC_release_contributions(
        model=model,
        nyc_release_components=nyc_release_components,
        reservoir_releases=reservoir_releases,
        reservoir_downstream_gages=reservoir_downstream_gages,
        colordict=colordict,
        plot_observed=plot_observed,
        plot_ensemble_mean=plot_ensemble_mean,
        start_date=start_date,
        end_date=end_date,
        fig_dpi=fig_dpi,
        fig_dir=fig_dir,
        fontsize=fontsize,
        use_log=use_log,
        q_lower_bound=q_lower_bound,
        q_upper_bound=q_upper_bound,
        ax=None,
    )
    ax.annotate(
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

    plot_ensemble_node_flow_contributions(
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
        start_date=start_date,
        end_date=end_date,
        fig_dpi=fig_dpi,
        fig_dir=fig_dir,
        fontsize=fontsize,
        use_log=use_log,
        q_lower_bound=q_lower_bound,
        q_upper_bound=q_upper_bound,
        ax=axs[2],
    )
    ax.annotate(
        labels[2],
        xy=(0.005, 0.975),
        xycoords="axes fraction",
        ha="left",
        va="top",
        weight="bold",
        fontsize=fontsize,
    )

    ### Clean up figure
    plt.xlim(start_date, end_date)
    start_year = str(pd.to_datetime(start_date).year)
    end_year = str(pd.to_datetime(end_date).year)
    filename = (
        f"{fig_dir}NYC_release_components_combined_{model}_{node}_"
        + f"{start_year}_{end_year}"
        + f'{"logscale" if use_log else ""}'
        + ".png"
    )

    plt.savefig(filename, bbox_inches="tight", dpi=fig_dpi)


def plot_ensemble_nyc_storage(
    storages,
    ffmp_level_boundaries,
    models,
    colordict=model_colors_historic_reconstruction,
    start_date="1999-10-01",
    end_date="2010-05-31",
    fig_dir=fig_dir,
    plot_ensemble_mean=False,
    percentiles_cmap=False,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    fill_ffmp_levels=True,
    plot_observed=True,
    ax=None,
    legend=True,
    ensemble_fill_alpha=0.8,
    smoothing_window=1,
    fontsize=10,
    dpi=200,
):
    """ """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), dpi=dpi)
        is_subplot = False
    else:
        is_subplot = True

    ### get reservoir storage capacities
    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities["combined"] = sum([capacities[r] for r in reservoir_list_nyc])

    ffmp_level_boundaries = (
        subset_timeseries(ffmp_level_boundaries, start_date, end_date) * 100
    )
    ffmp_level_boundaries["level1a"] = 100.0

    ### First plot FFMP levels as background color
    levels = [f"level{l}" for l in ["1a", "1b", "1c", "2", "3", "4", "5"]]
    level_colors = (
        [cm.get_cmap("Blues")(v) for v in [0.3, 0.2, 0.1]]
        + ["papayawhip"]
        + [cm.get_cmap("Reds")(v) for v in [0.1, 0.2, 0.3]]
    )
    level_alpha = [1] * 3 + [1] + [1] * 3
    x = ffmp_level_boundaries.index

    if fill_ffmp_levels:
        for i in range(len(levels)):
            y0 = ffmp_level_boundaries[levels[i]]
            if i == len(levels) - 1:
                y1 = 0.0
            else:
                y1 = ffmp_level_boundaries[levels[i + 1]]
            ax.fill_between(
                x,
                y0,
                y1,
                color=level_colors[i],
                lw=0.2,
                edgecolor="k",
                alpha=level_alpha[i],
                zorder=1,
                label=levels[i],
            )

    # Or just do thr drought emergency level 5
    else:
        y = ffmp_level_boundaries["level5"]
        drought_color = "maroon"  # cm.get_cmap('Reds')(0.3)
        drought_threshold_ls = "-"

        # Fill with hatch
        ax.fill_between(
            x,
            [0.0] * len(y),
            y,
            facecolor="none",
            edgecolor=drought_color,
            linewidth=1.0,
            hatch="XX",
            alpha=0.2,
            zorder=1,
            label="Drought Emergency",
        )

    # Observed storage
    if plot_observed:
        historic_storage = pd.read_csv(
            f"{input_dir}/historic_NYC/NYC_storage_daily_2000-2021.csv",
            sep=",",
            index_col=0,
        )
        historic_storage.index = pd.to_datetime(historic_storage.index)
        historic_storage = subset_timeseries(
            historic_storage["Total"], start_date, end_date
        )
        historic_storage *= 100 / capacities["combined"]
        historic_storage = historic_storage.rolling(
            smoothing_window, center=True
        ).mean()
        ax.plot(
            historic_storage,
            color=colordict["obs"],
            ls=":",
            lw=2,
            label=model_label_dict["obs"],
            zorder=10,
        )

    line_colors = [colordict[m] for m in models]

    # Loop through models
    for m, c in zip(models, line_colors):
        if "ensemble" in m:
            # Get realization numbers
            realization_numbers = list(storages[m].keys())
            for i, real in enumerate(realization_numbers):
                modeled_storage = subset_timeseries(
                    storages[m][real][reservoir_list_nyc], start_date, end_date
                ).sum(axis=1)
                modeled_storage *= 100 / capacities["combined"]
                if i == 0:
                    ensemble_modeled_storage = pd.DataFrame(
                        modeled_storage, columns=[real], index=modeled_storage.index
                    )
                else:
                    ensemble_modeled_storage[real] = modeled_storage

                ensemble_modeled_storage = ensemble_modeled_storage.rolling(
                    smoothing_window, center=True
                ).mean()

            # Plot quantiles
            if percentiles_cmap:
                cmap = (
                    create_mirrored_cmap("Oranges")
                    if "nhm" in m
                    else create_mirrored_cmap("Blues")
                )
                norm = Normalize(vmin=-0.25, vmax=1.25)
                percentiles = np.linspace(0.01, 0.50, 50)[::-1]
                for i in range(len(percentiles) - 1):
                    ax.fill_between(
                        ensemble_modeled_storage.index,
                        ensemble_modeled_storage.quantile(percentiles[i + 1], axis=1),
                        ensemble_modeled_storage.quantile(percentiles[i], axis=1),
                        color=cmap(norm(percentiles[i])),
                        alpha=ensemble_fill_alpha,
                        zorder=2,
                        lw=0.0,
                    )
                    ax.fill_between(
                        ensemble_modeled_storage.index,
                        ensemble_modeled_storage.quantile(
                            1 - percentiles[i + 1], axis=1
                        ),
                        ensemble_modeled_storage.quantile(1 - percentiles[i], axis=1),
                        color=cmap(norm(percentiles[i])),
                        alpha=ensemble_fill_alpha,
                        zorder=2,
                        lw=0.0,
                    )

            else:
                ax.fill_between(
                    ensemble_modeled_storage.index,
                    ensemble_modeled_storage.quantile(q_lower_bound, axis=1),
                    ensemble_modeled_storage.quantile(q_upper_bound, axis=1),
                    color=c,
                    alpha=ensemble_fill_alpha,
                    zorder=2,
                    lw=1.6,
                    label=model_label_dict[m],
                )

            if plot_ensemble_mean:
                ax.plot(
                    ensemble_modeled_storage.mean(axis=1),
                    color=c,
                    ls="-",
                    zorder=4,
                    lw=1.6,
                )
                ax.plot(
                    ensemble_modeled_storage.mean(axis=1),
                    color="k",
                    ls="-",
                    zorder=3,
                    lw=2,
                )

        else:
            modeled_storage = subset_timeseries(
                storages[m][reservoir_list_nyc], start_date, end_date
            ).sum(axis=1)
            modeled_storage *= 100 / capacities["combined"]
            modeled_storage = modeled_storage.rolling(
                smoothing_window, center=True
            ).mean()
            ax.plot(modeled_storage, color="k", ls="-", zorder=5, lw=2)
            ax.plot(
                modeled_storage,
                color=c,
                ls="-",
                label=model_label_dict[m],
                zorder=6,
                lw=1.6,
            )

    ### clean up figure
    ax.set_xlim([start_date, end_date])
    ax.set_ylabel("Combined NYC Storage (%)", fontsize=fontsize)
    ax.set_ylim([0, 100])

    if not is_subplot:
        ax = clean_xtick_labels([ax], start_date, end_date, fontsize=fontsize)[0]
        ax.set_xlabel("Year", fontsize=fontsize)
        ax.legend(
            frameon=False,
            fontsize=fontsize,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.2),
            ncols=3,
        )

        plt.savefig(
            f'{fig_dir}ensemble_nyc_storage_{start_date.strftime("%Y")}_{end_date.strftime("%Y")}.png',
            dpi=dpi,
            bbox_inches="tight",
        )
    else:
        return ax


##########################################################################


def plot_ensemble_deficit(
    results,
    models,
    node,
    start_date,
    end_date,
    plot_ensemble_mean=False,
    plot_observed=True,
    fill_between_quantiles=True,
    percentiles_cmap=True,
    fill_alpha=0.5,
    model_colors=model_colors_historic_reconstruction,
    model_label_dict=model_label_dict,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    fontsize=10,
    dpi=200,
    smoothing_window=1,
    ax=None,
):
    target = delTrenton_target if node == "delTrenton" else delMontague_target

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3), dpi=200)
        is_subplot = False
    else:
        is_subplot = True

    for model in models:
        assert model in results.keys(), f"{model} not in results dict."
        if "ensemble" in model:
            assert (
                node in results[model][list(results[model].keys())[0]].columns
            ), f"{node} not in results."
        else:
            assert node in results[model].columns, f"{node} not in results."

    for model in models:
        if "ensemble" in model:
            # Get ensemble realization numbers
            ensemble_realizations = list(results[model].keys())

            # Re-arrange data from all realiztions into single dataframe
            for i, real in enumerate(ensemble_realizations):
                Q_sim = results[model][real][node].loc[start_date:end_date]

                percent_target = Q_sim / target * 100
                percent_target[percent_target > 100] = 100

                if i == 0:
                    df = pd.DataFrame(
                        percent_target.values,
                        columns=[real],
                        index=percent_target.index,
                    )
                else:
                    df[real] = percent_target

            # smooth
            df = df.rolling(smoothing_window, center=True).mean()

            # Plot ensemble mean
            if plot_ensemble_mean:
                ax.plot(
                    df.mean(axis=1), color=model_colors[model], ls="-", lw=1, zorder=4
                )
                ax.plot(df.mean(axis=1), color="k", ls="-", lw=1.5, zorder=3)
            # if fill_between_quantiles:

            if percentiles_cmap:
                percentiles = np.linspace(0.01, 0.99, 50)[::-1]
                cmap = (
                    create_mirrored_cmap("Oranges")
                    if "nhm" in model
                    else create_mirrored_cmap("Blues")
                )

                norm = Normalize(vmin=-0.25, vmax=1.25)
                for i in range(len(percentiles) - 1):
                    ax.fill_between(
                        df.index,
                        df.quantile(percentiles[i + 1], axis=1),
                        df.quantile(percentiles[i], axis=1),
                        color=cmap(norm(percentiles[i])),
                        alpha=fill_alpha,
                        zorder=2,
                        lw=0.0,
                    )
            else:
                ax.fill_between(
                    df.index,
                    df.quantile(q_lower_bound, axis=1),
                    df.quantile(q_upper_bound, axis=1),
                    color=model_colors[model],
                    alpha=fill_alpha,
                    zorder=2,
                    lw=1.6,
                    label=model_label_dict[model],
                )
        else:
            Q_sim = results[model][node].loc[start_date:end_date]
            percent_target = Q_sim / target * 100
            percent_target[percent_target > 100] = 100
            percent_target[percent_target < 0] = 0
            percent_target = percent_target.rolling(
                smoothing_window, center=True
            ).mean()
            ax.plot(
                percent_target,
                color=model_colors[model],
                ls="-",
                lw=1,
                zorder=6,
                label=model_label_dict[model],
            )
            ax.plot(percent_target, color="k", ls="-", lw=1.5, zorder=5)

    if plot_observed:
        obs_color = model_colors_historic_reconstruction["obs"]
        obs = results["obs"][node].loc[start_date:end_date]
        obs = obs / target * 100
        obs[obs > 100] = 100
        obs[obs < 0] = 0
        obs = obs.rolling(smoothing_window, center=True).mean()

        obs.plot(
            ax=ax,
            color="k",
            linewidth=2,
            label=model_label_dict["obs"],
            ls=":",
            zorder=10,
        )

    ax.set_ylim([0, 100])
    ax.set_ylabel("Target Flow\nSatisfied (%)", fontsize=fontsize)
    ax.set_xlim([start_date, end_date])

    if not is_subplot:
        # ax = clean_xtick_labels(ax, start_date, end_date, fontsize=fontsize)
        ax.set_xlabel("Year", fontsize=fontsize)
        ax.legend(
            frameon=False,
            fontsize=fontsize,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.19),
            ncols=3,
        )
    else:
        return ax


##########################################################################


def plot_ensemble_node_flow(
    major_flows,
    models,
    node,
    start_date,
    end_date,
    colordict=model_colors_historic_reconstruction,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    plot_observed=True,
    plot_ensemble_mean=True,
    plot_target=True,
    percentile_cmap=False,
    fontsize=10,
    fill_alpha=0.9,
    smoothing_window=7,
    logscale=False,
    fig_dir=fig_dir,
    dpi=200,
    ax=None,
):
    target = delTrenton_target if node == "delTrenton" else delMontague_target
    ylabel = f"Trenton\nFlow (MGD)" if node == "delTrenton" else f"Montague\nFlow (MGD)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        is_subplot = False
    else:
        is_subplot = True

    for model in models:
        if "ensemble" in model:
            realizations = list(major_flows[model].keys())
            for i, real in enumerate(realizations):
                modeled_flow = major_flows[model][real][node]
                modeled_flow = subset_timeseries(modeled_flow, start_date, end_date)

                if i == 0:
                    ensemble_flow = pd.DataFrame(
                        modeled_flow, columns=[real], index=modeled_flow.index
                    )
                else:
                    ensemble_flow[real] = modeled_flow

            ensemble_flow = ensemble_flow.rolling(smoothing_window, center=True).mean()
            if percentile_cmap:
                percentiles = np.linspace(q_lower_bound, q_upper_bound, 50)[::-1]
                mirrored_cmap = (
                    create_mirrored_cmap("Oranges")
                    if "nhm" in model
                    else create_mirrored_cmap("Blues")
                )
                norm = Normalize(vmin=-0.25, vmax=1.25)
                for i in range(len(percentiles) - 1):
                    ax.fill_between(
                        ensemble_flow.index,
                        ensemble_flow.quantile(percentiles[i], axis=1),
                        ensemble_flow.quantile(percentiles[i + 1], axis=1),
                        color=mirrored_cmap(norm(percentiles[i])),
                        lw=0.0,
                        alpha=fill_alpha,
                        zorder=2,
                    )
            else:
                ax.fill_between(
                    ensemble_flow.index,
                    ensemble_flow.quantile(q_lower_bound, axis=1),
                    ensemble_flow.quantile(q_upper_bound, axis=1),
                    color=colordict[model],
                    lw=0.0,
                    alpha=fill_alpha,
                    zorder=2,
                )
            if plot_ensemble_mean:
                ax.plot(
                    ensemble_flow.index,
                    ensemble_flow.mean(axis=1),
                    color="k",
                    lw=2,
                    zorder=3,
                )
                ax.plot(
                    ensemble_flow.index,
                    ensemble_flow.mean(axis=1),
                    color=colordict[model],
                    lw=1.5,
                    zorder=4,
                    ls="-",
                )

        else:
            modeled_flow = major_flows[model][node]
            modeled_flow = subset_timeseries(modeled_flow, start_date, end_date)
            modeled_flow = modeled_flow.rolling(smoothing_window, center=True).mean()
            ax.plot(
                modeled_flow.index,
                modeled_flow,
                color=colordict[model],
                lw=1.5,
                zorder=4,
                ls="-",
            )

    scale_ymax = 1 if logscale else 0.6
    use_ymax = modeled_flow.max() * scale_ymax

    if plot_observed:
        obs_flow = major_flows["obs"][node]
        obs_flow = subset_timeseries(obs_flow, start_date, end_date)
        obs_flow = obs_flow.rolling(smoothing_window, center=True).mean()
        ax.plot(obs_flow.index, obs_flow, color="k", lw=2, ls=":", zorder=7)
    if plot_target:
        ax.axhline(
            target, color="maroon", lw=2, ls="--", zorder=0, label="Min. Flow Target"
        )

    ax.set_xlim(start_date, end_date)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if logscale:
        ax.set_yscale("log")
    ax.set_ylim(0, use_ymax)

    if not is_subplot:
        start_str = start_date.strftime("%Y") if type(start_date) != str else start_date
        end_str = end_date.strftime("%Y") if type(end_date) != str else end_date

        ax = clean_xtick_labels([ax], start_date, end_date)[0]
        ax.set_xlabel("Date", fontsize=fontsize)

        fig.tight_layout()
        fig.savefig(
            fig_dir + f"{node}_ensemble_flow_{start_str}_{end_str}.png", dpi=dpi
        )

    else:
        return ax


##########################################################################


def plot_ensemble_nyc_storage_and_deficit(
    storages,
    major_flows,
    ffmp_level_boundaries,
    models,
    colordict=model_colors_historic_reconstruction,
    start_date="1999-10-01",
    end_date="2010-05-31",
    fig_dir=fig_dir,
    plot_ensemble_mean=False,
    percentiles_cmap=True,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    fill_ffmp_levels=True,
    plot_observed=True,
    legend=True,
    ensemble_fill_alpha=0.8,
    fontsize=10,
    dpi=200,
):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
        figsize=(8, 6),
        dpi=dpi,
    )

    ### subplot a: Reservoir modeled storages
    plot_ensemble_nyc_storage(
        storages,
        ffmp_level_boundaries,
        models=models,
        colordict=colordict,
        start_date=start_date,
        end_date=end_date,
        fig_dir=fig_dir,
        plot_observed=plot_observed,
        ax=ax1,
        fill_ffmp_levels=fill_ffmp_levels,
        fontsize=fontsize,
        percentiles_cmap=percentiles_cmap,
        plot_ensemble_mean=plot_ensemble_mean,
        ensemble_fill_alpha=ensemble_fill_alpha,
        dpi=dpi,
        legend=False,
    )

    ### subplot b: deficits at Montague
    plot_ensemble_deficit(
        major_flows,
        models=models,
        node="delMontague",
        start_date=start_date,
        end_date=end_date,
        plot_observed=plot_observed,
        ax=ax2,
        plot_ensemble_mean=plot_ensemble_mean,
        percentiles_cmap=percentiles_cmap,
        fontsize=fontsize,
        smoothing_window=7,
        fill_alpha=ensemble_fill_alpha,
        dpi=dpi,
    )
    ax2.set_ylabel("Montague Target\nSatisfied (%)", fontsize=fontsize)
    ax2.set_yticks([0, 100])

    ### subplot c: deficits at Trenton
    plot_ensemble_deficit(
        major_flows,
        models=models,
        node="delTrenton",
        start_date=start_date,
        end_date=end_date,
        plot_observed=plot_observed,
        ax=ax3,
        percentiles_cmap=percentiles_cmap,
        plot_ensemble_mean=plot_ensemble_mean,
        fontsize=fontsize,
        smoothing_window=7,
        fill_alpha=ensemble_fill_alpha,
        dpi=dpi,
    )
    ax3.set_ylabel("Trenton Target\nSatisfied (%)", fontsize=fontsize)
    ax3.set_yticks([0, 100])

    ax3.set_xlabel("Year", fontsize=fontsize)
    handles, labels = get_subplot_handles_and_labels([ax1, ax2, ax3])

    ax3.legend(
        handles,
        labels,
        frameon=False,
        fontsize=fontsize,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncols=3,
    )

    # Clean up figure
    axes = (ax1, ax2, ax3)
    for i in range(3):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_xticklabels([])
    axes = clean_xtick_labels(axes, start_date, end_date, fontsize=fontsize)
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.1)

    # Save
    start_str = start_date.strftime("%Y") if type(start_date) != str else start_date
    end_str = end_date.strftime("%Y") if type(end_date) != str else end_date
    fname = f"{fig_dir}/ensemble_nyc_storage_and_deficit_{start_str}_{end_str}"
    fname = fname + "_percentiles" if percentiles_cmap else fname
    fig.savefig(f"{fname}.png", dpi=dpi, bbox_inches="tight")

    return


##########################################################################


def plot_ensemble_nyc_storage_flow_deficit(
    storages,
    major_flows,
    ffmp_level_boundaries,
    models,
    node,
    colordict=model_colors_historic_reconstruction,
    start_date="1999-10-01",
    end_date="2010-05-31",
    fig_dir=fig_dir,
    plot_ensemble_mean=False,
    percentiles_cmap=True,
    q_lower_bound=0.05,
    q_upper_bound=0.95,
    fill_ffmp_levels=True,
    plot_observed=True,
    legend=True,
    ensemble_fill_alpha=0.8,
    fontsize=10,
    dpi=200,
):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
        figsize=(8, 6),
        dpi=dpi,
    )

    ### subplot a: Reservoir modeled storages
    plot_ensemble_nyc_storage(
        storages,
        ffmp_level_boundaries,
        models=models,
        colordict=colordict,
        start_date=start_date,
        end_date=end_date,
        fig_dir=fig_dir,
        plot_observed=plot_observed,
        ax=ax1,
        fill_ffmp_levels=fill_ffmp_levels,
        fontsize=fontsize,
        percentiles_cmap=percentiles_cmap,
        plot_ensemble_mean=plot_ensemble_mean,
        ensemble_fill_alpha=ensemble_fill_alpha,
        dpi=dpi,
        legend=False,
    )

    ### subplot b: total node flow
    plot_ensemble_node_flow(
        major_flows,
        models=models,
        node=node,
        start_date=start_date,
        end_date=end_date,
        plot_observed=plot_observed,
        ax=ax2,
        percentile_cmap=percentiles_cmap,
        plot_ensemble_mean=plot_ensemble_mean,
        fontsize=fontsize,
        smoothing_window=7,
        fill_alpha=ensemble_fill_alpha,
        logscale=True,
        dpi=dpi,
    )

    ### subplot c: deficits
    plot_ensemble_deficit(
        major_flows,
        models=models,
        node=node,
        start_date=start_date,
        end_date=end_date,
        plot_observed=plot_observed,
        ax=ax3,
        plot_ensemble_mean=plot_ensemble_mean,
        percentiles_cmap=percentiles_cmap,
        fontsize=fontsize,
        smoothing_window=7,
        fill_alpha=ensemble_fill_alpha,
        dpi=dpi,
    )
    ylabel = (
        f"Montague Target\nSatisfied (%)"
        if node == "delMontague"
        else f"Trenton Target\nSatisfied (%)"
    )
    ax3.set_ylabel(ylabel, fontsize=fontsize)
    ax3.set_yticks([0, 100])

    ax3.set_xlabel("Year", fontsize=fontsize)
    handles, labels = get_subplot_handles_and_labels([ax1, ax2, ax3])

    ax3.legend(
        handles,
        labels,
        frameon=False,
        fontsize=fontsize,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncols=3,
    )

    # Clean up figure
    axes = (ax1, ax2, ax3)
    for i in range(3):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_xticklabels([])
    axes = clean_xtick_labels(axes, start_date, end_date, fontsize=fontsize)
    fig.tight_layout()
    fig.align_ylabels()
    fig.subplots_adjust(hspace=0.1)

    # Save
    start_str = start_date.strftime("%Y") if type(start_date) != str else start_date
    end_str = end_date.strftime("%Y") if type(end_date) != str else end_date
    fname = f"{fig_dir}/ensemble_nyc_storage_and_{node}_flow_and_deficit_{start_str}_{end_str}"
    fname = fname + "_percentiles" if percentiles_cmap else fname
    fig.savefig(f"{fname}.png", dpi=dpi, bbox_inches="tight")
    return


#####################################################################################


def make_polar_plot(
    data,
    metric_names,
    ideal_score,
    filename,
    sub_title,
    metric_mins,
    r_max=1.5,
    inner_r=0.5,
    normalize=False,
    cmap="rainbow",
    color_by=0,
    brush_by=0,
    brush_condition="under",
    brush_threshold=1,
    brush_alpha=1,
    scale_ideal=False,
    plot_spokes=True,
    buffer=0.0,
    cut_negatives=True,
    show_legend=True,
    figsize=(10, 10),
    line_width=1,
    line_alpha=0.1,
):
    # Checks
    assert data.shape[1] == len(
        metric_names
    ), "Number of data columns != number of metric names."
    # assert(len(ideal_score) == len(metric_names)), 'Length of ideal scores != number of metric names.'

    n_obs, n_metrics = data.shape
    n_spokes = n_metrics
    theta = np.linspace(0, 2 * np.pi, n_spokes)

    # Find the minimum and maximum achieved objective values
    data_mins = data.min(axis=0)
    data_maxs = data.max(axis=0)

    # Create a normalized data set
    if normalize:
        scaler = MinMaxScaler()
        scale_model = scaler.fit(np.vstack((data, metric_mins)))
        norm_data = scale_model.transform(data)
        norm_ideal = scale_model.transform(ideal_score)
    else:
        norm_data = data.copy()
        norm_ideal = ideal_score.copy()

    # Remove color_by and add the 1st metric to end
    stacked_norm_data = np.hstack(
        (
            np.delete(norm_data, color_by, axis=1),
            np.delete(norm_data, color_by, axis=1)[:, 0:1],
        )
    )
    stacked_data = np.hstack(
        (np.delete(data, color_by, axis=1), np.delete(data, color_by, axis=1)[:, 0:1])
    )
    stacked_ideal = np.hstack(
        (
            np.delete(norm_ideal, color_by, axis=1),
            np.delete(norm_ideal, color_by, axis=1)[:, 0:1],
        )
    )

    # Define the radial data - scaled according to norms
    r_data = np.zeros_like(stacked_norm_data)

    if scale_ideal:
        shift_ideal = np.ones(n_metrics) / norm_ideal
        shift_ideal = np.hstack(
            (np.delete(shift_ideal, color_by), np.delete(shift_ideal, color_by)[0])
        )
        r_max = r_max + inner_r + buffer  # + max(shift_ideal)
        for i in range(n_spokes):
            if shift_ideal[i] > 0:
                r_data[:, i] = (
                    (stacked_norm_data[:, i]) * shift_ideal[i] + inner_r + buffer
                )
                stacked_ideal[:, i] = (
                    (stacked_ideal[:, i]) * shift_ideal[i] + inner_r + buffer
                )
            else:
                r_data[:, i] = (stacked_norm_data[:, i]) + 1 + inner_r + buffer
                stacked_ideal[:, i] = (stacked_ideal[:, i]) + 1 + inner_r + buffer

    else:
        r_data = stacked_norm_data + inner_r + buffer
        stacked_ideal = stacked_ideal + inner_r + buffer
        r_max = r_max + inner_r + buffer

    if cut_negatives:
        r_data[np.argwhere(r_data < 0)] = 0

    # Initialize plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    cmap = plt.cm.get_cmap(cmap)

    # Plot inner and outer ring
    ax.plot(theta, np.array([r_max] * n_spokes), color="grey", alpha=0.01)
    ax.plot(theta, np.array([inner_r] * n_spokes), color="grey", alpha=0.3)

    # Plot spokes
    if plot_spokes:
        for s in range(n_spokes):
            ax.plot(
                np.array([theta[s], theta[s]]),
                np.array([inner_r, max(r_data[:, s])]),
                color="grey",
                alpha=0.3,
            )
            ax.plot(
                np.array([theta[s], theta[s]]),
                np.array([max(r_data[:, s]), r_max]),
                color="grey",
                alpha=0.3,
                linestyle="dashed",
            )

    # Plot all observations
    brush_counter = 0
    for i in range(n_obs):
        if brush_condition == "under":
            brush_header = (
                f"Brush criteria: {metric_names[brush_by]} < {brush_threshold}"
            )
            if data[i, brush_by] <= brush_threshold:
                a = brush_alpha
                ci = cmap(norm_data[i, color_by])
                brush_counter += 1
            else:
                a = line_alpha
                ci = "k"
        elif brush_condition == "over":
            brush_header = (
                f"Brush criteria: {metric_names[brush_by]} > {brush_threshold}"
            )
            if data[i, brush_by] >= brush_threshold:
                a = brush_alpha
                ci = cmap(norm_data[i, color_by])
                brush_counter += 1
            else:
                a = line_alpha
                ci = "k"
        else:
            print('Invalid brush_condition. Options are "under" or "over".')

        ax.plot(theta, r_data[i, :], c=ci, linewidth=line_width, alpha=a)

    # Plot ideal
    ax.plot(
        theta,
        stacked_ideal[0, :],
        c="k",
        linewidth=2,
        linestyle="dashed",
        label="Ideal",
    )

    # Add colorbar
    cb = plt.cm.ScalarMappable(cmap=cmap)
    cb.set_array([data_mins[color_by], data_maxs[color_by]])
    cbar = fig.colorbar(cb, anchor=(2.5, 0), pad=0.05)
    cbar.ax.set_ylabel(metric_names[color_by], fontsize=16)

    # Add legend
    if show_legend == True:
        ax.legend(bbox_to_anchor=(1.2, 1))

    # Make radial labels
    spoke_maxs = np.max(stacked_data, axis=0)
    spoke_labs = np.delete(metric_names, color_by)
    # outter_radial_labels = [f'{spoke_maxs[i]}\n{spoke_labs[i]}' for i in range(len(spoke_labs))]
    outter_radial_labels = np.delete(metric_names, color_by)

    # Add text for brush condition
    brush_text = str(brush_header + f"\nn = {brush_counter} of {n_obs}")
    ax.text(
        (3 / 2) * np.pi,
        r_max + 0.4 * r_max,
        brush_text,
        verticalalignment="bottom",
        horizontalalignment="center",
        color="k",
        fontsize=14,
    )

    # Add lower bound values
    actual_minimums = np.min(r_data, axis=0)
    spoke_min_labels = np.delete(metric_mins, color_by)
    for s in range(n_spokes - 1):
        ax.text(
            theta[s],
            inner_r - 0.3 * inner_r,
            f"{spoke_min_labels[s]:.1f}",
            horizontalalignment="center",
            verticalalignment="center",
            color="k",
            fontsize=10,
        )

    # Graphic features
    ax.set_rmax(r_max)
    ax.set_rticks([])  # Less radial ticks
    ax.spines["polar"].set_visible(False)
    ax.set_rlabel_position(-50.5)  # Move radial labels away from plotted line
    ax.set_title(
        f"Prediction Performance Metrics\n{sub_title}", va="bottom", fontsize=15
    )
    ax.set_xticklabels(outter_radial_labels, fontsize=16)
    ax.set_xticks(theta)
    ax.grid(False)
    fig.set_size_inches(figsize)
    fig.set_dpi(200)
    plt.show()
    return plt
