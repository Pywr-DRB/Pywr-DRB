"""
Diagnostic plots for manual STARFIT parameter tuning.

Conventions
-----------
- Observed data: black. Default parameters: gray dashed. Tuned parameters /
  tuned run: blue. Additional runs: orange, green (fixed by entity, not cycled).
- Storage plots are normalized by the parameter row's Adjusted_CAP_MG (the
  S_hat basis the STARFIT rule sees). For beltzvilleCombined the modified
  capacity (13,500 MG) is well below true combined storage, so observed
  fractions can exceed 1.0 - a reference line marks S_hat = 1.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.tuning_utils import (
    compute_nor_curves,
    compute_harmonic_release,
    get_effective_rmin_rmax,
    get_reservoir_params,
    get_obs_storage,
    get_obs_downstream_flow,
    get_sim_storage,
    get_sim_release,
    get_starfit_release_trace,
)

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

COLORS = {
    "obs": "black",
    "default": "0.45",
    "tuned": "#2166ac",
    "run2": "#e08214",
    "run3": "#1b7837",
}
_RUN_COLOR_ORDER = ["tuned", "run2", "run3"]

_MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def _style_doy_axis(ax):
    ax.set_xlim(1, 366)
    ax.set_xticks(_MONTH_STARTS)
    ax.set_xticklabels(_MONTH_LABELS)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def run_color(i):
    """Fixed color for the i-th simulation run."""
    return COLORS[_RUN_COLOR_ORDER[min(i, len(_RUN_COLOR_ORDER) - 1)]]


def savefig(fig, name):
    """Save a figure to the experiment figures/ directory."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


# --------------------------------------------------------------------------
# Static diagnostics (no simulation required)
# --------------------------------------------------------------------------
def plot_nor_envelope_doy(params, reservoir, obs_storage=None,
                          default_params=None, ax=None):
    """
    NOR envelope vs day-of-year with observed storage overlaid.

    Parameters
    ----------
    params : pd.Series
        Parameter row for the reservoir (tuned or default).
    reservoir : str
    obs_storage : pd.Series, optional
        Observed storage in MG (datetime index); plotted as per-year
        spaghetti + day-of-year median, normalized by params Adjusted_CAP_MG.
    default_params : pd.Series, optional
        If given, default NOR bounds are drawn dashed for comparison.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    cap = params["Adjusted_CAP_MG"]

    nor = compute_nor_curves(params)
    ax.fill_between(nor.index, nor["NORlo"], nor["NORhi"],
                    color=COLORS["tuned"], alpha=0.20, linewidth=0)
    ax.plot(nor.index, nor["NORhi"], color=COLORS["tuned"], lw=1.5)
    ax.plot(nor.index, nor["NORlo"], color=COLORS["tuned"], lw=1.5)

    if default_params is not None:
        nor_d = compute_nor_curves(default_params)
        ax.plot(nor_d.index, nor_d["NORhi"], color=COLORS["default"], lw=1.2, ls="--")
        ax.plot(nor_d.index, nor_d["NORlo"], color=COLORS["default"], lw=1.2, ls="--")

    ymax = 1.05
    if obs_storage is not None and len(obs_storage) > 0:
        frac = obs_storage / cap
        for _, grp in frac.groupby(frac.index.year):
            ax.plot(grp.index.dayofyear, grp.values,
                    color="0.6", lw=0.4, alpha=0.35, zorder=1)
        med = frac.groupby(frac.index.dayofyear).median()
        ax.plot(med.index, med.values, color=COLORS["obs"], lw=1.8, zorder=3)
        ymax = max(ymax, float(np.nanquantile(frac.values, 0.995)) * 1.05)

    if ymax > 1.05:
        ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
        ax.text(4, 1.0, f" S_hat=1 ({cap:,.0f} MG)", va="bottom", fontsize=7, color="0.3")

    ax.set_ylim(0, ymax)
    ax.set_ylabel("Storage / Adjusted_CAP_MG")
    ax.set_title(reservoir, fontsize=10)
    _style_doy_axis(ax)
    return ax


def plot_release_harmonic_doy(params, reservoir, obs_downstream=None,
                              default_params=None, ax=None, logy=True):
    """
    Seasonal release harmonic vs day-of-year, with effective R_min/R_max and
    observed downstream-gage climatology (median + IQR) when available.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    if obs_downstream is not None and len(obs_downstream) > 0:
        by_doy = obs_downstream.groupby(obs_downstream.index.dayofyear)
        q25, q50, q75 = by_doy.quantile(0.25), by_doy.median(), by_doy.quantile(0.75)
        ax.fill_between(q50.index, q25.values, q75.values,
                        color="0.75", alpha=0.5, linewidth=0, label="obs IQR")
        ax.plot(q50.index, q50.values, color=COLORS["obs"], lw=1.5, label="obs median")

    if default_params is not None:
        rel_d = compute_harmonic_release(default_params)
        ax.plot(rel_d.index, rel_d.values, color=COLORS["default"], lw=1.2,
                ls="--", label="default harmonic")

    rel = compute_harmonic_release(params)
    ax.plot(rel.index, rel.values, color=COLORS["tuned"], lw=1.8, label="harmonic")

    r_min, r_max = get_effective_rmin_rmax(params, reservoir)
    ax.axhline(r_min, color=COLORS["tuned"], lw=0.8, ls=":")
    ax.axhline(r_max, color=COLORS["tuned"], lw=0.8, ls=":")
    ax.text(4, r_min, " R_min", va="bottom", fontsize=7, color=COLORS["tuned"])
    ax.text(4, r_max, " R_max", va="bottom", fontsize=7, color=COLORS["tuned"])

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("Release (MGD)")
    ax.set_title(reservoir, fontsize=10)
    _style_doy_axis(ax)
    return ax


def plot_static_diagnostics(params_df, reservoirs, obs_data,
                            default_params_df=None, start=None, end=None):
    """
    Grid: NOR envelope (top row) + release harmonic (bottom row) per reservoir.
    """
    n = len(reservoirs)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 6.5))
    axes = np.atleast_2d(axes)
    for j, res in enumerate(reservoirs):
        params = get_reservoir_params(params_df, res)
        default = (
            get_reservoir_params(default_params_df, res)
            if default_params_df is not None else None
        )
        obs_sto = get_obs_storage(obs_data, res, start, end)
        obs_flow = get_obs_downstream_flow(obs_data, res, start, end)
        plot_nor_envelope_doy(params, res, obs_sto, default, ax=axes[0, j])
        plot_release_harmonic_doy(params, res, obs_flow, default, ax=axes[1, j])
        if j > 0:
            axes[0, j].set_ylabel("")
            axes[1, j].set_ylabel("")
    handles = [
        plt.Line2D([], [], color=COLORS["obs"], lw=1.8, label="observed (median)"),
        plt.Line2D([], [], color=COLORS["tuned"], lw=1.8, label="current params"),
    ]
    if default_params_df is not None:
        handles.append(plt.Line2D([], [], color=COLORS["default"], lw=1.2,
                                  ls="--", label="default params"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Offline quick-sim diagnostics
# --------------------------------------------------------------------------
def plot_offline_sim(offline_results, params_df, obs_data, reservoirs,
                     start=None, end=None):
    """
    Offline STARFIT sim vs observed: storage fraction timeseries per reservoir.
    """
    n = len(reservoirs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, reservoirs):
        params = get_reservoir_params(params_df, res)
        cap = params["Adjusted_CAP_MG"]

        sim = offline_results[res].loc[start:end]
        nor = compute_nor_curves(params)
        doy_idx = sim.index.dayofyear.values - 1
        ax.fill_between(sim.index,
                        nor["NORlo"].values[doy_idx],
                        nor["NORhi"].values[doy_idx],
                        color=COLORS["tuned"], alpha=0.12, linewidth=0)

        obs = get_obs_storage(obs_data, res, start, end)
        if len(obs) > 0:
            ax.plot(obs.index, obs / cap, color=COLORS["obs"], lw=0.9, label="obs")
        ax.plot(sim.index, sim["storage"] / cap, color=COLORS["tuned"],
                lw=0.9, label="offline sim")
        ax.set_ylabel("S / S_cap")
        ax.set_title(res, fontsize=9, loc="left")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Full-run sim vs obs comparisons
# --------------------------------------------------------------------------
def plot_sim_vs_obs_storage(sim_data, run_labels, reservoirs, params_df,
                            obs_data, start=None, end=None):
    """
    Simulated vs observed storage fraction per reservoir; NOR band from
    params_df (the tuned parameter table).
    """
    n = len(reservoirs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, reservoirs):
        params = get_reservoir_params(params_df, res)
        cap = params["Adjusted_CAP_MG"]

        obs = get_obs_storage(obs_data, res, start, end)
        if len(obs) > 0:
            ax.plot(obs.index, obs / cap, color=COLORS["obs"], lw=0.9, label="obs")
        for i, label in enumerate(run_labels):
            sim = get_sim_storage(sim_data, label, res, start, end)
            ax.plot(sim.index, sim / cap, color=run_color(i), lw=0.9, label=label)
            if i == 0:
                nor = compute_nor_curves(params)
                doy_idx = sim.index.dayofyear.values - 1
                ax.fill_between(sim.index,
                                nor["NORlo"].values[doy_idx],
                                nor["NORhi"].values[doy_idx],
                                color=COLORS["tuned"], alpha=0.12, linewidth=0)
        ax.set_ylabel("S / S_cap")
        ax.set_title(res, fontsize=9, loc="left")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False, fontsize=8,
                   ncol=len(run_labels) + 1)
    fig.tight_layout()
    return fig


def plot_sim_vs_obs_release(sim_data, run_labels, reservoirs, obs_data,
                            start=None, end=None):
    """
    Release comparison per reservoir: day-of-year climatology (left) and flow
    duration curve (right). Shows total release (solid) and the
    starfit_release parameter trace (dotted) for each run, vs observed
    downstream gage flow where available.
    """
    n = len(reservoirs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n))
    axes = np.atleast_2d(axes)

    def _fdc(s):
        v = np.sort(s.dropna().values)[::-1]
        p = np.arange(1, len(v) + 1) / (len(v) + 1)
        return p, v

    for r, res in enumerate(reservoirs):
        ax_c, ax_f = axes[r, 0], axes[r, 1]
        obs = get_obs_downstream_flow(obs_data, res, start, end)
        if obs is not None and len(obs) > 0:
            med = obs.groupby(obs.index.dayofyear).median()
            ax_c.plot(med.index, med.values, color=COLORS["obs"], lw=1.5, label="obs")
            p, v = _fdc(obs)
            ax_f.plot(p, v, color=COLORS["obs"], lw=1.5, label="obs")

        for i, label in enumerate(run_labels):
            total = get_sim_release(sim_data, label, res, start, end)
            med = total.groupby(total.index.dayofyear).median()
            ax_c.plot(med.index, med.values, color=run_color(i), lw=1.3,
                      label=f"{label} total")
            p, v = _fdc(total)
            ax_f.plot(p, v, color=run_color(i), lw=1.3, label=f"{label} total")

            trace = get_starfit_release_trace(sim_data, label, res, start, end)
            if trace is not None:
                med_t = trace.groupby(trace.index.dayofyear).median()
                ax_c.plot(med_t.index, med_t.values, color=run_color(i), lw=1.1,
                          ls=":", label=f"{label} starfit")
                p, v = _fdc(trace)
                ax_f.plot(p, v, color=run_color(i), lw=1.1, ls=":",
                          label=f"{label} starfit")

        ax_c.set_yscale("log")
        ax_c.set_ylabel("Release (MGD)")
        ax_c.set_title(f"{res} - day-of-year median", fontsize=9, loc="left")
        _style_doy_axis(ax_c)

        ax_f.set_yscale("log")
        ax_f.set_title(f"{res} - flow duration", fontsize=9, loc="left")
        ax_f.set_xlabel("Exceedance probability")
        ax_f.grid(alpha=0.25, linewidth=0.5)
        ax_f.spines[["top", "right"]].set_visible(False)

    axes[0, 1].legend(loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
