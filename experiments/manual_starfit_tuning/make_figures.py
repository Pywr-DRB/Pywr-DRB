"""
Generate diagnostic figures comparing runs vs observations.

Usage:
    python make_figures.py run_label1 [run_label2 ...] [--tag NAME]

Saves PNGs to figures/ using the shared plotting functions plus a per-reservoir
storage+release overview panel.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.tuning_utils import (
    RESERVOIRS, load_observations, load_default_params, apply_overrides,
    get_starfit_row_name, get_obs_storage, get_obs_downstream_flow,
    load_sim_results, get_sim_storage, get_sim_release, compute_nor_curves,
    get_reservoir_params,
)
from evaluate_runs import get_obs_prompton_release, OBS_START, OBS_END

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def overview_figure(run_labels, params_df_by_run, tag,
                    start=OBS_START, end=OBS_END):
    """
    One PNG per reservoir: storage timeseries (top), release monthly means
    (middle), release flow-duration curve (bottom-left) and storage DOY
    climatology with NOR band (bottom-right).
    """
    obs = load_observations()
    default_params = load_default_params()
    sim = load_sim_results([OUTPUT_DIR / f"{lbl}.hdf5" for lbl in run_labels])

    for r in RESERVOIRS:
        cap = default_params.loc[get_starfit_row_name(r), "Adjusted_CAP_MG"]
        obs_s = get_obs_storage(obs, r, start, end)
        if r == "prompton":
            obs_q = get_obs_prompton_release(start, end)
        else:
            obs_q = get_obs_downstream_flow(obs, r, start, end)

        fig, axes = plt.subplots(2, 2, figsize=(16, 9),
                                 gridspec_kw={"height_ratios": [1.2, 1]})
        ax_s, ax_m = axes[0, 0], axes[0, 1]
        ax_fdc, ax_doy = axes[1, 0], axes[1, 1]

        # --- storage timeseries ---
        if len(obs_s):
            ax_s.plot(obs_s.index, obs_s / cap * 100, color="k", lw=1.2,
                      label="obs", zorder=5)
        for i, lbl in enumerate(run_labels):
            s = get_sim_storage(sim, lbl, r, start, end)
            ax_s.plot(s.index, s / cap * 100, color=COLORS[i % 4], lw=0.9,
                      label=lbl, alpha=0.9)
        ax_s.set_ylabel("storage (% cap)")
        ax_s.set_title(f"{r} — storage")
        ax_s.legend(fontsize=8)

        # --- monthly mean release ---
        if obs_q is not None and len(obs_q):
            om = obs_q.resample("MS").mean()
            ax_m.plot(om.index, om, color="k", lw=1.0, label="obs", zorder=5)
        for i, lbl in enumerate(run_labels):
            if r == "prompton":
                q = get_sim_release(sim, lbl, r, start, end)
            else:
                q = sim.reservoir_downstream_gage[lbl][0][r].copy()
                q.index = pd.to_datetime(q.index)
                q = q.loc[start:end]
            qm = q.resample("MS").mean()
            ax_m.plot(qm.index, qm, color=COLORS[i % 4], lw=0.9, label=lbl,
                      alpha=0.9)
        ax_m.set_ylabel("release / downstream flow (MGD)")
        ax_m.set_title(f"{r} — monthly mean flow")
        ax_m.set_yscale("log")
        ax_m.legend(fontsize=8)

        # --- flow duration curve (daily) ---
        import numpy as np
        if obs_q is not None and len(obs_q):
            q_sorted = np.sort(obs_q.values)[::-1]
            p = np.arange(1, len(q_sorted) + 1) / (len(q_sorted) + 1)
            ax_fdc.plot(p, q_sorted, color="k", lw=1.2, label="obs")
        for i, lbl in enumerate(run_labels):
            if r == "prompton":
                q = get_sim_release(sim, lbl, r, start, end)
            else:
                q = sim.reservoir_downstream_gage[lbl][0][r].copy()
                q.index = pd.to_datetime(q.index)
                q = q.loc[start:end]
            if obs_q is not None and len(obs_q):
                q = q.reindex(obs_q.index).dropna()
            q_sorted = np.sort(q.values)[::-1]
            p = np.arange(1, len(q_sorted) + 1) / (len(q_sorted) + 1)
            ax_fdc.plot(p, q_sorted, color=COLORS[i % 4], lw=1.0, label=lbl)
        ax_fdc.set_yscale("log")
        ax_fdc.set_xlabel("exceedance probability")
        ax_fdc.set_ylabel("daily flow (MGD)")
        ax_fdc.set_title("flow duration")
        ax_fdc.legend(fontsize=8)

        # --- storage DOY climatology + NOR bands ---
        if len(obs_s):
            doy_obs = (obs_s / cap * 100).groupby(obs_s.index.dayofyear)
            ax_doy.fill_between(doy_obs.mean().index, doy_obs.quantile(0.1),
                                doy_obs.quantile(0.9), color="k", alpha=0.15,
                                label="obs 10-90%")
            ax_doy.plot(doy_obs.mean().index, doy_obs.median(), color="k",
                        lw=1.4, label="obs median")
        for i, lbl in enumerate(run_labels):
            s = get_sim_storage(sim, lbl, r, start, end)
            doy_sim = (s / cap * 100).groupby(s.index.dayofyear)
            ax_doy.plot(doy_sim.median().index, doy_sim.median(),
                        color=COLORS[i % 4], lw=1.2, label=f"{lbl} median")
            pdf = params_df_by_run.get(lbl)
            if pdf is not None:
                nor = compute_nor_curves(get_reservoir_params(pdf, r))
                ax_doy.plot(nor.index, nor["NORhi"] * 100, color=COLORS[i % 4],
                            ls="--", lw=0.8, alpha=0.7)
                ax_doy.plot(nor.index, nor["NORlo"] * 100, color=COLORS[i % 4],
                            ls=":", lw=0.8, alpha=0.7)
        ax_doy.set_xlabel("day of year")
        ax_doy.set_ylabel("storage (% cap)")
        ax_doy.set_title("storage climatology (dashed=NORhi, dotted=NORlo)")
        ax_doy.legend(fontsize=7)

        fig.tight_layout()
        out = FIG_DIR / f"overview_{r}_{tag}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved {out}")


def _get_sim_flow(sim, lbl, r, start, end):
    """Simulated release (prompton) or downstream gage flow (others), MGD."""
    if r == "prompton":
        return get_sim_release(sim, lbl, r, start, end)
    q = sim.reservoir_downstream_gage[lbl][0][r].copy()
    q.index = pd.to_datetime(q.index)
    return q.loc[start:end]


def _get_obs_flow(obs, r, start, end):
    if r == "prompton":
        return get_obs_prompton_release(start, end)
    return get_obs_downstream_flow(obs, r, start, end)


def rule_structure_figure(run_label, params_df, tag, start=OBS_START, end=OBS_END):
    """
    One PNG per reservoir showing the STARFIT rule itself against observations:
    left = NOR band vs observed storage climatology; right = seasonal release
    harmonic vs observed release climatology, with R_min/R_max.
    """
    import numpy as np
    from utils.tuning_utils import compute_harmonic_release, get_effective_rmin_rmax

    obs = load_observations()
    sim = load_sim_results([OUTPUT_DIR / f"{run_label}.hdf5"])

    for r in RESERVOIRS:
        p = get_reservoir_params(params_df, r)
        cap = p["Adjusted_CAP_MG"]
        r_min, r_max = get_effective_rmin_rmax(p, r)
        drbc_fixed = r in ("fewalter", "blueMarsh", "beltzvilleCombined")

        fig, (ax_nor, ax_rel) = plt.subplots(1, 2, figsize=(14, 5))

        # --- left: NOR band + storage climatology ---
        nor = compute_nor_curves(p)
        ax_nor.fill_between(nor.index, nor["NORlo"] * 100, nor["NORhi"] * 100,
                            color="#1f77b4", alpha=0.20,
                            label="NOR (normal operating range)")
        ax_nor.plot(nor.index, nor["NORhi"] * 100, color="#1f77b4", lw=1.0)
        ax_nor.plot(nor.index, nor["NORlo"] * 100, color="#1f77b4", lw=1.0)

        obs_s = get_obs_storage(obs, r, start, end)
        if len(obs_s):
            doy = (obs_s / cap * 100).groupby(obs_s.index.dayofyear)
            ax_nor.fill_between(doy.median().index, doy.quantile(0.1),
                                doy.quantile(0.9), color="k", alpha=0.12,
                                label="obs 10-90%")
            ax_nor.plot(doy.median().index, doy.median(), color="k", lw=1.5,
                        label="obs median")
        s = get_sim_storage(sim, run_label, r, start, end)
        doy_s = (s / cap * 100).groupby(s.index.dayofyear)
        ax_nor.plot(doy_s.median().index, doy_s.median(), color="#d62728",
                    lw=1.2, label="sim median")
        ax_nor.set_xlabel("day of year")
        ax_nor.set_ylabel("storage (% of capacity)")
        ax_nor.set_title(f"{r} — NOR band vs observed storage"
                         f"  (capacity {cap:,.0f} MG)")
        ax_nor.legend(fontsize=8, loc="best")

        # --- right: release harmonic + climatology ---
        harm = compute_harmonic_release(p)  # in-NOR target at avg inflow
        harm = harm.clip(lower=r_min, upper=r_max)
        ax_rel.plot(harm.index, harm, color="#1f77b4", lw=1.6,
                    label="STARFIT seasonal release\n(in-NOR, average inflow)")

        obs_q = _get_obs_flow(obs, r, start, end)
        if obs_q is not None and len(obs_q):
            doy_q = obs_q.groupby(obs_q.index.dayofyear)
            ax_rel.fill_between(doy_q.median().index, doy_q.quantile(0.25),
                                doy_q.quantile(0.75), color="k", alpha=0.12,
                                label="obs 25-75%")
            ax_rel.plot(doy_q.median().index, doy_q.median(), color="k",
                        lw=1.5, label="obs median")
        sim_q = _get_sim_flow(sim, run_label, r, start, end)
        doy_sq = sim_q.groupby(sim_q.index.dayofyear)
        ax_rel.plot(doy_sq.median().index, doy_sq.median(), color="#d62728",
                    lw=1.2, label="sim median")

        src = "DRBC rule" if drbc_fixed else "parameter file"
        ax_rel.axhline(r_min, color="grey", ls="--", lw=1.0)
        ax_rel.annotate(f"R_min = {r_min:.0f} MGD ({src})",
                        xy=(3, r_min), fontsize=8, color="dimgrey",
                        va="bottom")
        if r_max < 3 * max(harm.max(), doy_sq.median().max()):
            ax_rel.axhline(r_max, color="grey", ls="--", lw=1.0)
            ax_rel.annotate(f"R_max = {r_max:.0f} MGD ({src})",
                            xy=(3, r_max), fontsize=8, color="dimgrey",
                            va="top")
        ax_rel.set_yscale("log")
        ax_rel.set_xlabel("day of year")
        ax_rel.set_ylabel("flow (MGD)")
        ax_rel.set_title(f"{r} — seasonal release rule vs observed flow"
                         f"  (mean inflow {p['Adjusted_MEANFLOW_MGD']:.0f} MGD)")
        ax_rel.legend(fontsize=8, loc="best")

        fig.tight_layout()
        out = FIG_DIR / f"rule_{r}_{tag}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved {out}")


def release_dynamics_figure(run_label, tag, start=OBS_START, end=OBS_END):
    """
    One PNG per reservoir on sub-monthly release behavior: monthly
    distributions of daily flow, storm-event composites, day-over-day
    variability, and an example high-flow window.
    """
    import numpy as np

    obs = load_observations()
    sim = load_sim_results([OUTPUT_DIR / f"{run_label}.hdf5"])

    for r in RESERVOIRS:
        obs_q = _get_obs_flow(obs, r, start, end)
        if obs_q is None or not len(obs_q):
            continue
        sim_q = _get_sim_flow(sim, run_label, r, start, end)
        idx = obs_q.index.intersection(sim_q.index)
        o, s = obs_q.loc[idx], sim_q.loc[idx]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        ax_mon, ax_evt = axes[0]
        ax_var, ax_win = axes[1]

        # --- (a) monthly distributions of daily flow ---
        for m in range(1, 13):
            om, smn = o[o.index.month == m], s[s.index.month == m]
            for series, x_off, color in [(om, -0.18, "k"), (smn, 0.18, "#d62728")]:
                q10, q25, q50, q75, q90 = series.quantile([.1, .25, .5, .75, .9])
                ax_mon.plot([m + x_off] * 2, [q10, q90], color=color, lw=1.0)
                ax_mon.plot([m + x_off] * 2, [q25, q75], color=color, lw=3.0)
                ax_mon.plot(m + x_off, q50, marker="o", ms=3.5, color=color)
        ax_mon.plot([], [], color="k", lw=3, label="obs (10-25-50-75-90%)")
        ax_mon.plot([], [], color="#d62728", lw=3, label="sim")
        ax_mon.set_yscale("log")
        ax_mon.set_xticks(range(1, 13))
        ax_mon.set_xticklabels(list("JFMAMJJASOND"))
        ax_mon.set_ylabel("daily flow (MGD)")
        ax_mon.set_title(f"{r} — monthly distributions of daily flow")
        ax_mon.legend(fontsize=8)

        # --- (b) storm-event composite (top 20 independent obs peaks) ---
        peaks = []
        o_sorted = o.sort_values(ascending=False)
        for t in o_sorted.index:
            if all(abs((t - tp).days) > 10 for tp in peaks):
                peaks.append(t)
            if len(peaks) >= 20:
                break
        lags = np.arange(-5, 11)
        comp_o = np.full((len(peaks), len(lags)), np.nan)
        comp_s = np.full((len(peaks), len(lags)), np.nan)
        for i, t in enumerate(peaks):
            for j, L in enumerate(lags):
                d = t + pd.Timedelta(days=int(L))
                if d in o.index:
                    comp_o[i, j] = o.loc[d]
                    comp_s[i, j] = s.loc[d]
        ax_evt.plot(lags, np.nanmedian(comp_o, axis=0), color="k", lw=1.6,
                    marker="o", ms=3, label="obs (median of the 20 events)")
        ax_evt.fill_between(lags, np.nanquantile(comp_o, 0.25, axis=0),
                            np.nanquantile(comp_o, 0.75, axis=0), color="k",
                            alpha=0.12, label="obs (25-75% of events)")
        ax_evt.plot(lags, np.nanmedian(comp_s, axis=0), color="#d62728",
                    lw=1.6, marker="o", ms=3,
                    label="sim (median of the same 20 days)")
        ax_evt.fill_between(lags, np.nanquantile(comp_s, 0.25, axis=0),
                            np.nanquantile(comp_s, 0.75, axis=0),
                            color="#d62728", alpha=0.12,
                            label="sim (25-75% of events)")
        ax_evt.set_yscale("log")
        ax_evt.axvline(0, color="grey", lw=0.8, ls=":")
        ax_evt.annotate("day 0 = observed\nevent peak", xy=(0.2, 0.04),
                        xycoords="axes fraction", fontsize=7, color="dimgrey")
        ax_evt.set_xlabel("days before / after the observed flow peak")
        ax_evt.set_ylabel("daily flow (MGD)")
        ax_evt.set_title("average rise and recession around the 20 largest"
                         " observed flow events")
        ax_evt.legend(fontsize=7)

        # --- (c) day-over-day variability ---
        for series, color, lbl_ in [(o, "k", "obs"), (s, "#d62728", "sim")]:
            pct = (series.diff().abs() / series.shift(1)).dropna() * 100
            pct = pct[np.isfinite(pct)]
            x = np.sort(pct.values)[::-1]
            p_exc = np.arange(1, len(x) + 1) / (len(x) + 1)
            ax_var.plot(p_exc, x, color=color, lw=1.4, label=lbl_)
        ax_var.set_xlabel("fraction of days with a change at least this large")
        ax_var.set_ylabel("change in flow from the previous day (%)")
        ax_var.set_yscale("log")
        ax_var.set_ylim(bottom=0.5)
        ax_var.set_title("day-to-day flow variability"
                         "  (reading: x=0.2 means 20% of days)")
        ax_var.legend(fontsize=8)

        # --- (d) example wet-season window (largest obs event) ---
        t_peak = o.idxmax()
        w0, w1 = t_peak - pd.Timedelta(days=45), t_peak + pd.Timedelta(days=90)
        ax_win.plot(o.loc[w0:w1].index, o.loc[w0:w1], color="k", lw=1.3,
                    label="obs")
        ax_win.plot(s.loc[w0:w1].index, s.loc[w0:w1], color="#d62728", lw=1.3,
                    label="sim")
        ax_win.set_yscale("log")
        ax_win.set_ylabel("daily flow (MGD)")
        ax_win.set_title(f"daily flow around the largest observed event"
                         f" (peak {t_peak.date()})")
        ax_win.legend(fontsize=8)
        for lab in ax_win.get_xticklabels():
            lab.set_rotation(30)

        fig.tight_layout()
        out = FIG_DIR / f"dynamics_{r}_{tag}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    labels = args or ["tuned_v007", "baseline_default"]
    tag = labels[0]

    # NOR bands: draw for runs whose custom CSV exists in output/
    params_by_run = {}
    for lbl in labels:
        csv = OUTPUT_DIR / f"custom_starfit_{lbl.replace('tuned_', '')}.csv"
        if csv.exists():
            params_by_run[lbl] = pd.read_csv(csv, index_col=0)
        elif lbl == "baseline_default":
            params_by_run[lbl] = load_default_params()
    overview_figure(labels, params_by_run, tag)

    # Rule-structure + dynamics figures for the primary (first) run only
    if tag in params_by_run:
        rule_structure_figure(tag, params_by_run[tag], tag)
    release_dynamics_figure(tag, tag)
