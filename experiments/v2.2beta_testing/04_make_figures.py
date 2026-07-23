"""
Figures for the v2.2beta test runs.

Requires outputs from 01_run_simulations.py and 02_offline_starfit.py.

fig01  NYC aggregate storage, prediction modes vs observed (full period)
fig02  Montague/Trenton flow vs MRF target during the 2001-02 drought
fig03  Lower basin contributions to the Trenton target, by mode
fig04  Offline STARFIT simulator vs in-model releases
fig05  Offline storage spin-up across initial_volume_frac values
fig06  Default vs custom (demo CSV) STARFIT parameters, reservoir storage
fig07  Updated observed NYC storage record with simulation overlay
fig08  DRBC lower basin aggregate storage, by mode (full period)

Usage:
    python 04_make_figures.py
"""
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pywrdrb
from pywrdrb.path_manager import get_pn_object

from utils import START_DATE, END_DATE, OUTPUT_DIR, FIG_DIR, make_dirs, require_files

MODES = ["regression_disagg", "perfect_foresight"]
RUN_LABELS = MODES + ["custom_starfit_demo"]
COLORS = {
    "regression_disagg": "#1f77b4",
    "perfect_foresight": "#d62728",
    "custom_starfit_demo": "#9467bd",
}
NYC_RESERVOIRS = ["cannonsville", "pepacton", "neversink"]
LOWER_BASIN = ["beltzvilleCombined", "blueMarsh", "nockamixon"]
DROUGHTS = [
    ("1963-06-01", "1967-06-30"),
    ("2001-10-01", "2002-11-30"),
    ("2016-09-01", "2017-04-30"),
]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def get_df(data, results_set, label):
    df = getattr(data, results_set)[label][0].copy()
    df.index = pd.to_datetime(df.index)
    return df


def save(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def shade_droughts(ax):
    for start, end in DROUGHTS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="red")


def fig01_nyc_storage(data):
    fig, ax = plt.subplots(figsize=(20, 5))
    for mode in MODES:
        storage = get_df(data, "res_storage", mode)
        nyc = storage[NYC_RESERVOIRS].sum(axis=1) / 1000
        ax.plot(nyc, color=COLORS[mode], lw=1, label=mode)
    obs = get_df(data, "res_storage", "obs")
    obs_nyc = (obs[NYC_RESERVOIRS].sum(axis=1, skipna=False) / 1000).dropna()
    ax.plot(obs_nyc.loc[START_DATE:END_DATE], color="k", ls="--", lw=1,
            label="observed", zorder=5)
    shade_droughts(ax)
    ax.set_ylabel("NYC combined storage (BG)")
    ax.set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))
    ax.legend(ncol=3, loc="lower right")
    ax.set_title("NYC aggregate storage by flow prediction mode")
    save(fig, "fig01_nyc_storage_modes.png")


def fig02_mrf_flows(data):
    zoom = slice("2001-06-01", "2002-12-31")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, node in zip(axes, ["delMontague", "delTrenton"]):
        for mode in MODES:
            flow = get_df(data, "major_flow", mode)[node].loc[zoom]
            target = get_df(data, "mrf_target", mode)[node].loc[zoom]
            ax.plot(flow, color=COLORS[mode], lw=1, label=mode)
            ax.plot(target, color=COLORS[mode], lw=0.8, ls=":", alpha=0.8)
        ax.set_yscale("log")
        ax.set_ylabel(f"{node} flow (MGD)")
        ax.set_title(f"{node}: flow (solid) and MRF target (dotted)")
    axes[0].legend(ncol=2, loc="upper right")
    save(fig, "fig02_mrf_flows_drought.png")


def fig03_lower_basin(data):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for mode in MODES:
        contrib = get_df(data, "lower_basin_mrf_contributions", mode)
        total = contrib.sum(axis=1).rolling(30).mean()
        axes[0].plot(total, color=COLORS[mode], lw=1, label=mode)
    shade_droughts(axes[0])
    axes[0].set_ylabel("Total contribution (MGD, 30-day mean)")
    axes[0].set_title("Lower basin releases for the Trenton target")
    axes[0].legend(ncol=2)
    axes[0].set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))

    width = 0.3
    xpos = range(len(LOWER_BASIN))
    for i, mode in enumerate(MODES):
        contrib = get_df(data, "lower_basin_mrf_contributions", mode)
        totals = [contrib[f"mrf_trenton_{r}"].sum() / 1000 for r in LOWER_BASIN]
        offset = (i - (len(MODES) - 1) / 2) * width
        axes[1].bar([x + offset for x in xpos], totals, width,
                    color=COLORS[mode], label=mode)
    axes[1].set_xticks(list(xpos))
    axes[1].set_xticklabels(LOWER_BASIN)
    axes[1].set_ylabel("Total volume 2000-2023 (BG)")
    axes[1].legend()
    save(fig, "fig03_lower_basin_contributions.png")


def fig04_offline_vs_inmodel(data):
    offline = pd.read_csv(OUTPUT_DIR / "offline_releases_default.csv",
                          index_col=0, parse_dates=True)
    inmodel = get_df(data, "res_release", "perfect_foresight")
    zoom = slice("2002-01-01", "2002-12-31")

    # prompton and mongaupeCombined follow pure STARFIT rules in-model, so the
    # offline simulator should match closely; blueMarsh gets additional FFMP
    # releases for the Trenton target, so it deviates by design
    reservoirs = ["prompton", "mongaupeCombined", "blueMarsh"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), width_ratios=[2.5, 1])
    for row, res in zip(axes, reservoirs):
        row[0].plot(inmodel[res].loc[zoom], color="C0", lw=1, label="in-model")
        row[0].plot(offline[res].loc[zoom], color="C3", lw=1, ls="--", label="offline")
        row[0].set_ylabel(f"{res}\nrelease (MGD)")

        common = inmodel[res].align(offline[res], join="inner")
        row[1].scatter(common[0], common[1], s=2, alpha=0.2, color="C0")
        lim = max(common[0].max(), common[1].max())
        row[1].plot([0, lim], [0, lim], color="k", lw=0.8)
        row[1].set_xlabel("in-model (MGD)")
        row[1].set_ylabel("offline (MGD)")
    axes[0][0].legend()
    axes[0][0].set_title("2002 releases")
    axes[0][1].set_title("Full period, daily")
    save(fig, "fig04_offline_vs_inmodel_releases.png")


def fig05_initial_volume_sweep():
    pn = get_pn_object()
    params = pd.read_csv(pn.operational_constants.get_str("istarf_conus.csv"), index_col=0)
    cap = params.loc["prompton", "Adjusted_CAP_MG"]

    storage = pd.read_csv(OUTPUT_DIR / "offline_storage_sweep_prompton.csv",
                          index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in storage.columns:
        frac = col.replace("ivf_", "")
        ax.plot(storage[col].loc["1945":"1948"] / cap * 100, lw=1,
                label=f"initial fraction {frac}")
    ax.set_ylabel("prompton storage (% capacity)")
    ax.set_title("Offline simulator spin-up from different initial storages")
    ax.legend()
    save(fig, "fig05_offline_initial_volume.png")


def fig06_custom_starfit(data):
    reservoirs = ["blueMarsh", "beltzvilleCombined", "fewalter", "prompton"]
    obs = get_df(data, "res_storage", "obs")

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for ax, res in zip(axes, reservoirs):
        default = get_df(data, "res_storage", "perfect_foresight")[res]
        custom = get_df(data, "res_storage", "custom_starfit_demo")[res]
        ax.plot(default, color=COLORS["perfect_foresight"], lw=1, label="default params")
        ax.plot(custom, color=COLORS["custom_starfit_demo"], lw=1, label="custom demo params")
        if res in obs.columns:
            ax.plot(obs[res].loc[START_DATE:END_DATE].dropna(), color="k", ls="--",
                    lw=0.8, label="observed", zorder=5)
        ax.set_ylabel(f"{res}\nstorage (MG)")
    axes[0].legend(ncol=3, loc="upper right")
    axes[0].set_title("Storage with default vs custom STARFIT parameters (perfect_foresight runs)")
    axes[-1].set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))
    save(fig, "fig06_custom_starfit_storage.png")


def fig07_obs_record(data):
    obs = get_df(data, "res_storage", "obs")
    sim = get_df(data, "res_storage", "perfect_foresight")
    drbc_period = (pd.Timestamp("1999-12-01"), pd.Timestamp("2021-11-30"))
    prior_end = pd.Timestamp("2023-12-31")

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, res in zip(axes, NYC_RESERVOIRS):
        ax.plot(obs[res].dropna(), color="k", lw=0.8, label="observed")
        ax.plot(sim[res], color=COLORS["perfect_foresight"], lw=0.8, alpha=0.7,
                label="simulated (perfect_foresight)")
        ax.axvspan(*drbc_period, alpha=0.06, color="blue")
        ax.axvline(prior_end, color="gray", ls=":", lw=1)
        ax.set_ylabel(f"{res}\nstorage (MG)")
    axes[0].legend(ncol=2, loc="lower left")
    axes[0].set_title(
        "Observed NYC storage record: DRBC-based period (blue), "
        "extension past v2.1 record (right of dotted line)"
    )
    save(fig, "fig07_obs_record_update.png")


def fig08_lower_basin_storage(data):
    # no observed line: nockamixon has no storage record, so the observed
    # aggregate cannot be formed
    fig, ax = plt.subplots(figsize=(20, 5))
    for mode in MODES:
        storage = get_df(data, "res_storage", mode)
        total = storage[LOWER_BASIN].sum(axis=1) / 1000
        ax.plot(total, color=COLORS[mode], lw=1, label=mode)
    shade_droughts(ax)
    ax.set_ylabel("DRBC lower basin storage (BG)")
    ax.set_xlim(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE))
    ax.legend(ncol=2, loc="lower right")
    ax.set_title(
        "Aggregate lower basin storage (blueMarsh + beltzvilleCombined + nockamixon) "
        "by flow prediction mode"
    )
    save(fig, "fig08_lower_basin_storage.png")


def main():
    make_dirs()
    require_files(
        [OUTPUT_DIR / f"{label}.hdf5" for label in RUN_LABELS],
        "Run 01_run_simulations.py first.",
    )
    require_files(
        [OUTPUT_DIR / "offline_releases_default.csv",
         OUTPUT_DIR / "offline_storage_sweep_prompton.csv"],
        "Run 02_offline_starfit.py first.",
    )

    data = pywrdrb.Data(print_status=False)
    data.load_output(
        output_filenames=[str(OUTPUT_DIR / f"{label}.hdf5") for label in RUN_LABELS],
        results_sets=["res_storage", "major_flow", "res_release",
                      "mrf_target", "lower_basin_mrf_contributions"],
    )
    data.load_observations(results_sets=["res_storage"])

    fig01_nyc_storage(data)
    fig02_mrf_flows(data)
    fig03_lower_basin(data)
    fig04_offline_vs_inmodel(data)
    fig05_initial_volume_sweep()
    fig06_custom_starfit(data)
    fig07_obs_record(data)
    fig08_lower_basin_storage(data)


if __name__ == "__main__":
    main()
