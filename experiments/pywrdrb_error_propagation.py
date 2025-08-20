# -*- coding: utf-8 -*-
"""
STARFIT Diagnostics & Error Analysis Toolkit
============================================

This script extends the Pywr-DRB modeling workflow with a comprehensive
set of diagnostic plots and rolling-skill analyses for surrogate policies
(STARFIT, PWL, RBF). It is designed for evaluating simulation skill,
tracing residuals, and visualizing error propagation across temporal
scales and management periods.

Main Features
-------------
1. **Simulation & Data Handling**
   - Runs a Pywr-DRB model for selected inflow types and loads simulation output.
   - Retrieves and aligns observed reservoir releases, storage, and gage flows.
   - Supports optional NWIS retrieval for Prompton inflow augmentation.

2. **Reservoir Diagnostics**
   - **6-panel Release Diagnostics**: Daily, monthly, and annual time series,
     flow duration curves (FDCs), climatologies, with error shading and NSE/KGE/Bias badges.
   - **9-panel Storage + Release Diagnostics**: Percent-capacity storage, paired with
     release diagnostics and FDCs, sharing axes across timescales.

3. **Rolling-Skill Evaluation**
   - Monthly rolling NSE/KGE/log-skill with customizable window length.
   - Heatmap visualization across drought/management periods, excluding full-period rows.
   - TwoSlopeNorm color scaling centered at zero (no-skill).

4. **Residual Error Analysis**
   - Enhanced error time series with running means, LOWESS smoothing, and decadal skill summaries.
   - Error vs. observed flow percentile plots (by decade/season), with LOWESS trends and acceptable bands.
   - 4-panel seasonal × decadal scatter diagnostics.
   - Joint density (hexbin) plots of error vs. flow percentile, with marginal histograms.

5. **Distributional Comparisons**
   - Period-stacked FDCs of observed vs. simulated flows, stratified by drought/management periods.
   - Explicit style separation (solid = obs, dashed = sim) with color-coded periods.

6. **Error Propagation**
   - Reservoir release → downstream flow propagation diagnostics (Montague, Trenton).
   - Time series + FDC comparisons with NSE/KGE inset metrics.
   - Lower-basin contribution stacking to visualize Trenton MRF activation.

Usage
-----
- Requires `pywrdrb` installed and access to pre-computed simulation outputs and
  observational data files.
- Generates publication-quality figures under `./figures/`, organized by theme
  (release, storage, rolling_skill, error_ts, error_vs_pct, error_propagation, etc.).
- Intended for collaborative debugging, paper figures, and diagnostic workflows.

Notes
-----
- Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, optional `hydroeval`, `seaborn`, `statsmodels`.
- Storage diagnostics use hard-coded reservoir capacities (MG), update if new reservoirs are added.
- Rolling metrics exclude "Full Period" rows to emphasize sub-period variability.

"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

# optional deps
try:
    import seaborn as sns  # not strictly required here
except Exception:
    sns = None

try:
    import hydroeval as he
    _HAS_HE = True
except Exception:
    _HAS_HE = False

# -------------------------------------------------------------------
# 0) Run/Load model & data
# -------------------------------------------------------------------
import pywrdrb

inflow_types = ['pub_nhmv10_BC_withObsScaled']
start_date = "1983-10-01"
end_date = "2023-12-31"

for inflow_type in inflow_types:
    print(f"\nRunning simulation for inflow type: {inflow_type}")
    mb = pywrdrb.ModelBuilder(inflow_type=inflow_type, start_date=start_date, end_date=end_date)
    mb.make_model()
    model_filename = f"./model_{inflow_type}.json"
    output_filename = f"./pywrdrb_output_{inflow_type}.hdf5"
    mb.write_model(model_filename)
    model = pywrdrb.Model.load(model_filename)
    recorder = pywrdrb.OutputRecorder(model, output_filename)
    _stats = model.run()
    assert os.path.exists(output_filename), f"Simulation output not found for {inflow_type}"
    print(f" Simulation completed and output file saved: {output_filename}")

reference_output = "./pywrdrb_output_pub_nhmv10_BC_withObsScaled.hdf5"
results_sets = ['major_flow', 'res_storage', 'reservoir_downstream_gage', 'lower_basin_mrf_contributions']

# Simulated
output_filename_nwm = "./pywrdrb_output_pub_nhmv10_BC_withObsScaled.hdf5"
data_nwm = pywrdrb.Data(print_status=True, results_sets=results_sets, output_filenames=[output_filename_nwm])
data_nwm.load_output()
print(data_nwm.major_flow.keys())
print(data_nwm.res_storage.keys())
print(data_nwm.reservoir_downstream_gage.keys())

key = os.path.splitext(os.path.basename(output_filename_nwm))[0]

df_nwm_major_flow = data_nwm.major_flow[key][0]
df_nwm_res_storage = data_nwm.res_storage[key][0]
df_nwm_downstream_gage = data_nwm.reservoir_downstream_gage[key][0]
df_lower_basin_mrf_contributions = data_nwm.lower_basin_mrf_contributions[key][0]

# Observed
obs_data = pywrdrb.Data(print_status=True, results_sets=results_sets, output_filenames=[reference_output])
obs_data.load_observations()
df_obs_major_flow = obs_data.major_flow["obs"][0]
df_obs_res_storage = obs_data.res_storage["obs"][0]
df_obs_downstream_gage = obs_data.reservoir_downstream_gage["obs"][0]

# OPTIONAL: append Prompton DV from NWIS (convert cfs -> MGD to match model units)
try:
    from dataretrieval import nwis
    def _make_tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        idx = pd.to_datetime(idx)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return idx

    df_obs_downstream_gage.index = _make_tz_naive(df_obs_downstream_gage.index)
    df_obs_downstream_gage.sort_index(inplace=True)
    s_date = df_obs_downstream_gage.index.min().strftime("%Y-%m-%d")
    e_date = df_obs_downstream_gage.index.max().strftime("%Y-%m-%d")

    print(f"Retrieving Prompton (01429000) DV 00060 from {s_date} to {e_date} ...")
    df_prompton_raw = nwis.get_record(sites="01429000", start=s_date, end=e_date, parameterCd="00060", service="dv")
    if df_prompton_raw is not None and len(df_prompton_raw) > 0:
        df_prompton_raw.index = _make_tz_naive(pd.to_datetime(df_prompton_raw.index))
        df_prompton_raw.sort_index(inplace=True)
        col = None
        for c in df_prompton_raw.columns:
            if "00060" in c and ("Mean" in c or c.endswith("_Mean")):
                col = c; break
        if col is None:
            cand = [c for c in df_prompton_raw.columns if "qual" not in c.lower()]
            if len(cand) == 1: col = cand[0]
        if col is not None:
            df_prompton = df_prompton_raw[[col]].rename(columns={col: "prompton"}).astype(float)
            # convert cfs -> MGD
            df_prompton["prompton"] = df_prompton["prompton"] * 0.646317
            prompton_aligned = df_prompton.reindex(df_obs_downstream_gage.index)
            df_obs_downstream_gage.loc[:, "prompton"] = prompton_aligned["prompton"].values
            obs_data.reservoir_downstream_gage["obs"][0] = df_obs_downstream_gage
            print("Prompton appended to df_obs_downstream_gage (units: MGD).")
except Exception as e:
    print(f"NWIS Prompton append skipped: {e}")

# -------------------------------------------------------------------
# 1) Periods & reservoirs
# -------------------------------------------------------------------
drought_periods = {
    "Drought: 1995": ("1995-01-01", "1995-12-31"),
    "Drought: 1998–2002": ("1998-01-01", "2002-12-31"),
    "Drought: 2016": ("2016-01-01", "2016-12-31"),
    "Drought: 2022–2023": ("2022-01-01", "2023-12-31"),
    "Non-Drought: 1983–1994": ("1983-10-01", "1994-12-31"),
    "Non-Drought: 1996–1997": ("1996-01-01", "1997-12-31"),
    "Non-Drought: 2003–2015": ("2003-01-01", "2015-12-31"),
    "Non-Drought: 2017–2021": ("2017-01-01", "2021-12-31"),
    "Management: 2017–2023": ("2017-01-01", "2023-12-31"),
    "Full Period: 1983–2023": ("1983-10-01", "2023-12-31"),
    "Subset: 2017–2018": ("2017-01-01", "2018-12-31"),
    "Subset: 2019–2020": ("2019-01-01", "2020-12-31"),
    "Subset: 2021–2023": ("2021-01-01", "2023-12-31"),
}
reservoirs_of_interest = ["blueMarsh", "beltzvilleCombined", "fewalter", "prompton"]
os.makedirs("figures/release", exist_ok=True)
os.makedirs("figures/release_storage", exist_ok=True)

# -------------------------------------------------------------------
# 2) Helpers (capacities, metrics, shading, annotations)
# -------------------------------------------------------------------
RESERVOIR_CAP_MG = {
    "prompton": 27956.02,
    "beltzvilleCombined": 48317.0588,
    "fewalter": 35800.0,
    "blueMarsh": 42320.3544,
}

def safe_label(s: str) -> str:
    s = s.replace("–", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _resample(series: pd.Series, code: str):
    if code == "D": return series
    if code == "M": return series.resample("ME").mean()
    if code == "Y": return series.resample("YE").mean()
    raise ValueError(f"Unknown timescale {code!r}")

def _align_xy(obs: pd.Series, sim: pd.Series, code="D"):
    o = _resample(obs, code)
    s = _resample(sim, code)
    o, s = o.align(s, join="inner")
    m = (~o.isna()) & (~s.isna())
    return o[m], s[m]

def to_percent_storage(storage_mg: pd.Series, reservoir_name: str) -> pd.Series:
    cap = RESERVOIR_CAP_MG.get(reservoir_name)
    if not cap:
        raise KeyError(f"Missing capacity for {reservoir_name!r}")
    return 100.0 * storage_mg / cap

# metrics (using hydroeval if present)
def calculate_all_error_metrics(obs, modeled, timescale="D"):
    if timescale == "M":
        obs = obs.resample("ME").mean(); modeled = modeled.resample("ME").mean()
    elif timescale == "Y":
        obs = obs.resample("YE").mean(); modeled = modeled.resample("YE").mean()
    elif timescale == "Full":
        obs = pd.Series([obs.mean()], index=[obs.index.min()])
        modeled = pd.Series([modeled.mean()], index=[modeled.index.min()])
    obs, modeled = obs.align(modeled, join='inner')
    mask = (~obs.isna()) & (~modeled.isna()); obs = obs[mask]; modeled = modeled[mask]
    if len(obs) < 2: return None
    obs_safe = obs.clip(lower=1e-6); modeled_safe = modeled.clip(lower=1e-6)
    if _HAS_HE:
        kge, r, alpha, beta = he.evaluator(he.kge, modeled, obs)
        nse = he.evaluator(he.nse, modeled, obs)
        logkge, logr, logalpha, logbeta = he.evaluator(he.kge, modeled_safe, obs_safe, transform="log")
        lognse = he.evaluator(he.nse, modeled_safe, obs_safe, transform="log")
    else:
        # simple fallbacks
        def _nse(x,y):
            den = np.sum((y - y.mean())**2)
            return np.nan if den == 0 else 1 - np.sum((x - y)**2)/den
        def _kge(x,y):
            r = np.corrcoef(x,y)[0,1] if (np.std(x)>0 and np.std(y)>0) else np.nan
            alpha = np.std(x)/np.std(y) if np.std(y)!=0 else np.nan
            beta  = np.mean(x)/np.mean(y) if np.mean(y)!=0 else np.nan
            if np.any(np.isnan([r,alpha,beta])): return np.nan
            return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
        nse = [_nse(modeled, obs)]; kge = [_kge(modeled, obs)]
        lognse = [_nse(np.log(modeled_safe), np.log(obs_safe))]
        logkge = [_kge(np.log(modeled_safe), np.log(obs_safe))]
        r=alpha=beta=logr=logalpha=logbeta=[np.nan]

    # FDC matches
    kss, _ = stats.ks_2samp(modeled, obs)
    fdc_match_horiz = 1 - kss
    obs_ordered = np.log(np.sort(obs_safe)); modeled_ordered = np.log(np.sort(modeled_safe))
    fdc_range = max(obs_ordered.max(), modeled_ordered.max()) - min(obs_ordered.min(), modeled_ordered.min())
    fdc_match_vert = 1 - np.abs(obs_ordered - modeled_ordered).max() / fdc_range if fdc_range != 0 else np.nan

    # AC/roughness
    def autocorr_ratio(series_obs, series_mod, lag):
        olog = np.log(series_obs.clip(lower=1e-6)); mlog = np.log(series_mod.clip(lower=1e-6))
        oa = np.corrcoef(olog[lag:], olog[:-lag])[0,1]; ma = np.corrcoef(mlog[lag:], mlog[:-lag])[0,1]
        return ma/oa if oa != 0 else np.nan
    rel_autocorr1 = autocorr_ratio(obs, modeled, lag=1)
    rel_autocorr7 = autocorr_ratio(obs, modeled, lag=7)
    o_dl = np.diff(np.log(obs_safe)); m_dl = np.diff(np.log(modeled_safe))
    rel_roughness_log = np.std(m_dl)/np.std(o_dl) if np.std(o_dl)!=0 else np.nan

    # FDC slope biases
    def slope_quantile(data, q1, q2):
        d = data.clip(lower=1e-6)
        return (np.log(np.quantile(d, q2)) - np.log(np.quantile(d, q1))) / (q2 - q1)
    sfdc_2575_obs_log = slope_quantile(obs, 0.25, 0.75)
    sfdc_0199_obs_log = slope_quantile(obs, 0.01, 0.99)
    sfdc_MinMax_obs_log = np.log(obs_safe.max()) - np.log(obs_safe.min())
    sfdc_2575_mod_log = slope_quantile(modeled, 0.25, 0.75)
    sfdc_0199_mod_log = slope_quantile(modeled, 0.01, 0.99)
    sfdc_MinMax_mod_log = np.log(modeled_safe.max()) - np.log(modeled_safe.min())

    return {
        "nse": float(nse[0]), "kge": float(kge[0]),
        "lognse": float(lognse[0]), "logkge": float(logkge[0]),
        "fdc_match_horiz": fdc_match_horiz, "fdc_match_vert": fdc_match_vert,
        "rel_autocorr1": rel_autocorr1, "rel_autocorr7": rel_autocorr7,
        "rel_roughness_log": rel_roughness_log,
    }

# legend placement control (prevents overlap with metric box)
LEGEND_OUTSIDE = False
LEGEND_KW_INSIDE  = dict(loc="lower left", frameon=True, framealpha=0.85)
LEGEND_KW_OUTSIDE = dict(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True, framealpha=0.85)

def shade_error_band(ax, obs, sim, timescale="D"):
    o, s = _align_xy(obs, sim, timescale)
    if len(o) == 0: return
    ax.fill_between(o.index, o, s, where=(s>=o),
                    facecolor=(0.2,0.6,0.2,0.18), edgecolor="none", zorder=1)
    ax.fill_between(o.index, o, s, where=(s< o),
                    facecolor=(0.8,0.2,0.2,0.18), edgecolor="none", zorder=1)

def annotate_metrics_box(ax, obs, sim, timescale="D", label="", loc="UR"):
    m = calculate_all_error_metrics(obs.copy(), sim.copy(), timescale=timescale)
    if not m: return
    o, s = _align_xy(obs, sim, timescale)
    bias = float((s - o).mean()) if len(o) else float("nan")
    txt = f"{label}  NSE={m['nse']:.2f}  KGE={m['kge']:.2f}\nBias={bias:.2f}"
    loc_map = {"UR": 1, "UL": 2, "LL": 3, "LR": 4}
    at = AnchoredText(txt, loc=loc_map.get(loc, 1), prop=dict(size=9),
                      frameon=True, borderpad=0.4)
    at.patch.set_alpha(0.8); at.zorder = 10
    ax.add_artist(at)

# -------------------------------------------------------------------
# 3) 6-panel: Release diagnostics (error + metrics + sharing)
# -------------------------------------------------------------------
def plot_reservoir_release_diagnostics(
    reservoir, sim_df, obs_df, start, end, ylabel="Flow (MGD)", save_path=None
):
    sim = sim_df[reservoir].loc[start:end]
    obs = None
    if reservoir in obs_df.columns:
        obs = obs_df[reservoir].loc[start:end]
        if obs.dropna().empty: obs = None

    monthly_sim = sim.resample("ME").mean()
    annual_sim  = sim.resample("YE").mean()
    if obs is not None:
        monthly_obs = obs.resample("ME").mean()
        annual_obs  = obs.resample("YE").mean()

    def get_fdc(data):
        vals = np.sort(data.dropna())[::-1]
        prob = np.linspace(0, 100, len(vals))
        return prob, vals

    fdc_daily_sim   = get_fdc(sim)
    fdc_monthly_sim = get_fdc(monthly_sim)
    fdc_annual_sim  = get_fdc(annual_sim)
    if obs is not None:
        fdc_daily_obs   = get_fdc(obs)
        fdc_monthly_obs = get_fdc(monthly_obs)
        fdc_annual_obs  = get_fdc(annual_obs)

    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=False)
    # share y within columns
    for r in (1,2): axs[r,0].sharey(axs[0,0])
    for r in (1,2): axs[r,1].sharey(axs[0,1])

    legend_kw = LEGEND_KW_OUTSIDE if LEGEND_OUTSIDE else LEGEND_KW_INSIDE
    if LEGEND_OUTSIDE: fig.subplots_adjust(right=0.86)

    # Daily TS
    if obs is not None: axs[0,0].plot(obs, label="Observed", lw=1.5, color="black", zorder=2)
    axs[0,0].plot(sim, label="Simulated", lw=1.2, ls="--", color="tab:blue", zorder=2)
    if obs is not None:
        shade_error_band(axs[0,0], obs, sim, "D")
        annotate_metrics_box(axs[0,0], obs, sim, "D", "Daily", loc="UR")
    axs[0,0].set_title(f"{reservoir}: Daily Time Series")
    axs[0,0].set_ylabel(ylabel); axs[0,0].legend(**legend_kw); axs[0,0].grid(True)

    # Daily FDC
    if obs is not None: axs[0,1].plot(*fdc_daily_obs, label="Observed", color="black")
    axs[0,1].plot(*fdc_daily_sim, label="Simulated", ls="--", color="tab:blue")
    axs[0,1].set_title(f"{reservoir}: Daily FDC")
    axs[0,1].set_xlabel("Exceedance Probability (%)")
    axs[0,1].set_yscale("log"); axs[0,1].legend(**legend_kw); axs[0,1].grid(True)

    # Monthly clim
    mg_sim = monthly_sim.groupby(monthly_sim.index.month).mean().values
    if obs is not None:
        mg_obs = monthly_obs.groupby(monthly_obs.index.month).mean().values
        axs[1,0].plot(month_names, mg_obs, label="Observed", lw=2, color="black", marker="o", zorder=2)
    axs[1,0].plot(month_names, mg_sim, label="Simulated", lw=1.2, ls="--", color="tab:blue", marker="o", zorder=2)
    if obs is not None:
        annotate_metrics_box(axs[1,0], monthly_obs, monthly_sim, "M", "Monthly", loc="UR")
    axs[1,0].set_title(f"{reservoir}: Monthly Avg Flow Climatology")
    axs[1,0].set_ylabel(ylabel); axs[1,0].legend(**legend_kw); axs[1,0].grid(True)

    # Monthly FDC
    if obs is not None: axs[1,1].plot(*fdc_monthly_obs, label="Observed", color="black")
    axs[1,1].plot(*fdc_monthly_sim, label="Simulated", ls="--", color="tab:blue")
    axs[1,1].set_title(f"{reservoir}: Monthly FDC")
    axs[1,1].set_xlabel("Exceedance Probability (%)")
    axs[1,1].set_yscale("log"); axs[1,1].legend(**legend_kw); axs[1,1].grid(True)

    # Annual avg
    if obs is not None:
        axs[2,0].plot(annual_obs.index.year, annual_obs.values, label="Observed", lw=2, marker='o', color="black", zorder=2)
    axs[2,0].plot(annual_sim.index.year, annual_sim.values, label="Simulated", lw=1.5, marker='o', ls='--', color="tab:blue", zorder=2)
    if obs is not None:
        annotate_metrics_box(axs[2,0], annual_obs, annual_sim, "Y", "Annual", loc="UR")
    axs[2,0].set_title(f"{reservoir}: Annual Avg Flow")
    axs[2,0].set_xlabel("Year"); axs[2,0].set_ylabel(ylabel); axs[2,0].legend(**legend_kw); axs[2,0].grid(True)

    # Annual FDC
    if obs is not None: axs[2,1].plot(*fdc_annual_obs, label="Observed", color="black")
    axs[2,1].plot(*fdc_annual_sim, label="Simulated", ls="--", color="tab:blue")
    axs[2,1].set_title(f"{reservoir}: Annual FDC")
    axs[2,1].set_xlabel("Exceedance Probability (%)")
    axs[2,1].set_yscale("log"); axs[2,1].legend(**legend_kw); axs[2,1].grid(True)

    fig.suptitle(f"{reservoir} – Release Diagnostics ({start} to {end})", fontsize=15, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path: fig.savefig(save_path, dpi=300); plt.close()
    else: plt.show()

# -------------------------------------------------------------------
# 4) 9-panel: Storage+Release (percent storage + error + metrics + sharing)
# -------------------------------------------------------------------
def plot_reservoir_release_storage_diagnostics(
    reservoir, sim_df, obs_release_df, sim_storage_df, obs_storage_df,
    start, end, ylabel="Flow (MGD)", storage_ylabel="Storage (% of capacity)", save_path=None
):
    if reservoir not in sim_df.columns: print(f"Skipping {reservoir}: no sim release"); return
    if reservoir not in sim_storage_df.columns: print(f"Skipping {reservoir}: no sim storage"); return

    sim_release = sim_df[reservoir].loc[start:end]
    sim_storage_pct = to_percent_storage(sim_storage_df[reservoir].loc[start:end], reservoir)
    if sim_release.dropna().empty or sim_storage_pct.dropna().empty:
        print(f"Skipping {reservoir} {start}–{end}: missing data"); return

    obs_release = obs_release_df[reservoir].loc[start:end] if reservoir in obs_release_df.columns else None
    if obs_release is not None and obs_release.dropna().empty: obs_release = None
    obs_storage_pct = to_percent_storage(obs_storage_df[reservoir].loc[start:end], reservoir) \
                      if reservoir in obs_storage_df.columns else None
    if obs_storage_pct is not None and obs_storage_pct.dropna().empty: obs_storage_pct = None

    monthly_sim_release = sim_release.resample("ME").mean()
    annual_sim_release  = sim_release.resample("YE").mean()
    monthly_sim_storage = sim_storage_pct.resample("ME").mean()
    annual_sim_storage  = sim_storage_pct.resample("YE").mean()
    if obs_release is not None:
        monthly_obs_release = obs_release.resample("ME").mean()
        annual_obs_release  = obs_release.resample("YE").mean()
    if obs_storage_pct is not None:
        monthly_obs_storage = obs_storage_pct.resample("ME").mean()
        annual_obs_storage  = obs_storage_pct.resample("YE").mean()

    def get_fdc(data):
        vals = np.sort(data.dropna())[::-1]; prob = np.linspace(0, 100, len(vals))
        return prob, vals

    fdc_daily_sim   = get_fdc(sim_release)
    fdc_monthly_sim = get_fdc(monthly_sim_release)
    fdc_annual_sim  = get_fdc(annual_sim_release)
    if obs_release is not None:
        fdc_daily_obs   = get_fdc(obs_release)
        fdc_monthly_obs = get_fdc(monthly_obs_release)
        fdc_annual_obs  = get_fdc(annual_obs_release)

    mnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    mg_sim_rel   = monthly_sim_release.groupby(monthly_sim_release.index.month).mean()
    mg_sim_store = monthly_sim_storage.groupby(monthly_sim_storage.index.month).mean()
    if obs_release is not None:
        mg_obs_rel = monthly_obs_release.groupby(monthly_obs_release.index.month).mean()
    if obs_storage_pct is not None:
        mg_obs_store = monthly_obs_storage.groupby(monthly_obs_storage.index.month).mean()

    fig, axs = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=False)
    # share within columns (storage, release, FDC)
    for r in (1,2): axs[r,0].sharey(axs[0,0])
    for r in (1,2): axs[r,1].sharey(axs[0,1])
    for r in (1,2): axs[r,2].sharey(axs[0,2])
    legend_kw = LEGEND_KW_OUTSIDE if LEGEND_OUTSIDE else LEGEND_KW_INSIDE
    if LEGEND_OUTSIDE: fig.subplots_adjust(right=0.86)

    # Row 1: Daily Storage
    axs[0,0].plot(sim_storage_pct.index, sim_storage_pct.values, label="Simulated", color="tab:green", zorder=2)
    if obs_storage_pct is not None:
        axs[0,0].plot(obs_storage_pct.index, obs_storage_pct.values, label="Observed", color="black", zorder=2)
        shade_error_band(axs[0,0], obs_storage_pct, sim_storage_pct, "D")
    axs[0,0].set_title(f"{reservoir}: Daily Storage (Percent)")
    axs[0,0].set_ylabel(storage_ylabel); axs[0,0].legend(**legend_kw); axs[0,0].grid(True)

    # Row 1: Daily Release
    if obs_release is not None: axs[0,1].plot(obs_release, label="Observed", color="black", zorder=2)
    axs[0,1].plot(sim_release, label="Simulated", ls="--", color="tab:blue", zorder=2)
    if obs_release is not None:
        shade_error_band(axs[0,1], obs_release, sim_release, "D")
        annotate_metrics_box(axs[0,1], obs_release, sim_release, "D", "Daily", loc="UR")
    axs[0,1].set_title(f"{reservoir}: Daily Release")
    axs[0,1].set_ylabel(ylabel); axs[0,1].legend(**legend_kw); axs[0,1].grid(True)

    # Row 1: Daily Release FDC
    if obs_release is not None: axs[0,2].plot(*fdc_daily_obs, label="Observed", color="black")
    axs[0,2].plot(*fdc_daily_sim, label="Simulated", ls="--", color="tab:blue")
    axs[0,2].set_title(f"{reservoir}: Daily Release FDC")
    axs[0,2].set_xlabel("Exceedance Probability (%)"); axs[0,2].set_yscale("log")
    axs[0,2].legend(**legend_kw); axs[0,2].grid(True)

    # Row 2: Monthly Storage
    axs[1,0].plot(mnames, mg_sim_store.values, label="Simulated", color="tab:green", marker="o", zorder=2)
    if obs_storage_pct is not None:
        axs[1,0].plot(mnames, mg_obs_store.values, label="Observed", color="black", marker="o", zorder=2)
    axs[1,0].set_title(f"{reservoir}: Monthly Avg Storage"); axs[1,0].set_ylabel(storage_ylabel)
    axs[1,0].legend(**legend_kw); axs[1,0].grid(True)

    # Row 2: Monthly Release (+ metrics)
    if obs_release is not None:
        axs[1,1].plot(mnames, mg_obs_rel.values, label="Observed", color="black", marker="o", zorder=2)
    axs[1,1].plot(mnames, mg_sim_rel.values, label="Simulated", ls="--", color="tab:blue", marker="o", zorder=2)
    if obs_release is not None:
        annotate_metrics_box(axs[1,1], monthly_obs_release, monthly_sim_release, "M", "Monthly", loc="UR")
    axs[1,1].set_title(f"{reservoir}: Monthly Avg Release"); axs[1,1].set_ylabel(ylabel)
    axs[1,1].legend(**legend_kw); axs[1,1].grid(True)

    # Row 2: Monthly Release FDC
    if obs_release is not None: axs[1,2].plot(*fdc_monthly_obs, label="Observed", color="black")
    axs[1,2].plot(*fdc_monthly_sim, label="Simulated", ls="--", color="tab:blue")
    axs[1,2].set_title(f"{reservoir}: Monthly Release FDC")
    axs[1,2].set_xlabel("Exceedance Probability (%)"); axs[1,2].set_yscale("log")
    axs[1,2].legend(**legend_kw); axs[1,2].grid(True)

    # Row 3: Annual Storage
    axs[2,0].plot(annual_sim_storage.index.year, annual_sim_storage.values, label="Simulated", color="tab:green", marker="o", zorder=2)
    if obs_storage_pct is not None:
        axs[2,0].plot(annual_obs_storage.index.year, annual_obs_storage.values, label="Observed", color="black", marker="o", zorder=2)
    axs[2,0].set_title(f"{reservoir}: Annual Avg Storage"); axs[2,0].set_ylabel(storage_ylabel)
    axs[2,0].set_xlabel("Year"); axs[2,0].legend(**legend_kw); axs[2,0].grid(True)

    # Row 3: Annual Release (+ metrics)
    if obs_release is not None:
        axs[2,1].plot(annual_obs_release.index.year, annual_obs_release.values, label="Observed", marker="o", color="black", zorder=2)
    axs[2,1].plot(annual_sim_release.index.year, annual_sim_release.values, label="Simulated", ls="--", marker="o", color="tab:blue", zorder=2)
    if obs_release is not None:
        annotate_metrics_box(axs[2,1], annual_obs_release, annual_sim_release, "Y", "Annual", loc="UR")
    axs[2,1].set_title(f"{reservoir}: Annual Avg Release"); axs[2,1].set_ylabel(ylabel)
    axs[2,1].set_xlabel("Year"); axs[2,1].legend(**legend_kw); axs[2,1].grid(True)

    # Row 3: Annual Release FDC
    if obs_release is not None: axs[2,2].plot(*fdc_annual_obs, label="Observed", color="black")
    axs[2,2].plot(*fdc_annual_sim, label="Simulated", ls="--", color="tab:blue")
    axs[2,2].set_title(f"{reservoir}: Annual Release FDC")
    axs[2,2].set_xlabel("Exceedance Probability (%)"); axs[2,2].set_yscale("log")
    axs[2,2].legend(**legend_kw); axs[2,2].grid(True)

    fig.suptitle(f"{reservoir} – Storage and Release Diagnostics\n{start} to {end}", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.94])
    if save_path: fig.savefig(save_path, dpi=300); plt.close()
    else: plt.show()

# -------------------------------------------------------------------
# 5) Rolling-skill heatmap (exclude "Full Period")
# -------------------------------------------------------------------
def _align(obs: pd.Series, sim: pd.Series):
    o, s = obs.align(sim, join="inner")
    m = (~o.isna()) & (~s.isna())
    return o[m], s[m]

def _nse(sim, obs):
    sim = np.asarray(sim, float); obs = np.asarray(obs, float)
    if sim.size < 2 or obs.size < 2: return np.nan
    denom = np.sum((obs - np.nanmean(obs))**2)
    if denom == 0: return np.nan
    return 1.0 - np.sum((sim - obs)**2) / denom

def _kge(sim, obs):
    sim = np.asarray(sim, float); obs = np.asarray(obs, float)
    if sim.size < 2 or obs.size < 2: return np.nan
    if _HAS_HE:
        return float(he.evaluator(he.kge, sim, obs))
    sstd, ostd = np.nanstd(sim), np.nanstd(obs)
    if sstd == 0 or ostd == 0: return np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sstd / ostd
    beta = np.nanmean(sim) / (np.nanmean(obs) if np.nanmean(obs)!=0 else np.nan)
    if np.any(np.isnan([r, alpha, beta])): return np.nan
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def _metric_value(sim, obs, metric: str):
    m = metric.lower()
    if m == "nse":    return _nse(sim, obs)
    if m == "kge":    return _kge(sim, obs)
    if m == "lognse":
        sim = np.log(np.clip(sim, 1e-6, None)); obs = np.log(np.clip(obs, 1e-6, None))
        return _nse(sim, obs)
    if m == "logkge":
        sim = np.log(np.clip(sim, 1e-6, None)); obs = np.log(np.clip(obs, 1e-6, None))
        return _kge(sim, obs)
    raise ValueError(f"Unknown metric '{metric}'")

def rolling_metric_monthly(obs: pd.Series, sim: pd.Series, start: str, end: str,
                           window_days: int = 30, metric: str = "nse") -> pd.Series:
    start_ts = pd.to_datetime(start); end_ts = pd.to_datetime(end)
    pad_start = start_ts - pd.Timedelta(days=window_days-1)
    o, s = _align(obs, sim); o = o.loc[pad_start:end_ts].astype(float); s = s.loc[pad_start:end_ts].astype(float)
    df = pd.DataFrame({"obs": o, "sim": s}).dropna()
    if df.empty: return pd.Series(dtype=float)
    eval_points = pd.date_range(start_ts, end_ts, freq="M")
    vals = []
    for t in eval_points:
        w = df.loc[t - pd.Timedelta(days=window_days-1): t]
        if len(w) < max(2, window_days // 3): vals.append(np.nan); continue
        vals.append(_metric_value(w["sim"].values, w["obs"].values, metric))
    return pd.Series(vals, index=eval_points)

def plot_time_by_period_heatmap(
    obs_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    reservoir: str,
    drought_periods: dict,
    metric: str = "nse",
    window_days: int = 30,
    vmin: float = -0.5, vmax: float = 1.0,
    cmap: str = "RdYlGn",
    period_order: list | None = None,
    exclude_patterns: list | None = None,
    show: bool = True,
    save_path: str | None = None
):
    if reservoir not in obs_df.columns or reservoir not in sim_df.columns:
        raise KeyError(f"Reservoir '{reservoir}' not found in obs/sim dataframes.")

    if period_order is None:
        period_order = sorted(drought_periods.keys(),
                              key=lambda k: pd.to_datetime(drought_periods[k][0]))

    if exclude_patterns:
        comps = [re.compile(p) for p in exclude_patterns]
        period_order = [k for k in period_order if not any(c.search(k) for c in comps)]

    rows, labels, max_cols = [], [], 0
    for lab in period_order:
        start, end = drought_periods[lab]
        s = rolling_metric_monthly(obs_df[reservoir], sim_df[reservoir],
                                   start=start, end=end, window_days=window_days, metric=metric)
        rows.append(s.values); labels.append(lab); max_cols = max(max_cols, len(s))
    if max_cols == 0: print(f"No data to plot for {reservoir}."); return None

    heat = np.full((len(rows), max_cols), np.nan)
    for i, r in enumerate(rows): heat[i, :len(r)] = r

    fig, ax = plt.subplots(figsize=(12, 3 + 0.55*len(labels)), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(heat, aspect="auto", origin="upper", norm=norm, cmap=cmap, interpolation="nearest")

    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)

    if max_cols <= 24: major_step = 3
    elif max_cols <= 60: major_step = 6
    else: major_step = 12
    major_ticks = np.arange(0, max_cols, major_step, dtype=int)
    ax.set_xticks(major_ticks); ax.set_xticklabels([f"{m}m" for m in major_ticks], rotation=0)

    ax.set_xticks(np.arange(0, max_cols, 1), minor=True)
    ax.grid(which="minor", axis="x", color="k", alpha=0.08, linewidth=0.5)
    ax.grid(which="major", axis="x", color="k", alpha=0.25, linewidth=0.8)

    for x in range(12, max_cols, 12):
        ax.axvline(x - 0.5, color="k", lw=0.8, alpha=0.2)

    def m_to_y(m): return m / 12.0
    def y_to_m(y): return y * 12.0
    secax = ax.secondary_xaxis("top", functions=(m_to_y, y_to_m))
    max_year = int(np.ceil(max_cols / 12.0))
    secax.set_xticks(np.arange(0, max_year + 1, 1))
    secax.set_xticklabels([f"{y}y" for y in range(0, max_year + 1)])
    secax.set_xlabel("Years since start")

    ax.set_xlabel("Months since period start")
    ax.set_ylabel("Drought / management period")
    pretty_metric = metric.upper()
    ax.set_title(f"{reservoir} — Rolling {pretty_metric} ({window_days}-day window)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(pretty_metric)
    try:
        cbar.set_ticks([vmin, 0.0, 0.5, 0.75, vmax])
        cbar.set_ticklabels([f"{vmin:g}", "0.0 (no skill)", "0.5", "0.75", f"{vmax:g}"])
    except Exception:
        pass

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show: plt.show()
    else: plt.close(fig)
    return fig

# -------------------------------------------------------------------
# 6) Generate figures
# -------------------------------------------------------------------
# 6.1 Release (6-panel) + Storage/Release (9-panel)
for res in reservoirs_of_interest:
    for label, (s, e) in drought_periods.items():
        save_release = f"figures/release/{safe_label(res)}_{safe_label(label)}_release_diag.png"
        plot_reservoir_release_diagnostics(
            reservoir=res, sim_df=df_nwm_downstream_gage, obs_df=df_obs_downstream_gage,
            start=s, end=e, save_path=save_release
        )
        save_sr = f"figures/release_storage/{safe_label(res)}_{safe_label(label)}_stor_release_diag.png"
        plot_reservoir_release_storage_diagnostics(
            reservoir=res,
            sim_df=df_nwm_downstream_gage,
            obs_release_df=df_obs_downstream_gage,
            sim_storage_df=df_nwm_res_storage,
            obs_storage_df=df_obs_res_storage,
            start=s, end=e,
            save_path=save_sr
        )

# 6.2 Rolling skill heatmaps (exclude "Full Period")
metric = "nse"        # try: "kge", "lognse", "logkge"
window_days = 30
base_out = os.path.join("figures", "rolling_skill", safe_label(metric))
os.makedirs(base_out, exist_ok=True)
period_order = sorted(drought_periods.keys(), key=lambda k: pd.to_datetime(drought_periods[k][0]))

for res in reservoirs_of_interest:
    save_path = os.path.join(base_out, f"{safe_label(res)}_rolling_{safe_label(metric)}_{window_days}d.png")
    plot_time_by_period_heatmap(
        obs_df=df_obs_downstream_gage,
        sim_df=df_nwm_downstream_gage,
        reservoir=res,
        drought_periods=drought_periods,
        metric=metric,
        window_days=window_days,
        vmin=-0.5, vmax=1.0,
        cmap="RdYlGn",
        period_order=period_order,
        exclude_patterns=[r"^Full Period:"],  # <-- hide the full-period row
        show=False,
        save_path=save_path
    )

print("All figures written to ./figures/")

# ====== helpers & styling ======
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    _HAS_LOWESS = True
except Exception:
    _HAS_LOWESS = False

SEASON_ORDER  = ["Winter","Spring","Summer","Fall"]
DECADE_ORDER  = ["1980s","1990s","2000s","2010s","2020s"]
DECADE_COLORS = {
    "1980s": "#1f77b4",
    "1990s": "#ff7f0e",
    "2000s": "#2ca02c",
    "2010s": "#d62728",
    "2020s": "#9467bd",
}
SEASON_COLORS = {
    "Winter": "#1f77b4",
    "Spring": "#2ca02c",
    "Summer": "#ff7f0e",
    "Fall":   "#9467bd",
}
SEASON_MAP = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
              6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}

def _season(m): return SEASON_MAP.get(int(m), "NA")
def _decade_label(y): d = (int(y)//10)*10; return f"{d}s"

def _make_residual_frame(obs_s: pd.Series, sim_s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"obs": obs_s}).join(pd.DataFrame({"sim": sim_s}), how="inner").dropna()
    if df.empty: return df
    df["residual"] = df["sim"] - df["obs"]
    df["flow_pct"] = df["obs"].rank(pct=True) * 100.0  # 0..100
    idx = pd.DatetimeIndex(df.index)
    df["year"]   = idx.year
    df["month"]  = idx.month
    df["season"] = df["month"].map(_season)
    df["decade"] = df["year"].map(_decade_label)
    return df

def _robust_sym_limits(y, lo=1.0, hi=99.0, pad=0.08):
    if len(y) == 0: return (-1, 1)
    ql, qh = np.nanpercentile(y, [lo, hi])
    m = max(abs(ql), abs(qh))
    return (-m*(1+pad), m*(1+pad))

def _lowess_line(x_num, y, frac=0.10):
    if len(x_num) < 20: return None, None
    order = np.argsort(x_num)
    x_sorted = x_num[order]; y_sorted = y[order]
    if _HAS_LOWESS:
        sm = _lowess(y_sorted, x_sorted, frac=frac, it=1, return_sorted=True)
        return sm[:,0], sm[:,1]
    # fallback: rolling median
    w = max(11, int(0.05*len(y_sorted)//2*2+1))
    y_med = pd.Series(y_sorted).rolling(w, center=True).median().to_numpy()
    return x_sorted, y_med

# ─────────────────────────────────────────────────────────
# 1) Error time-series with running mean + LOWESS + decade badge
# ─────────────────────────────────────────────────────────
def plot_error_time_series_enhanced(
    df_obs, df_sim, reservoirs, start=None, end=None,
    period_label="", save_folder="figures/error_ts",
    window_days=60, lowess_frac=0.10, acceptable_band=None,
    annotate_decadal_metrics=True, color_points_by_decade=False
):
    os.makedirs(save_folder, exist_ok=True)
    slicer = slice(start, end) if (start or end) else slice(None)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns: continue
        obs = df_obs[res].loc[slicer].dropna()
        sim = df_sim[res].reindex(obs.index)
        df  = _make_residual_frame(obs, sim)
        if df.empty: continue

        ylo, yhi = _robust_sym_limits(df["residual"].values)

        fig, ax = plt.subplots(figsize=(12, 4))
        if color_points_by_decade:
            for dec, sub in df.groupby("decade"):
                ax.plot(sub.index, sub["residual"], lw=0.6,
                        color=DECADE_COLORS.get(dec, "0.5"), label=dec, alpha=0.7)
        else:
            ax.plot(df.index, df["residual"], lw=0.8, color="steelblue",
                    label="Residual (Sim − Obs)")

        if acceptable_band is not None:
            ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.10, zorder=0)
            ax.text(0.005, 0.95, f"±{acceptable_band:g} MGD band",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8, color="gray")

        if window_days and window_days > 1:
            run = df["residual"].rolling(window_days, center=True).mean()
            ax.plot(df.index, run, lw=1.6, color="orange", label=f"{window_days}-day mean")

        if lowess_frac:
            tnum = mdates.date2num(df.index.to_pydatetime())
            xs, ys = _lowess_line(tnum, df["residual"].to_numpy(), frac=lowess_frac)
            if xs is not None:
                ax.plot(mdates.num2date(xs), ys, lw=2.0, color="firebrick", alpha=0.9,
                        label=f"LOWESS (frac={lowess_frac})")

        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"{period_label} — Error Evolution Over Time — {res}", fontsize=13, weight="bold")
        ax.set_ylabel("Error (MGD)"); ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.grid(True, alpha=0.25)

        if annotate_decadal_metrics:
            lines = []
            for decade, g in df.groupby("decade"):
                m = calculate_all_error_metrics(obs.loc[g.index], sim.loc[g.index], timescale="D")
                if m is None or np.isnan(m["nse"]): continue
                lines.append(f"{decade}: NSE={m['nse']:.2f}  KGE={m['kge']:.2f}")
            if lines:
                at = AnchoredText("Decadal skill (daily)\n" + "\n".join(lines),
                                  loc=1, prop=dict(size=9), frameon=True, borderpad=0.4)
                at.patch.set_alpha(0.85); ax.add_artist(at)

        # legend placement
        if color_points_by_decade:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
            fig.tight_layout(rect=[0,0,0.86,1])
        else:
            ax.legend(loc="lower left", framealpha=0.85)
            fig.tight_layout()

        out = f"{save_folder}/error_timeseries_{safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────
# 2) Error vs Flow Percentile — clearer axes + decade color legend
# ─────────────────────────────────────────────────────────
def plot_error_vs_flow_percentile_enhanced(
    df_obs, df_sim, reservoirs, period_label,
    save_folder="figures/error_vs_pct",
    acceptable_band=None, lowess_frac=0.10,
    color_by="decade",     # "decade" or "season"
    alpha=0.25, s=10
):
    os.makedirs(save_folder, exist_ok=True)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns: continue
        df = _make_residual_frame(df_obs[res].dropna(), df_sim[res])
        if df.empty: continue

        x = df["flow_pct"].to_numpy()
        y = df["residual"].to_numpy()
        ylo, yhi = _robust_sym_limits(y)

        fig, ax = plt.subplots(figsize=(9, 6))
        if color_by == "decade":
            for dec in DECADE_ORDER:
                sub = df[df["decade"] == dec]
                if sub.empty: continue
                ax.scatter(sub["flow_pct"], sub["residual"], s=s, alpha=alpha,
                           color=DECADE_COLORS.get(dec, "0.5"), label=dec)
        else:  # season
            for sea in SEASON_ORDER:
                sub = df[df["season"] == sea]
                if sub.empty: continue
                ax.scatter(sub["flow_pct"], sub["residual"], s=s, alpha=alpha,
                           color=SEASON_COLORS.get(sea, "0.5"), label=sea)

        xs, ys = _lowess_line(x, y, frac=lowess_frac)
        if xs is not None:
            ax.plot(xs, ys, lw=2.0, color="black", label="LOWESS trend")

        if acceptable_band is not None:
            ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.10, zorder=0)
            ax.text(0.01, 0.95, f"±{acceptable_band:g} MGD", transform=ax.transAxes,
                    va="top", ha="left", fontsize=8, color="gray")

        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xlim(0, 100); ax.set_ylim(ylo, yhi)
        ax.set_xticks([0,20,40,60,80,100])
        ax.set_title(f"{period_label} — Error vs Flow Percentile — {res}", fontsize=13, weight="bold")
        ax.set_xlabel("Observed Flow Percentile"); ax.set_ylabel("Error (Sim − Obs, MGD)")
        ax.grid(True, alpha=0.25)

        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9)
        fig.tight_layout(rect=[0,0,0.86,1])
        out = f"{save_folder}/error_vs_flow_percentile_{safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────
# 3) Season 4-panel — panels=seasons, dots colored by decade + legend
# ─────────────────────────────────────────────────────────
def plot_seasonal_decadal_panels(
    df_obs, df_sim, reservoirs, period_label,
    save_folder="figures/error_vs_pct_seasons",
    lowess_frac=0.12, acceptable_band=None
):
    os.makedirs(save_folder, exist_ok=True)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns: continue
        df = _make_residual_frame(df_obs[res].dropna(), df_sim[res])
        if df.empty: continue

        # robust y shared across panels
        ylo, yhi = _robust_sym_limits(df["residual"].values)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        axes = axes.ravel()

        for i, sea in enumerate(SEASON_ORDER):
            ax = axes[i]
            sub = df[df["season"] == sea]
            if sub.empty:
                ax.set_title(sea); ax.grid(True, alpha=0.25); continue

            for dec in DECADE_ORDER:
                d2 = sub[sub["decade"] == dec]
                if d2.empty: continue
                ax.scatter(d2["flow_pct"], d2["residual"], s=13, alpha=0.28,
                           color=DECADE_COLORS.get(dec, "0.5"), label=dec)

            xs, ys = _lowess_line(sub["flow_pct"].to_numpy(), sub["residual"].to_numpy(), frac=lowess_frac)
            if xs is not None: ax.plot(xs, ys, lw=2.0, color="black")

            if acceptable_band is not None:
                ax.axhspan(-acceptable_band, acceptable_band, color="grey", alpha=0.08, zorder=0)

            ax.axhline(0, color="black", ls="--", lw=1)
            ax.set_title(sea); ax.set_xlim(0, 100); ax.set_ylim(ylo, yhi)
            ax.grid(True, alpha=0.25)

        for ax in (axes[2], axes[3]): ax.set_xlabel("Observed Flow Percentile")
        for ax in (axes[0], axes[2]): ax.set_ylabel("Error (Sim − Obs, MGD)")
        for ax in axes: ax.set_xticks([0,20,40,60,80,100])

        # decade color legend (shared)
        handles = [Line2D([0],[0], marker='o', color='none',
                          markerfacecolor=DECADE_COLORS[d], markersize=7, label=d)
                   for d in DECADE_ORDER]
        fig.legend(handles, [h.get_label() for h in handles],
                   loc="center left", bbox_to_anchor=(1.02, 0.5),
                   title="Decade", framealpha=0.9)

        fig.suptitle(f"{period_label} — {res}: Residual vs Flow Percentile by Season & Decade",
                     fontsize=14, weight="bold")
        fig.tight_layout(rect=[0,0,0.86,0.96])
        out = f"{save_folder}/error_vs_flow_percentile_season4_{safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────
# 4) Joint density (hexbin) — crisp with marginals
# ─────────────────────────────────────────────────────────
def plot_joint_error_flow_percentile_crisp(
    df_obs, df_sim, reservoirs, period_label,
    save_folder="figures/error_vs_pct_joint_crisp",
    gridsize=50
):
    os.makedirs(save_folder, exist_ok=True)

    for res in reservoirs:
        if res not in df_obs.columns or res not in df_sim.columns: continue
        df = _make_residual_frame(df_obs[res].dropna(), df_sim[res])
        if df.empty or len(df) < 50: continue

        x = df["flow_pct"].to_numpy()
        y = df["residual"].to_numpy()
        ylo, yhi = _robust_sym_limits(y)

        fig = plt.figure(figsize=(9, 7))
        gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.15, wspace=0.15)
        ax  = fig.add_subplot(gs[1:4, 0:3])
        axx = fig.add_subplot(gs[0,   0:3], sharex=ax)
        axy = fig.add_subplot(gs[1:4, 3],   sharey=ax)

        hb = ax.hexbin(x, y, gridsize=gridsize, extent=(0, 100, ylo, yhi),
                       mincnt=2, bins='log', cmap="Blues", linewidths=0)
        cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("log10(Count)")

        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_xlim(0, 100); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("Observed Flow Percentile"); ax.set_ylabel("Error (Sim − Obs, MGD)")
        ax.grid(True, alpha=0.15)
        ax.set_xticks([0,20,40,60,80,100])

        axx.hist(x, bins=40, alpha=0.8, edgecolor="white", linewidth=0.3)
        axy.hist(y, bins=40, orientation="horizontal", alpha=0.8, edgecolor="white", linewidth=0.3)
        axx.set_ylabel("Count"); axy.set_xlabel("Count")
        plt.setp(axx.get_xticklabels(), visible=False)
        plt.setp(axy.get_yticklabels(), visible=False)

        fig.suptitle(f"{period_label} — {res}: Residual vs Flow Percentile (Joint Hexbin)",
                     fontsize=14, weight="bold")
        fig.tight_layout()
        out = f"{save_folder}/error_vs_flow_percentile_joint_{safe_label(res)}.png"
        fig.savefig(out, dpi=300); plt.close(fig)
        print(f"Saved: {out}")

period = "1993–2023 (Stable Policy)"
plot_error_time_series_enhanced(
    df_obs=df_obs_downstream_gage, df_sim=df_nwm_downstream_gage,
    reservoirs=reservoirs_of_interest, start="1993-01-01", end="2023-12-31",
    period_label=period, save_folder="figures/release_error_ts",
    window_days=90, lowess_frac=0.10, acceptable_band=20.0,
    annotate_decadal_metrics=True, color_points_by_decade=False
)

plot_error_vs_flow_percentile_enhanced(
    df_obs=df_obs_downstream_gage, df_sim=df_nwm_downstream_gage,
    reservoirs=reservoirs_of_interest, period_label=period,
    save_folder="figures/release_error_vs_pct",
    acceptable_band=20.0, lowess_frac=0.10, color_by="decade"
)

plot_seasonal_decadal_panels(
    df_obs=df_obs_downstream_gage, df_sim=df_nwm_downstream_gage,
    reservoirs=reservoirs_of_interest, period_label=period,
    save_folder="figures/release_error_vs_pct_seasons",
    lowess_frac=0.12, acceptable_band=20.0
)

plot_joint_error_flow_percentile_crisp(
    df_obs=df_obs_downstream_gage, df_sim=df_nwm_downstream_gage,
    reservoirs=reservoirs_of_interest, period_label=period,
    save_folder="figures/release_error_vs_pct_joint_crisp",
    gridsize=50
)

period_storage = "1993–2023 (Stable Policy) — Storage (MG)"

# 1) Error time series
plot_error_time_series_enhanced(
    df_obs=df_obs_res_storage, df_sim=df_nwm_res_storage,
    reservoirs=reservoirs_of_interest,
    start="1993-01-01", end="2023-12-31",
    period_label=period_storage,
    save_folder="figures/storage_error_ts_MG",
    window_days=90, lowess_frac=0.10,
    acceptable_band=500.0,     # tweak for your units/scale
    annotate_decadal_metrics=True,
    color_points_by_decade=False
)

# 2) Error vs “storage percentile”
plot_error_vs_flow_percentile_enhanced(
    df_obs=df_obs_res_storage, df_sim=df_nwm_res_storage,
    reservoirs=reservoirs_of_interest,
    period_label=period_storage,
    save_folder="figures/storage_error_vs_pct_MG",
    acceptable_band=500.0,
    lowess_frac=0.10,
    color_by="decade"          # or "season"
)

# 3) 4-panel (season × decade)
plot_seasonal_decadal_panels(
    df_obs=df_obs_res_storage, df_sim=df_nwm_res_storage,
    reservoirs=reservoirs_of_interest,
    period_label=period_storage,
    save_folder="figures/storage_error_vs_pct_seasons_MG",
    lowess_frac=0.12,
    acceptable_band=500.0
)

# 4) Joint density (hexbin) + marginals
plot_joint_error_flow_percentile_crisp(
    df_obs=df_obs_res_storage, df_sim=df_nwm_res_storage,
    reservoirs=reservoirs_of_interest,
    period_label=period_storage,
    save_folder="figures/storage_error_vs_pct_joint_crisp_MG",
    gridsize=50
)

# capacities you already provided
RESERVOIR_CAP_MG = {
    "prompton": 27956.02,
    "beltzvilleCombined": 48317.0588,
    "fewalter": 35800.0,
    "blueMarsh": 42320.3544,
}

def as_percent_storage(df, capacities):
    df_pct = df.copy()
    for r, cap in capacities.items():
        if r in df_pct.columns and cap and cap > 0:
            df_pct[r] = (df_pct[r] / cap) * 100.0
    return df_pct

# convert both observed & simulated storage to %
df_obs_res_storage_pct = as_percent_storage(df_obs_res_storage, RESERVOIR_CAP_MG)
df_nwm_res_storage_pct = as_percent_storage(df_nwm_res_storage, RESERVOIR_CAP_MG)

period_storage_pct = "1993–2023 (Stable Policy) — Storage (% of capacity)"

# 1) Error time series (percent points)
plot_error_time_series_enhanced(
    df_obs=df_obs_res_storage_pct, df_sim=df_nwm_res_storage_pct,
    reservoirs=reservoirs_of_interest,
    start="1993-01-01", end="2023-12-31",
    period_label=period_storage_pct,
    save_folder="figures/storage_error_ts_pct",
    window_days=90, lowess_frac=0.10,
    acceptable_band=5.0,      # ±5 percentage points band (edit as needed)
    annotate_decadal_metrics=True
)

# 2) Error vs “storage percentile” (percent units)
plot_error_vs_flow_percentile_enhanced(
    df_obs=df_obs_res_storage_pct, df_sim=df_nwm_res_storage_pct,
    reservoirs=reservoirs_of_interest,
    period_label=period_storage_pct,
    save_folder="figures/storage_error_vs_pct_pct",
    acceptable_band=5.0,
    lowess_frac=0.10,
    color_by="decade"
)

# 3) 4-panel (season × decade) in %
plot_seasonal_decadal_panels(
    df_obs=df_obs_res_storage_pct, df_sim=df_nwm_res_storage_pct,
    reservoirs=reservoirs_of_interest,
    period_label=period_storage_pct,
    save_folder="figures/storage_error_vs_pct_seasons_pct",
    lowess_frac=0.12,
    acceptable_band=5.0
)

# 4) Joint density (hexbin) in %
plot_joint_error_flow_percentile_crisp(
    df_obs=df_obs_res_storage_pct, df_sim=df_nwm_res_storage_pct,
    reservoirs=reservoirs_of_interest,
    period_label=period_storage_pct,
    save_folder="figures/storage_error_vs_pct_joint_crisp_pct",
    gridsize=50
)

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- (safe helper; no-op if you already defined one) ---
def safe_label(s: str) -> str:
    s = s.replace("–", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

# --- shared: color map for periods (chronological & stable) ---
def _period_colors(drought_periods, palette="tab20"):
    # sort by start date
    order = sorted(drought_periods.keys(),
                   key=lambda k: pd.to_datetime(drought_periods[k][0]))
    cmap = plt.get_cmap(palette)
    colors = {p: cmap(i % cmap.N) for i, p in enumerate(order)}
    return order, colors

# --- shared: slice a single column by period label ---
def _slice_period(df: pd.DataFrame, col: str, period_label: str, periods: dict) -> pd.Series:
    start, end = periods[period_label]
    return df[col].loc[start:end].dropna()

# --- shared: FDC XY (exceedance %) ---
def _fdc_xy(series: pd.Series, max_points=None):
    v = np.asarray(series.dropna().values, dtype=float)
    if v.size == 0:
        return None, None
    v = np.sort(v)[::-1]  # high→low
    if max_points and v.size > max_points:
        # thin using quantiles to regularize density
        qs = np.linspace(0, 1, max_points)
        v = np.quantile(v, qs[::-1])  # keep high→low
    p = np.linspace(0, 100, len(v))  # exceedance probability (%)
    return p, v

# ================================================================
# 1) Period-Stacked FDCs (one panel; obs vs sim per period)
# ================================================================
def plot_period_stacked_fdcs(
    df_obs, df_sim, reservoir, drought_periods,
    save_folder="figures/stacked_fdcs", include_regex=None,
    ylog=True, max_points=500, title_suffix=""
):
    os.makedirs(save_folder, exist_ok=True)
    order, colors = _period_colors(drought_periods)

    if include_regex:
        import re as _re
        order = [p for p in order if _re.search(include_regex, p)]

    fig, ax = plt.subplots(figsize=(9, 6))

    any_plotted = False
    for period in order:
        s_obs = _slice_period(df_obs, reservoir, period, drought_periods)
        s_sim = _slice_period(df_sim, reservoir, period, drought_periods)
        if s_obs.empty or s_sim.empty:
            continue

        px_o, vy_o = _fdc_xy(s_obs, max_points=max_points)
        px_s, vy_s = _fdc_xy(s_sim, max_points=max_points)
        if px_o is None or px_s is None:
            continue

        c = colors[period]
        ax.plot(px_o, vy_o, color=c, lw=1.25, label=period)       # Observed (solid)
        ax.plot(px_s, vy_s, color=c, lw=1.25, ls="--", alpha=0.95) # Simulated (dashed)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"[WARN] No data to plot for {reservoir}.")
        return

    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel("Flow")
    ax.grid(True, alpha=0.25)

    # Two-part legend: line style key + period colors
    style_handles = [
        Line2D([0],[0], lw=1.5, color="k", ls="-",  label="Observed"),
        Line2D([0],[0], lw=1.5, color="k", ls="--", label="Simulated"),
    ]
    leg1 = ax.legend(handles=style_handles, loc="upper right", framealpha=0.9)
    ax.add_artist(leg1)
    # Period legend on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.9, title="Period")

    title = f"{reservoir} — Period-Stacked FDCs"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=13, weight="bold")

    out = os.path.join(save_folder, f"{safe_label(reservoir)}_stacked_fdcs.png")
    fig.tight_layout(rect=[0,0,0.86,1])
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"Saved: {out}")


period_filter = r"^Drought"   # or None to include all

for res in reservoirs_of_interest:
    # 1) Period-stacked FDCs
    plot_period_stacked_fdcs(
        df_obs=df_obs_downstream_gage,
        df_sim=df_nwm_downstream_gage,
        reservoir=res,
        drought_periods=drought_periods,
        save_folder="figures/dist/stacked_fdcs",
        include_regex=period_filter,
        ylog=True,
        max_points=600,
        title_suffix="Releases"
    )


# ================================
# Error propagation (reservoir → Montague/Trenton) figure
# ================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- small helpers ---
def _align(obs: pd.Series, sim: pd.Series):
    o, s = obs.align(sim, join="inner")
    m = (~o.isna()) & (~s.isna())
    return o[m].astype(float), s[m].astype(float)

def _fdc(series: pd.Series):
    """Return (exceedance %, sorted values descending)."""
    x = pd.Series(series).dropna().astype(float)
    if x.empty:
        return np.array([]), np.array([])
    vals = np.sort(x.values)[::-1]
    exc  = np.linspace(0, 100, len(vals), endpoint=False)  # 0..<100
    return exc, vals

def _add_metric_box(ax, title, metrics_dict, loc="upper right"):
    """Small anchored text box with metrics (rounded)."""
    if metrics_dict is None: 
        return
    txt = [title]
    for k in ("nse", "kge"):
        v = metrics_dict.get(k, np.nan)
        if v is not None and not np.isnan(v):
            txt.append(f"{k.upper()}={v:.2f}")
    text = "\n".join(txt)
    bbox_props = dict(boxstyle="round,pad=0.35", fc="white", ec="0.4", alpha=0.85)
    xy = dict(upper_right=(0.98,0.98), upper_left=(0.02,0.98),
              lower_right=(0.98,0.02), lower_left=(0.02,0.02))
    x, y = xy.get(loc, (0.98, 0.98))
    ax.text(x, y, text, transform=ax.transAxes, va="top" if "upper" in loc else "bottom",
            ha="right" if "right" in loc else "left", fontsize=9, bbox=bbox_props)

def plot_release_to_downstream_fdcs(
    reservoir: str,
    period_label: str,
    drought_periods: dict,
    df_obs_release: pd.DataFrame,      # df_obs_downstream_gage (columns = reservoirs)
    df_sim_release: pd.DataFrame,      # df_nwm_downstream_gage
    df_obs_major: pd.DataFrame,        # df_obs_major_flow
    df_sim_major: pd.DataFrame,        # df_nwm_major_flow
    montague_col: str = "delMontague",
    trenton_col: str  = "delTrenton",
    ylabel_release: str = "Release (MGD)",
    ylabel_flow: str    = "Flow (MGD)",
    save_folder: str = "figures/error_propagation",
    sharex_fdcs: bool = True
):
    """One 2x2 figure: releases (TS + FDC) and Montague/Trenton FDCs for the *same* period."""
    if period_label not in drought_periods:
        raise KeyError(f"Unknown period '{period_label}'")
    start, end = drought_periods[period_label]

    # Slice data
    # Releases
    obs_r = df_obs_release.get(reservoir, pd.Series(dtype=float)).loc[start:end]
    sim_r = df_sim_release.get(reservoir, pd.Series(dtype=float)).loc[start:end]
    obs_r, sim_r = _align(obs_r, sim_r)

    # Major flows (Montague / Trenton)
    obs_m = df_obs_major.get(montague_col, pd.Series(dtype=float)).loc[start:end]
    sim_m = df_sim_major.get(montague_col, pd.Series(dtype=float)).loc[start:end]
    obs_m, sim_m = _align(obs_m, sim_m)

    obs_t = df_obs_major.get(trenton_col, pd.Series(dtype=float)).loc[start:end]
    sim_t = df_sim_major.get(trenton_col, pd.Series(dtype=float)).loc[start:end]
    obs_t, sim_t = _align(obs_t, sim_t)

    # Metrics
    rel_metrics = calculate_all_error_metrics(obs_r, sim_r, timescale="D") if len(obs_r) > 1 else None
    mon_metrics = calculate_all_error_metrics(obs_m, sim_m, timescale="D") if len(obs_m) > 1 else None
    tre_metrics = calculate_all_error_metrics(obs_t, sim_t, timescale="D") if len(obs_t) > 1 else None

    # FDCs
    e_r_o, f_r_o = _fdc(obs_r); e_r_s, f_r_s = _fdc(sim_r)
    e_m_o, f_m_o = _fdc(obs_m); e_m_s, f_m_s = _fdc(sim_m)
    e_t_o, f_t_o = _fdc(obs_t); e_t_s, f_t_s = _fdc(sim_t)

    # Figure
    os.makedirs(save_folder, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_ts, ax_fdc_r = axes[0,0], axes[0,1]
    ax_fdc_m, ax_fdc_t = axes[1,0], axes[1,1]

    # ---- (1) Time series: reservoir release ----
    if len(obs_r):
        ax_ts.plot(obs_r.index, obs_r.values, color="black", lw=1.4, label="Observed")
    if len(sim_r):
        ax_ts.plot(sim_r.index, sim_r.values, color="tab:blue", lw=1.2, ls="--", label="Simulated")
    ax_ts.set_title(f"{reservoir} — Releases (Daily) • {period_label}", fontsize=12, weight="bold")
    ax_ts.set_ylabel(ylabel_release); ax_ts.grid(True, alpha=0.25)
    ax_ts.legend(loc="upper left", framealpha=0.9)
    _add_metric_box(ax_ts, "Releases", rel_metrics, loc="upper right")

    # ---- (2) FDC: reservoir release ----
    if len(f_r_o): ax_fdc_r.plot(e_r_o, f_r_o, color="black", lw=1.4, label="Obs")
    if len(f_r_s): ax_fdc_r.plot(e_r_s, f_r_s, color="tab:blue", lw=1.2, ls="--", label="Sim")
    ax_fdc_r.set_title(f"{reservoir} — Releases FDC", fontsize=12, weight="bold")
    ax_fdc_r.set_xlabel("Exceedance probability (%)"); ax_fdc_r.set_ylabel(ylabel_release)
    ax_fdc_r.set_yscale("log"); ax_fdc_r.grid(True, alpha=0.25)
    ax_fdc_r.legend(loc="upper right", framealpha=0.9)

    # ---- (3) FDC: Montague ----
    if len(f_m_o): ax_fdc_m.plot(e_m_o, f_m_o, color="black", lw=1.4, label="Obs")
    if len(f_m_s): ax_fdc_m.plot(e_m_s, f_m_s, color="tab:blue", lw=1.2, ls="--", label="Sim")
    ax_fdc_m.set_title("Montague — Flow FDC", fontsize=12, weight="bold")
    ax_fdc_m.set_xlabel("Exceedance probability (%)"); ax_fdc_m.set_ylabel(ylabel_flow)
    ax_fdc_m.set_yscale("log"); ax_fdc_m.grid(True, alpha=0.25)
    _add_metric_box(ax_fdc_m, "Montague", mon_metrics, loc="upper right")

    # ---- (4) FDC: Trenton ----
    if len(f_t_o): ax_fdc_t.plot(e_t_o, f_t_o, color="black", lw=1.4, label="Obs")
    if len(f_t_s): ax_fdc_t.plot(e_t_s, f_t_s, color="tab:blue", lw=1.2, ls="--", label="Sim")
    ax_fdc_t.set_title("Trenton — Flow FDC", fontsize=12, weight="bold")
    ax_fdc_t.set_xlabel("Exceedance probability (%)"); ax_fdc_t.set_ylabel(ylabel_flow)
    ax_fdc_t.set_yscale("log"); ax_fdc_t.grid(True, alpha=0.25)
    _add_metric_box(ax_fdc_t, "Trenton", tre_metrics, loc="upper right")

    # Align FDC x axes
    if sharex_fdcs:
        ax_fdc_m.sharex(ax_fdc_r)
        ax_fdc_t.sharex(ax_fdc_r)
        ax_fdc_r.set_xlim(0, 100)

    fig.suptitle(f"Error Propagation Snapshot — {reservoir} → (Montague, Trenton) • {period_label}",
                 fontsize=14, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])

    out = os.path.join(save_folder, f"{safe_label(reservoir)}_{safe_label(period_label)}_propagation.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print(f"Saved: {out}")

# Example: run for all reservoirs across your drought periods
for res in reservoirs_of_interest:
    for period_label in drought_periods.keys():
        plot_release_to_downstream_fdcs(
            reservoir=res,
            period_label=period_label,
            drought_periods=drought_periods,
            df_obs_release=df_obs_downstream_gage,
            df_sim_release=df_nwm_downstream_gage,
            df_obs_major=df_obs_major_flow,
            df_sim_major=df_nwm_major_flow,
            montague_col="delMontague",
            trenton_col="delTrenton",
            save_folder="figures/error_propagation"
        )


# ==== Trenton activation with LB contributions (full replacement) ====
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------- small helpers ----------
def safe_label(s: str) -> str:
    s = s.replace("–", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _period_slice(df: pd.DataFrame | pd.Series, start, end):
    return df.loc[pd.to_datetime(start):pd.to_datetime(end)]

def _lb_cols(df: pd.DataFrame):
    """Lower-basin contributions that feed Trenton MRF (mrf_trenton_*)."""
    return [c for c in df.columns if re.match(r"^mrf_trenton_", c)]

# ---------- single-figure maker ----------
def plot_trenton_with_lb_activation_one(
    df_contrib: pd.DataFrame,
    df_trenton_sim: pd.DataFrame,
    df_trenton_obs: pd.DataFrame | None,
    period_label: str,
    drought_periods: dict,
    save_folder: str = "figures/lb_activation",
    ylimit_trenton: tuple | None = None,
    legend_ncols: int = 3,
):
    """
    Create a 2-row figure for one period:
      Row 1: Trenton flow (Sim & Obs if available)
      Row 2: Stacked lower-basin contributions to Trenton MRF (mrf_trenton_*)

    Saves: figures/lb_activation/trenton_lb_activation_<period>.png
    """
    start, end = drought_periods[period_label]
    cols = _lb_cols(df_contrib)
    if not cols:
        print(f"[{period_label}] No mrf_trenton_* columns found."); 
        return

    os.makedirs(save_folder, exist_ok=True)

    # Period slices
    c = _period_slice(df_contrib[cols], start, end).astype(float).fillna(0.0)
    t_sim = _period_slice(df_trenton_sim["delTrenton"], start, end).astype(float)

    if df_trenton_obs is not None and "delTrenton" in df_trenton_obs.columns:
        t_obs = _period_slice(df_trenton_obs["delTrenton"], start, end).astype(float)
    else:
        t_obs = None

    # Align on common index across contributions and whichever Trenton series exist
    idx = c.index
    idx = idx.intersection(t_sim.index)
    if t_obs is not None:
        idx = idx.intersection(t_obs.index)

    if len(idx) < 10:
        print(f"[{period_label}] Too few points to plot.")
        return

    c = c.reindex(idx)
    t_sim = t_sim.reindex(idx)
    if t_obs is not None:
        t_obs = t_obs.reindex(idx)

    # ---- figure ----
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]}
    )

    # Row 1: Trenton flow (Sim + Obs)
    ax0.plot(t_sim.index, t_sim.values, lw=1.3, linestyle="--", label="Trenton (Sim)")
    if t_obs is not None:
        ax0.plot(t_obs.index, t_obs.values, lw=1.6, label="Trenton (Obs)")

    ax0.set_title(
        f"{period_label} — Trenton Flow & Lower-Basin Contributions (activation)",
        fontsize=13, weight="bold"
    )
    ax0.set_ylabel("Flow at Trenton (MGD)")
    if ylimit_trenton:
        ax0.set_ylim(*ylimit_trenton)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper left", framealpha=0.9, ncols=2)

    # Nice x ticks
    ax0.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Row 2: Stacked contributions
    labels = [re.sub(r"^mrf_trenton_", "", k) for k in cols]
    ax1.stackplot(c.index, [c[k].values for k in cols], labels=labels)
    ax1.set_ylabel("LB Contribution (MGD)")
    ax1.set_xlabel("Date")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", ncols=min(legend_ncols, len(cols)), framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(
        save_folder,
        f"trenton_lb_activation_{safe_label(period_label)}.png"
    )
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

# ---------- batch driver over periods ----------
def run_trenton_lb_activation_all_periods(
    df_contrib: pd.DataFrame,
    df_trenton_sim: pd.DataFrame,
    df_trenton_obs: pd.DataFrame | None,
    drought_periods: dict,
    wanted_prefixes: tuple = ("Drought",),  # only droughts by default
    save_folder: str = "figures/lb_activation",
    legend_ncols: int = 3,
):
    """
    Loops periods in chronological order and generates the activation plots.
    """
    # choose periods by prefix
    labels = [
        k for k in drought_periods
        if k.startswith(wanted_prefixes)
    ]
    # chronological order by start date
    labels = sorted(labels, key=lambda k: pd.to_datetime(drought_periods[k][0]))

    if not labels:
        print("No matching periods found for prefixes:", wanted_prefixes)
        return

    for lab in labels:
        plot_trenton_with_lb_activation_one(
            df_contrib=df_contrib,
            df_trenton_sim=df_trenton_sim,
            df_trenton_obs=df_trenton_obs,
            period_label=lab,
            drought_periods=drought_periods,
            save_folder=save_folder,
            legend_ncols=legend_ncols,
        )

# ========================
# Example call:
# ========================
# Use your loaded dataframes:
#   df_lower_basin_mrf_contributions
#   df_nwm_major_flow      (SIM: contains 'delTrenton')
#   df_obs_major_flow      (OBS: contains 'delTrenton')
#   drought_periods        (dict)

run_trenton_lb_activation_all_periods(
    df_contrib=df_lower_basin_mrf_contributions,
    df_trenton_sim=df_nwm_major_flow,
    df_trenton_obs=df_obs_major_flow,   # pass None if you don't have obs
    drought_periods=drought_periods,
    wanted_prefixes=("Drought", "Management"),  #("Drought",),       # or ("Drought","Management")
    save_folder="figures/lb_activation",
    legend_ncols=3
)
