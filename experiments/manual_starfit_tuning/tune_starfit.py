"""
Fit-and-screen driver for STARFIT tuning.

Fits NOR bands to observed storage climatology and release harmonics by mass
balance, generates candidate parameter sets over a small grid of choices
(inflow-response p2, band quantiles), and screens them with the offline
simulator against observations.

Usage:
    python tune_starfit.py            # fit + screen, prints ranked candidates
    python tune_starfit.py --tag v002 # also write the best CSV as custom_starfit_v002.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.tuning_utils import (
    RESERVOIRS, load_default_params, apply_overrides, write_custom_csv,
    get_starfit_row_name, run_offline_sim,
)
from utils.fit_starfit import (
    DOY, doy_climatology, fit_nor_bound, nor_curve, fit_release_harmonic,
    build_override,
)
from utils.metrics import nse, log_nse, kge, pbias
from pywrdrb.path_manager import get_pn_object
from pywrdrb.utils.constants import cfs_to_mgd

pn = get_pn_object()
HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"

INFLOW_TYPE = "pub_nhmv10_BC_withObsScaled"
SIM_START, SIM_END = "1983-10-01", "2023-12-31"
OBS_START, OBS_END = "2004-01-01", "2023-12-31"

# Downstream gauge node in the dataset whose marginal catchment inflow is
# added to the offline release to compare against the observed gauge.
DOWNSTREAM_GAUGE_NODE = {
    "fewalter": "01447800",
    "beltzvilleCombined": "01449800",
    "blueMarsh": "01470960",
    "prompton": None,  # obs gauge 01429000 measures the release directly
}

# Capacity corrections (MG). Observed beltzville normal pool (~14,000 MG)
# exceeds the current Adjusted_CAP_MG (13,500); use the storage-curve-derived
# observed flood maximum instead so the observed pool is attainable.
CAP_OVERRIDE = {"beltzvilleCombined": 17750.0}

# Band quantiles for (NORlo, NORhi) targets
BAND_QUANTILES = {
    "fewalter": (0.20, 0.85),
    "beltzvilleCombined": (0.10, 0.90),
    "blueMarsh": (0.10, 0.90),
    "prompton": (0.15, 0.85),
}

# Effective release bounds for prompton (MGD). R_min ~ observed late-summer
# low releases (~7 MGD = 10.8 cfs); R_max ~ observed high controlled releases.
PROMPTON_R_MIN_MGD = 7.0
PROMPTON_R_MAX_MGD = 300.0


def load_inputs():
    obs_storage = pd.read_csv(
        pn.observations.get_str() + os.sep + "reservoir_storage_mg.csv",
        index_col=0, parse_dates=True,
    )
    obs_gage = pd.read_csv(
        pn.observations.get_str() + os.sep + "gage_flow_mgd.csv",
        index_col=0, parse_dates=True,
    )
    inflows = pd.read_csv(
        str(pn.sc.get(f"flows/{INFLOW_TYPE}") / "catchment_inflow_mgd.csv"),
        index_col=0, parse_dates=True,
    )
    return obs_storage, obs_gage, inflows


def fit_reservoir(reservoir, obs_storage, inflows, default_params,
                  p1=0.183, p2=0.732, band_quantiles=None, cap_override=None):
    """Fit one reservoir; returns an override dict."""
    row = get_starfit_row_name(reservoir)
    cap = cap_override or float(default_params.loc[row, "Adjusted_CAP_MG"])

    s_obs = obs_storage[reservoir].dropna()
    q_lo, q_hi = band_quantiles or BAND_QUANTILES[reservoir]

    # NOR band targets in percent of (possibly corrected) capacity
    lo_curve = doy_climatology(s_obs, q=q_lo) / cap * 100
    hi_curve = doy_climatology(s_obs, q=q_hi) / cap * 100
    # keep a minimum band width of 0.6% so the reservoir can ride in-band
    hi_curve = np.maximum(hi_curve, lo_curve + 0.6)

    nor_lo = fit_nor_bound(lo_curve)
    nor_hi = fit_nor_bound(hi_curve)

    # dataset inflow climatology + mean
    inflow = inflows[reservoir].loc[OBS_START:OBS_END]
    I_bar = float(inflow.mean())
    I_clim = doy_climatology(inflow, stat="mean")

    # target storage trajectory = observed median climatology (MG)
    S_target = doy_climatology(s_obs, stat="median")

    lo_frac = nor_curve(**{k: nor_lo[k] for k in ("mu", "alpha", "beta")},
                        mn=nor_lo["min"], mx=nor_lo["max"]) / 100
    hi_frac = nor_curve(**{k: nor_hi[k] for k in ("mu", "alpha", "beta")},
                        mn=nor_hi["min"], mx=nor_hi["max"]) / 100

    # effective R_min/R_max for clipping the mass-balance target
    if reservoir == "prompton":
        r_min, r_max = PROMPTON_R_MIN_MGD, PROMPTON_R_MAX_MGD
        release_min = r_min / I_bar - 1.0
        release_max = r_max / I_bar - 1.0
    else:
        from pywrdrb.parameters.lower_basin_ffmp import (
            conservation_releases, max_discharges,
        )
        r_min = conservation_releases[reservoir]
        r_max = max_discharges[reservoir]
        release_min = release_max = None

    rel = fit_release_harmonic(
        I_clim, S_target, lo_frac, hi_frac, I_bar, cap,
        p1=p1, p2=p2, r_min=r_min, r_max=r_max,
    )
    return build_override(
        nor_hi, nor_lo, rel, I_bar,
        cap_mg=cap_override, p1=p1, p2=p2,
        release_min=release_min, release_max=release_max,
    )


def offline_score(offline, obs_storage, obs_gage, inflows, default_params,
                  cap_overrides=None):
    """Metrics per reservoir for one offline run. Returns DataFrame."""
    cap_overrides = cap_overrides or {}
    rows = {}
    for r in RESERVOIRS:
        cap_norm = float(default_params.loc[get_starfit_row_name(r), "Adjusted_CAP_MG"])
        sim = offline[r]
        obs_s = obs_storage[r].dropna().loc[OBS_START:OBS_END]
        sim_s = sim["storage"].loc[OBS_START:OBS_END]
        md = {
            "storage_nse": nse(sim_s / cap_norm, obs_s / cap_norm),
            "storage_mae_pct": float(
                (sim_s / cap_norm - obs_s / cap_norm).dropna().abs().mean() * 100
            ),
        }
        # release proxy
        gauge_node = DOWNSTREAM_GAUGE_NODE[r]
        sim_q = sim["release"].loc[OBS_START:OBS_END]
        if gauge_node is None:
            obs_q = obs_gage["01429000"].dropna().loc[OBS_START:OBS_END]
        else:
            obs_col = r  # downstream gage obs stored under reservoir name? no:
            obs_q = obs_gage[gauge_node].dropna().loc[OBS_START:OBS_END]
            sim_q = sim_q + inflows[gauge_node].loc[OBS_START:OBS_END]
        md["release_lognse_d"] = log_nse(sim_q, obs_q)
        md["release_nse_m"] = nse(sim_q.resample("MS").mean(), obs_q.resample("MS").mean())
        md["release_kge_d"] = kge(sim_q, obs_q)
        md["release_pbias"] = pbias(sim_q, obs_q)
        aligned = pd.concat([sim_q.rename("s"), obs_q.rename("o")], axis=1).dropna()
        for q in (0.05, 0.5, 0.95):
            md[f"release_q{int(q*100):02d}_relerr"] = float(
                (aligned["s"].quantile(q) - aligned["o"].quantile(q))
                / max(aligned["o"].quantile(q), 1e-9)
            )
        rows[r] = md
    return pd.DataFrame(rows).T


def composite(df):
    """Composite score per reservoir (higher better)."""
    return (
        0.45 * df["storage_nse"].clip(lower=-1)
        + 0.25 * df["release_lognse_d"].clip(lower=-1)
        + 0.20 * df["release_nse_m"].clip(lower=-1)
        - 0.10 * df["release_q05_relerr"].abs()
    )


if __name__ == "__main__":
    tag = None
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]

    obs_storage, obs_gage, inflows = load_inputs()
    default_params = load_default_params()

    results = {}
    candidates = {}
    for p2 in (0.5, 0.732, 0.9):
        overrides = {
            r: fit_reservoir(
                r, obs_storage, inflows, default_params, p2=p2,
                cap_override=CAP_OVERRIDE.get(r),
            )
            for r in RESERVOIRS
        }
        params = apply_overrides(default_params, overrides)
        csv = write_custom_csv(params, OUTPUT_DIR, tag=f"screen_p2_{p2}")
        offline = run_offline_sim(INFLOW_TYPE, SIM_START, SIM_END, csv_path=csv)
        score_df = offline_score(offline, obs_storage, obs_gage, inflows,
                                 default_params)
        score_df["composite"] = composite(score_df)
        results[f"p2={p2}"] = score_df
        candidates[f"p2={p2}"] = overrides
        print(f"\n===== candidate p2={p2} =====")
        with pd.option_context("display.width", 200, "display.float_format",
                               "{:0.3f}".format):
            print(score_df)

    # pick best p2 per reservoir
    best = {}
    for r in RESERVOIRS:
        scores = {k: results[k].loc[r, "composite"] for k in results}
        best_k = max(scores, key=scores.get)
        best[r] = (best_k, candidates[best_k][r])
        print(f"best for {r}: {best_k} (composite {scores[best_k]:.3f})")

    if tag:
        overrides = {r: best[r][1] for r in RESERVOIRS}
        params = apply_overrides(default_params, overrides)
        csv = write_custom_csv(params, OUTPUT_DIR, tag=tag)
        print(f"\nWrote {csv}")
        import json
        with open(OUTPUT_DIR / f"overrides_{tag}.json", "w") as f:
            json.dump({r: best[r][1] for r in RESERVOIRS}, f, indent=2)
