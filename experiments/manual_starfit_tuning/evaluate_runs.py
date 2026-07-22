"""
Evaluate pywrdrb runs against observations for the STARFIT tuning reservoirs.

Usage:
    python evaluate_runs.py run_label1 [run_label2 ...]

Each run_label must correspond to output/{run_label}.hdf5. Produces a metric
table (printed + saved to output/metrics_<labels>.csv) and comparison figures
in figures/.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils.tuning_utils import (
    RESERVOIRS, load_observations, load_default_params, get_starfit_row_name,
    get_obs_storage, get_obs_downstream_flow, load_sim_results,
    get_sim_storage, get_sim_release,
)
from utils.metrics import storage_metrics, release_metrics, summarize_metrics
from pywrdrb.path_manager import get_pn_object

pn = get_pn_object()

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
OBS_START, OBS_END = "2004-01-01", "2023-12-31"


def get_obs_prompton_release(start=None, end=None):
    """Observed Prompton total release (MGD) from supplemental gauge 01429000."""
    path = pn.observations.get_str() + os.sep + "gage_flow_mgd.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "01429000" not in df.columns:
        return None
    return df["01429000"].loc[start:end].dropna()


def evaluate(run_labels, obs_start=OBS_START, obs_end=OBS_END, quiet=False):
    """Compute metric tables for a list of run labels. Returns DataFrame."""
    obs = load_observations()
    params = load_default_params()
    caps = {
        r: params.loc[get_starfit_row_name(r), "Adjusted_CAP_MG"] for r in RESERVOIRS
    }

    sim = load_sim_results([OUTPUT_DIR / f"{lbl}.hdf5" for lbl in run_labels])

    all_metrics = {}
    for lbl in run_labels:
        res_metrics = {}
        for r in RESERVOIRS:
            md = {}
            # --- storage ---
            obs_s = get_obs_storage(obs, r, obs_start, obs_end)
            sim_s = get_sim_storage(sim, lbl, r, obs_start, obs_end)
            sm = storage_metrics(sim_s, obs_s, caps[r])
            md.update({f"storage_{k}": v for k, v in sm.items()})

            # --- release / downstream flow ---
            if r == "prompton":
                obs_q = get_obs_prompton_release(obs_start, obs_end)
                sim_q = get_sim_release(sim, lbl, r, obs_start, obs_end)
            else:
                obs_q = get_obs_downstream_flow(obs, r, obs_start, obs_end)
                sim_q = sim.reservoir_downstream_gage[lbl][0][r].copy()
                sim_q.index = pd.to_datetime(sim_q.index)
                sim_q = sim_q.loc[obs_start:obs_end]
            if obs_q is not None and len(obs_q) > 0:
                rm = release_metrics(sim_q, obs_q)
                md.update({f"release_{k}": v for k, v in rm.items()})
            res_metrics[r] = md
        all_metrics[lbl] = res_metrics

    table = summarize_metrics(all_metrics)
    if not quiet:
        with pd.option_context("display.max_rows", 200, "display.width", 160,
                               "display.float_format", "{:0.3f}".format):
            print(table)
    tag = "_".join(run_labels)[:80]
    out_csv = OUTPUT_DIR / f"metrics_{tag}.csv"
    table.to_csv(out_csv)
    if not quiet:
        print(f"\nSaved: {out_csv}")
    return table


# Headline metrics for quick comparison across iterations
HEADLINE = [
    ("storage_nse", "Storage NSE"),
    ("storage_kge", "Storage KGE"),
    ("storage_pbias", "Storage PBIAS%"),
    ("release_nse_m", "Release NSE (monthly)"),
    ("release_lognse_d", "Release logNSE (daily)"),
    ("release_kge_d", "Release KGE (daily)"),
    ("release_pbias", "Release PBIAS%"),
]


def headline_table(table):
    """Compact view: headline metrics only."""
    idx = [m for m, _ in HEADLINE]
    sub = table[table.index.get_level_values("metric").isin(idx)]
    return sub


if __name__ == "__main__":
    labels = sys.argv[1:] or ["tuned_v007", "baseline_default"]
    t = evaluate(labels)
    print("\n=== Headline metrics ===")
    with pd.option_context("display.max_rows", 100, "display.width", 160,
                           "display.float_format", "{:0.3f}".format):
        print(headline_table(t))
