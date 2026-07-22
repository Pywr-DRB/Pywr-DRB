"""
Metrics for assessing STARFIT parameter quality against observations.

Storage metrics are computed on storage as a fraction of capacity so that
reservoirs of different sizes are comparable. Release metrics are computed on
daily and monthly-mean series; monthly aggregation reflects the seasonal rule
quality without being dominated by individual flood peaks.
"""
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Elemental metrics
# --------------------------------------------------------------------------
def _align(sim, obs):
    """Align two series on common index, dropping NaNs."""
    df = pd.concat([sim.rename("sim"), obs.rename("obs")], axis=1).dropna()
    return df["sim"], df["obs"]


def nse(sim, obs):
    """Nash-Sutcliffe efficiency."""
    sim, obs = _align(sim, obs)
    if len(obs) < 2:
        return np.nan
    return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)


def log_nse(sim, obs, eps=None):
    """NSE of log-transformed series (low-flow emphasis)."""
    sim, obs = _align(sim, obs)
    if len(obs) < 2:
        return np.nan
    if eps is None:
        eps = max(obs[obs > 0].min() * 0.1, 1e-6)
    return nse(np.log(sim + eps), np.log(obs + eps))


def kge(sim, obs, return_components=False):
    """Kling-Gupta efficiency (2009)."""
    sim, obs = _align(sim, obs)
    if len(obs) < 2:
        return (np.nan, np.nan, np.nan, np.nan) if return_components else np.nan
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    k = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    if return_components:
        return k, r, alpha, beta
    return k


def pbias(sim, obs):
    """Percent bias (positive = sim overestimates)."""
    sim, obs = _align(sim, obs)
    if len(obs) == 0:
        return np.nan
    return 100.0 * (sim.sum() - obs.sum()) / obs.sum()


def rmse(sim, obs):
    sim, obs = _align(sim, obs)
    if len(obs) == 0:
        return np.nan
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def quantile_errors(sim, obs, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Relative error of sim vs obs quantiles (flow-duration-curve match)."""
    sim, obs = _align(sim, obs)
    if len(obs) == 0:
        return {q: np.nan for q in quantiles}
    return {
        q: float((sim.quantile(q) - obs.quantile(q)) / max(obs.quantile(q), 1e-9))
        for q in quantiles
    }


# --------------------------------------------------------------------------
# Composite evaluation
# --------------------------------------------------------------------------
def storage_metrics(sim_storage, obs_storage, capacity_mg):
    """
    Metrics for a storage trace (MG). Computed on fraction-of-capacity.

    Returns a dict with keys: nse, kge, pbias, rmse_frac, mean_abs_err_frac.
    """
    sim_f = sim_storage / capacity_mg
    obs_f = obs_storage / capacity_mg
    out = {
        "nse": nse(sim_f, obs_f),
        "kge": kge(sim_f, obs_f),
        "pbias": pbias(sim_f, obs_f),
        "rmse_frac": rmse(sim_f, obs_f),
    }
    s, o = _align(sim_f, obs_f)
    out["mae_frac"] = float((s - o).abs().mean()) if len(o) else np.nan
    out["n_days"] = len(o)
    return out


def release_metrics(sim_release, obs_release):
    """
    Metrics for a release trace (MGD): daily and monthly-mean skill.

    Returns a dict with keys: nse_d, lognse_d, kge_d, pbias, nse_m, kge_m,
    plus 5th/50th/95th percentile relative errors.
    """
    out = {
        "nse_d": nse(sim_release, obs_release),
        "lognse_d": log_nse(sim_release, obs_release),
        "kge_d": kge(sim_release, obs_release),
        "pbias": pbias(sim_release, obs_release),
    }
    s, o = _align(sim_release, obs_release)
    if len(o) > 60:
        sm = s.resample("MS").mean()
        om = o.resample("MS").mean()
        out["nse_m"] = nse(sm, om)
        out["kge_m"] = kge(sm, om)
    else:
        out["nse_m"] = np.nan
        out["kge_m"] = np.nan
    qe = quantile_errors(s, o, quantiles=(0.05, 0.5, 0.95))
    out["q05_relerr"] = qe[0.05]
    out["q50_relerr"] = qe[0.5]
    out["q95_relerr"] = qe[0.95]
    out["n_days"] = len(o)
    return out


def summarize_metrics(metrics_by_run):
    """
    Build a tidy comparison table.

    Parameters
    ----------
    metrics_by_run : dict
        {run_label: {reservoir: {metric: value}}}

    Returns
    -------
    pd.DataFrame with MultiIndex (reservoir, metric) and one column per run.
    """
    frames = {}
    for run, res_dict in metrics_by_run.items():
        rows = {}
        for res, md in res_dict.items():
            for metric, val in md.items():
                rows[(res, metric)] = val
        frames[run] = pd.Series(rows)
    df = pd.DataFrame(frames)
    df.index.names = ["reservoir", "metric"]
    return df
